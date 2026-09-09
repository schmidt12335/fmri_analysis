"""
Continue "normal" preprocessing (as done in mol_fMRI_v5.3L_newcoreg.ipynb) on an
already-stitched, already motion-corrected functional series produced by
electric_stim_stitching.py.

Scope
-----
This script starts from mc_stitched_func.nii.gz (already motion-corrected by
electric_stim_stitching.py) and performs, in order:

  1. Functional masking (reuses/copies an existing single-scan mask if the
     scans are back-to-back with unchanged geometry, or falls back to manual
     fsleyes drawing).
  2. Spatial smoothing (fslmaths gaussian kernel) - produced as a preprocessed
     output, not used further by this script.
  3. Mean functional image (whole series), masked, for use as the
     coregistration source.
  4. Coregistration to the structural image using the "new" rigid-body,
     normalized-mutual-information AFNI 3dAllineate approach from
     mol_fMRI_v5.3L_newcoreg.ipynb (coregistration_afni, NOT the old
     coregistration_afni_old/"crU" cost version), estimating only (no SCM to
     apply the transform to).
  5. Structural intensity masking of the coregistered mean functional image.


Usage
-----
python preprocess_stitched_scan.py \
    --stitched_dir /path/to/stitched_EPI \
    --struct_dir /path/to/14anatomy \
    --mask_source_dir /path/to/33functionalEPI

Expected inputs
----------------
--stitched_dir must already contain (produced by electric_stim_stitching.py):
    mc_stitched_func.nii.gz, middle_vol.nii.gz
--struct_dir must already contain a converted structural image, ideally
    cleaned_struct.nii.gz (falls back to struct.nii.gz if not present).
--mask_source_dir (optional) is an existing single-scan functional directory
    (e.g. 33functionalEPI) containing mask_mean_mc_func.nii.gz and
    mask_mean_mc_func_cannulas.nii.gz to reuse as-is. Only valid if the
    stitched scans share identical geometry with that scan (true for
    back-to-back scans with unchanged acquisition parameters).
"""

import argparse
import datetime
import getpass
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
from nipype.interfaces import afni, fsl

# Force compressed NIfTI output regardless of the shell's FSLOUTPUTTYPE setting.
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    NOTIFICATION = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message, color):
    line = "*" * 134
    print()
    print(f"{color}{line}{bcolors.ENDC}")
    print(f"{color}{message.center(len(line))}{bcolors.ENDC}")
    print(f"{color}{line}{bcolors.ENDC}")
    print()


def print_statement(message, color):
    print(f"{color}{message}{bcolors.ENDC}")


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return result.stdout.strip()


def masking_file(input_file, mask_file, output_file):
    math = fsl.maths.ApplyMask()
    math.inputs.in_file = input_file
    math.inputs.mask_file = mask_file
    math.inputs.out_file = output_file
    math.run()
    print_statement(f"[OK] Masked file saved -> {output_file}", bcolors.OKGREEN)
    return output_file


def spatial_smoothing(input_file, output_file, fwhm):
    cmd = ["fslmaths", input_file, "-kernel", "gauss", str(fwhm), "-fmean", output_file]
    subprocess.run(cmd, check=True)
    print_statement(f"[OK] Spatial smoothing complete -> {output_file}", bcolors.OKGREEN)


def compute_mean_range(input_file, prefix, start_idx, end_idx):
    afni_cmd = ["3dTstat", "-mean", "-prefix", prefix, f"{input_file}[{start_idx}..{end_idx}]"]
    print_statement(f"[INFO] Running: {' '.join(afni_cmd)}", bcolors.OKBLUE)
    subprocess.run(afni_cmd, check=True)
    print_statement(f"[OK] Mean image saved -> {prefix}", bcolors.OKGREEN)


def mean_image(input_file, output_file):
    mean = fsl.maths.MeanImage()
    mean.inputs.in_file = input_file
    mean.inputs.out_file = output_file
    mean.run()
    print_statement(f"[OK] Mean image saved -> {output_file}", bcolors.OKGREEN)
    return output_file


def create_intensity_mask(input_img, output_mask, low_percentile, high_percentile):
    if not (0 <= low_percentile < high_percentile <= 100):
        raise ValueError("Percentiles must satisfy: 0 <= low < high <= 100")

    p_low = float(run_cmd(["fslstats", input_img, "-l", "0.0001", "-P", str(low_percentile)]))
    p_high = float(run_cmd(["fslstats", input_img, "-l", "0.0001", "-P", str(high_percentile)]))

    subprocess.run(["fslmaths", input_img, "-thr", str(p_low), "-uthr", str(p_high), "-bin", output_mask], check=True)
    print_statement(f"[OK] Intensity mask created -> {output_mask} (thresholds {p_low:.3f}-{p_high:.3f})", bcolors.OKGREEN)


def shrink_mask_xz_linewise(in_mask, out_mask, trim_x=2, trim_z=2):
    img = nib.load(in_mask)
    data = img.get_fdata()

    if data.ndim != 3:
        raise ValueError("Input mask must be 3D")

    X, Y, Z = data.shape
    mask = data.copy()

    tmp_mask = np.zeros_like(mask, dtype=mask.dtype)
    final_mask = np.zeros_like(mask, dtype=mask.dtype)

    for y in range(Y):
        for z in range(Z):
            line = mask[:, y, z]
            idx = np.where(line > 0)[0]
            if idx.size < 2 * trim_x + 1:
                continue
            x_min = idx.min() + trim_x
            x_max = idx.max() - trim_x
            tmp_mask[x_min:x_max + 1, y, z] = mask[x_min:x_max + 1, y, z]

    for y in range(Y):
        for x in range(X):
            line = tmp_mask[x, y, :]
            idx = np.where(line > 0)[0]
            if idx.size < 2 * trim_z + 1:
                continue
            z_min = idx.min() + trim_z
            z_max = idx.max() - trim_z
            final_mask[x, y, z_min:z_max + 1] = tmp_mask[x, y, z_min:z_max + 1]

    nib.save(nib.Nifti1Image(final_mask, img.affine, img.header), out_mask)
    print_statement(f"[OK] Line-wise shrunk mask saved -> {out_mask}", bcolors.OKGREEN)


def coregistration_afni(input_file1=None, input_file2=None, reference_file=None,
                         output_file1=None, output_file2=None,
                         estimate_affine=True, apply_affine=True,
                         affine_mat="mean_func_struct_aligned.aff12.1D"):
    """
    "New" coregistration approach (mol_fMRI_v5.3L_newcoreg.ipynb): rigid-body
    only (shift_rotate) with a normalized-mutual-information cost, which AFNI
    recommends for cross-modal (EPI <-> anatomical) registration - as opposed
    to the older coregistration_afni_old, which used a generic affine warp
    with cost="crU".
    """
    results = {}

    if reference_file is None:
        raise ValueError("reference_file must be provided")

    if estimate_affine:
        if output_file1 is None or input_file1 is None:
            raise ValueError("input_file1 and output_file1 must be provided when estimate_affine=True")

        coreg_wo_affine = afni.Allineate()
        coreg_wo_affine.inputs.in_file = input_file1
        coreg_wo_affine.inputs.reference = reference_file
        coreg_wo_affine.inputs.warp_type = "shift_rotate"
        coreg_wo_affine.inputs.cost = "nmi"
        coreg_wo_affine.inputs.two_pass = True
        coreg_wo_affine.inputs.verbose = True
        coreg_wo_affine.inputs.out_matrix = affine_mat
        coreg_wo_affine.inputs.out_param_file = "params.1D"
        coreg_wo_affine.inputs.out_file = output_file1
        coreg_wo_affine.run()

        print_statement(f"[OK] Rigid transform estimated -> {affine_mat}", bcolors.OKGREEN)
        print_statement(f"[OK] Coregistered image (step 1) -> {output_file1}", bcolors.OKGREEN)
        results["step1"] = output_file1

    if apply_affine:
        if output_file2 is None or input_file2 is None:
            raise ValueError("input_file2 and output_file2 must be provided when apply_affine=True")

        coreg_with_affine = afni.Allineate()
        coreg_with_affine.inputs.in_file = input_file2
        coreg_with_affine.inputs.reference = reference_file
        coreg_with_affine.inputs.in_matrix = affine_mat
        coreg_with_affine.inputs.master = reference_file
        coreg_with_affine.inputs.final_interpolation = "linear"
        coreg_with_affine.inputs.verbose = True
        coreg_with_affine.inputs.out_file = output_file2
        coreg_with_affine.run()

        print_statement(f"[OK] Transform applied -> {output_file2}", bcolors.OKGREEN)
        results["step2"] = output_file2

    return results


def ensure_functional_masks(stitched_dir, mask_source_dir):
    """
    Ensure mask_mean_mc_func.nii.gz / mask_mean_mc_func_cannulas.nii.gz exist
    in stitched_dir. Reuses masks from mask_source_dir (an existing single-scan
    functional directory with identical geometry) if provided and present,
    otherwise falls back to manual fsleyes drawing - exactly as in
    mol_fMRI_v5.3L_newcoreg.ipynb.
    """
    mask_file = stitched_dir / "mask_mean_mc_func.nii.gz"
    mask_file_cannulas = stitched_dir / "mask_mean_mc_func_cannulas.nii.gz"
    middle_vol = stitched_dir / "middle_vol.nii.gz"

    if not mask_file.exists() and mask_source_dir is not None:
        src = Path(mask_source_dir) / "mask_mean_mc_func.nii.gz"
        if src.exists():
            shutil.copyfile(src, mask_file)
            print_statement(f"[OK] Reused functional mask from {src}", bcolors.OKGREEN)

    if mask_file.exists():
        print_statement(f"Mask image ({mask_file}) exists.", bcolors.OKGREEN)
    else:
        print_statement(
            f"Mask image does not exist. Please create it and save as {mask_file}", bcolors.FAIL
        )
        subprocess.run(["fsleyes", str(middle_vol)])
        if not mask_file.exists():
            raise FileNotFoundError(f"{mask_file} was not created. Re-run once it exists.")

    if not mask_file_cannulas.exists() and mask_source_dir is not None:
        src = Path(mask_source_dir) / "mask_mean_mc_func_cannulas.nii.gz"
        if src.exists():
            shutil.copyfile(src, mask_file_cannulas)
            print_statement(f"[OK] Reused cannula mask from {src}", bcolors.OKGREEN)

    if mask_file_cannulas.exists():
        print_statement(f"Mask image including cannulas ({mask_file_cannulas}) exists.", bcolors.OKGREEN)
    else:
        print_statement(
            f"Mask including cannulas does not exist. Copying base mask as a starting point "
            f"and opening fsleyes to edit -> {mask_file_cannulas}", bcolors.NOTIFICATION
        )
        shutil.copyfile(mask_file, mask_file_cannulas)
        subprocess.run(["fsleyes", str(middle_vol), str(mask_file_cannulas)])

    return mask_file, mask_file_cannulas


def main(args):
    stitched_dir = Path(args.stitched_dir).resolve()
    struct_dir = Path(args.struct_dir).resolve()
    mc_file = stitched_dir / "mc_stitched_func.nii.gz"

    if not mc_file.exists():
        raise FileNotFoundError(f"{mc_file} not found - run electric_stim_stitching.py first.")

    print_header("Preprocessing stitched functional series", bcolors.HEADER)

    # ---- 1) Functional masking (reuse or draw) ----
    mask_file, mask_file_cannulas = ensure_functional_masks(stitched_dir, args.mask_source_dir)
    mask_file_shrunk = stitched_dir / "mask_mean_mc_func_cleaned.nii.gz"
    mask_file_cannulas_shrunk = stitched_dir / "mask_mean_mc_func_cannulas_cleaned.nii.gz"
    shrink_mask_xz_linewise(str(mask_file), str(mask_file_shrunk), trim_x=args.trim_x, trim_z=args.trim_z)
    shrink_mask_xz_linewise(str(mask_file_cannulas), str(mask_file_cannulas_shrunk), trim_x=args.trim_x, trim_z=args.trim_z)

    # ---- 2) Move into a fresh, timestamped analysis directory ----
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    user = getpass.getuser()
    analysis_dir = stitched_dir / f"{timestamp}_{user}"
    analysis_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(analysis_dir)
    print_statement(f"Analysis directory: {analysis_dir}", bcolors.NOTIFICATION)

    shutil.copyfile(mc_file, "mc_stitched_func.nii.gz")

    # ---- 3) Spatial smoothing (produced as an available output) ----
    print_header("Spatial smoothing", bcolors.HEADER)
    spatial_smoothing("mc_stitched_func.nii.gz", "sm_mc_stitched_func.nii.gz", args.fwhm)

    # ---- 4) Mean functional image + masking (for coregistration) ----
    mean_image("mc_stitched_func.nii.gz", "mean_mc_func.nii.gz")
    masking_file("mean_mc_func.nii.gz", str(mask_file_shrunk), "cleaned_shrunk_mean_mc_func.nii.gz")
    masking_file("mean_mc_func.nii.gz", str(mask_file), "cleaned_mean_mc_func.nii.gz")

    create_intensity_mask("cleaned_shrunk_mean_mc_func.nii.gz", "mask_thresholded.nii.gz", args.intensity_low, args.intensity_high)

    # ---- 5) Structural coregistration (new coreg: rigid + nmi), estimate only ----
    print_header("Coregistration to structural image", bcolors.HEADER)
    cleaned_struct = struct_dir / "cleaned_struct.nii.gz"
    structural_file_for_coregistration = cleaned_struct if cleaned_struct.exists() else struct_dir / "struct.nii.gz"
    print_statement(f"Using structural reference: {structural_file_for_coregistration}", bcolors.NOTIFICATION)

    coregistration_afni(
        input_file1="cleaned_mean_mc_func.nii.gz",
        reference_file=str(structural_file_for_coregistration),
        output_file1="mean_func_struct_aligned.nii.gz",
        estimate_affine=True,
        apply_affine=False,
        affine_mat="mean_func_struct_aligned.aff12.1D",
    )

    create_intensity_mask(str(structural_file_for_coregistration), "struct_mask_thresholded.nii.gz", args.intensity_low, args.intensity_high)
    shrink_mask_xz_linewise("struct_mask_thresholded.nii.gz", "struct_mask_thresholded_shrunk.nii.gz", trim_x=args.trim_x, trim_z=args.trim_z)
    masking_file(
        "mean_func_struct_aligned.nii.gz",
        "struct_mask_thresholded_shrunk.nii.gz",
        "cleaned_mean_func_struct_aligned.nii.gz",
    )

    print_header("Preprocessing complete", bcolors.HEADER)
    print_statement(f"Final coregistered functional mean image: "
                     f"{analysis_dir / 'cleaned_mean_func_struct_aligned.nii.gz'}", bcolors.OKGREEN)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Continue standard preprocessing (mol_fMRI_v5.3L_newcoreg.ipynb conventions) "
                    "on an already-stitched, already motion-corrected functional series: masking, "
                    "spatial smoothing, and rigid coregistration to structural (no SCM/tSNR/ROI analysis)."
    )
    ap.add_argument("--stitched_dir", type=str, required=True,
                     help="Directory containing mc_stitched_func.nii.gz and middle_vol.nii.gz "
                          "(output of electric_stim_stitching.py).")
    ap.add_argument("--struct_dir", type=str, required=True,
                     help="Directory containing the converted structural image "
                          "(cleaned_struct.nii.gz, falls back to struct.nii.gz).")
    ap.add_argument("--mask_source_dir", type=str, default=None,
                     help="Optional existing single-scan functional directory (e.g. 33functionalEPI) "
                          "to reuse mask_mean_mc_func.nii.gz / mask_mean_mc_func_cannulas.nii.gz from. "
                          "Only valid if scans share identical geometry (back-to-back, unchanged params).")
    ap.add_argument("--fwhm", type=float, default=0.297, help="Spatial smoothing kernel FWHM (mm).")
    ap.add_argument("--intensity_low", type=float, default=15, help="Lower percentile for intensity masks.")
    ap.add_argument("--intensity_high", type=float, default=80, help="Upper percentile for intensity masks.")
    ap.add_argument("--trim_x", type=int, default=2, help="Voxels to trim along X when shrinking masks.")
    ap.add_argument("--trim_z", type=int, default=2, help="Voxels to trim along Z when shrinking masks.")
    args = ap.parse_args()

    main(args)
