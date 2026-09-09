#!/usr/bin/env python3
"""Coregister subject images to an atlas and extract ROI timecourses."""
import argparse
import glob
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv
from nibabel.processing import resample_from_to
from nipype.interfaces import afni

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


# ============================================================================
# CONFIGURATION: Set your paths here (optional)
# Leave as None to be prompted via dialogs, or set to your paths
# ============================================================================
CONFIG = {
    # Subject data
    "subject_dir": None,  # Path to single subject or None
    "group_root": "/home/pschmidt/fmri_analysis/QC/QC_in/data",  # Path to group root (multiple subjects) or None
    "subject_glob": "sub*",  # Glob pattern for finding subjects
    
    # Anatomical and functional images
    "struct_pattern": "*struct*.nii*",  # Pattern to find structural image
    "func_pattern": "*func*.nii*",      # Pattern to find functional image
    
    # Atlas and ROIs
    "atlas_reference": "/home/pschmidt/fmri_analysis/QC/QC_in/atlas/SIGMA_InVivo_Brain_Template_Masked.nii",  # Path to atlas reference image or None
    "roi_masks_dir": "/home/pschmidt/fmri_analysis/QC/QC_in/rois",    # Path to ROI masks directory or None
    "roi_mask_glob": "*.nii.gz*",  # Glob pattern for finding ROI masks
    
    # Output directories
    "output_dir": "/home/pschmidt/fmri_analysis/QC/QC_out",  # Directory for aligned images and results or None
    "plot_dir": "/home/pschmidt/fmri_analysis/QC/QC_out",    # Directory for plots or None
}
# ============================================================================


def choose_folder_via_dialog(title: str) -> Path:
    if tk is None or filedialog is None:
        raise RuntimeError("Tkinter is not available for folder selection dialogs.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    if not folder:
        raise FileNotFoundError(f"No folder selected: {title}")
    return Path(folder)
def choose_file_via_dialog(title: str, filetypes=None) -> Path:
    if tk is None or filedialog is None:
        raise RuntimeError("Tkinter is not available for file selection dialogs.")
    if filetypes is None:
        filetypes = [("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not file_path:
        raise FileNotFoundError(f"No file selected: {title}")
    return Path(file_path)
def find_subject_dirs(group_root: Path, subject_glob: str = "*") -> list[Path]:
    direct_subject_paths = sorted(
        [Path(p) for p in glob.glob(str(group_root / subject_glob)) if Path(p).is_dir()]
    )

    if direct_subject_paths:
        return direct_subject_paths

    nested_subject_paths = sorted(
        [
            p
            for p in group_root.iterdir()
            if p.is_dir() and (any(p.glob("*struct*.nii*")) or any(p.glob("*func*.nii*")))
        ]
    )

    if nested_subject_paths:
        return nested_subject_paths

    raise FileNotFoundError(
        f"No subject directories found in {group_root} using pattern {subject_glob}"
    )
def find_single_file(subject_dir: Path, pattern: str, description: str) -> Path:
    matches = sorted(subject_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {description} found in {subject_dir} using pattern {pattern}"
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple {description} found in {subject_dir} using pattern {pattern}: {matches}"
        )
    return matches[0]
def trim_functional_to_first_n_volumes(func_path: Path, output_path: Path | None = None, max_volumes: int = 600) -> Path:
    """Return a functional image cropped to the first max_volumes volumes."""
    func_img = nib.load(str(func_path))
    func_data = func_img.get_fdata(dtype=np.float32)
    if func_data.ndim != 4:
        raise ValueError(f"Functional image must be 4D: {func_path}")

    n_volumes = func_data.shape[-1]
    if n_volumes <= max_volumes:
        return func_path

    trimmed_data = func_data[..., :max_volumes]
    trimmed_img = nib.Nifti1Image(trimmed_data, func_img.affine, func_img.header)

    if output_path is None:
        output_path = func_path.with_name(f"{func_path.stem}_first{max_volumes}{func_path.suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        nib.save(trimmed_img, str(output_path))
        print(f"[INFO] Trimmed {func_path.name} to first {max_volumes} volumes -> {output_path.name}")

    return output_path


def register_struct_to_functional(
    subject_name: str,
    subject_struct: Path,
    subject_func: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Register structural image to functional space and return the transform matrix and aligned structural image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    struct2func_mat = output_dir / f"{subject_name}_struct2func.aff12.1D"
    struct_aligned_to_func = output_dir / f"{subject_name}_struct_func.nii.gz"

    if struct2func_mat.exists() and struct_aligned_to_func.exists():
        print(f"[SKIP] Structural-to-functional registration already exists for {subject_name}")
        return struct2func_mat, struct_aligned_to_func

    print(f"[INFO] Registering {subject_name} structural image to functional image")
    coreg = afni.Allineate()
    coreg.inputs.in_file = str(subject_struct)
    coreg.inputs.reference = str(subject_func)
    coreg.inputs.out_matrix = str(struct2func_mat)
    coreg.inputs.out_file = str(struct_aligned_to_func)
    coreg.inputs.out_param_file = str(output_dir / f"{subject_name}_struct2func_params.1D")
    coreg.inputs.warp_type = "shift_rotate"
    coreg.inputs.cost = "nmi"
    coreg.inputs.two_pass = True
    coreg.inputs.verbose = True
    coreg.inputs.overwrite = True
    coreg.run()
    print(f"[OK] Structural aligned to functional: {struct_aligned_to_func}")

    return struct2func_mat, struct_aligned_to_func


def register_subject_to_atlas(
    subject_name: str,
    subject_struct: Path,
    subject_func: Path,
    atlas_reference: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Register the original structural image directly to the atlas, and separately
    register structural to functional.  Both transforms are returned so they can be
    chained (atlas -> struct -> func) when resampling ROI masks.

    Using the original (high-res) structural for atlas registration avoids the
    quality loss that occurs when registering a previously downsampled image.
    lpa (local Pearson absolute) cost is used because it captures local intensity
    structure and outperforms nmi for same-modality (T2->T2) rodent brain registration.

    Returns: (struct2atlas_mat, struct2func_mat, functional_image_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: struct -> functional space (rigid body, needed for ROI chain later)
    struct2func_mat, _ = register_struct_to_functional(
        subject_name,
        subject_struct,
        subject_func,
        output_dir,
    )

    # Direct registration: original high-res struct → atlas (single step, no CoM pre-alignment)
    affine_mat = output_dir / f"{subject_name}_struct2atlas.aff12.1D"
    struct_aligned = output_dir / f"{subject_name}_struct_atlas.nii.gz"

    if affine_mat.exists() and struct_aligned.exists():
        print(f"[SKIP] Structural-to-atlas registration already exists for {subject_name}")
        return affine_mat, struct2func_mat, subject_func

    print(f"[INFO] Registering {subject_name} structural image to atlas")
    # Use subprocess directly: nipype's Allineate trait allowlist excludes lpa/lpc.
    try:
        subprocess.run(
            ["3dAllineate",
             "-source",           str(subject_struct),
             "-base",             str(atlas_reference),
             "-1Dmatrix_save",    str(affine_mat),
             "-prefix",           str(struct_aligned),
             "-warp",             "shift_rotate",
             "-cost",             "nmi",
             "-twopass",
             "-verb",
             "-overwrite"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"3dAllineate struct->atlas failed: {e}")
    print(f"[OK] Structural aligned to atlas: {struct_aligned}")

    return affine_mat, struct2func_mat, subject_func


def resample_mask_to_functional(
    roi_mask_path: Path,
    func_path: Path,
    atlas_reference: Path,
    struct2atlas_mat: Path,
    struct2func_mat: Path,
    output_dir: Path,
    subject_name: str,
) -> Path:
    """Resample ROI mask from atlas space to functional space.

    The path atlas -> func is the chain of two transforms:
      1. inv(struct2atlas): atlas coords -> struct coords
      2. struct2func:       struct coords -> func coords

    cat_matvec concatenates them in left-to-right order and honours AFNI's
    internal DICOM/RAI conventions.  The resulting matrix is applied with
    -1Dmatrix_apply (correct flag for .aff12.1D matrix files).
    """
    roi_name = roi_mask_path.stem.replace(".nii", "")
    resampled_mask = output_dir / f"{subject_name}_{roi_name}_funcspace.nii.gz"

    if resampled_mask.exists():
        return resampled_mask

    print(f"[INFO] Resampling {roi_name} to functional space for {subject_name}")

    # Chain: inv(struct2atlas) then struct2func  =>  atlas -> struct -> func
    atlas2func_mat = output_dir / f"{subject_name}_{roi_name}_atlas2func.aff12.1D"
    chain_cmd = [
        "cat_matvec", "-ONELINE",
        str(struct2atlas_mat), "-I",  # invert struct2atlas: atlas -> struct
        str(struct2func_mat),          # struct -> func
    ]
    try:
        result = subprocess.run(chain_cmd, check=True, capture_output=True, text=True)
        atlas2func_mat.write_text(result.stdout)
        print(f"[DEBUG] Chained atlas->func matrix saved to: {atlas2func_mat}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"cat_matvec chaining failed: {e}\nstderr: {e.stderr}"
        )

    # Apply the chained transform with -1Dmatrix_apply (for .aff12.1D matrix files)
    cmd = [
        "3dAllineate",
        "-source", str(roi_mask_path),
        "-master", str(func_path),
        "-1Dmatrix_apply", str(atlas2func_mat),
        "-NN",  # nearest-neighbour interpolation for binary masks
        "-prefix", str(resampled_mask),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[OK] Mask resampled to functional space: {resampled_mask}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"AFNI 3dAllineate failed: {e}\nstdout: {e.stdout}\nstderr: {e.stderr}"
        )

    return resampled_mask


def extract_roi_timecourse(func_path: Path, roi_mask_path: Path) -> np.ndarray:
    func_img = nib.load(str(func_path))
    func_data = func_img.get_fdata(dtype=np.float32)
    if func_data.ndim != 4:
        raise ValueError(f"Functional image must be 4D: {func_path}")

    mask_img = nib.load(str(roi_mask_path))
    mask_data = mask_img.get_fdata(dtype=np.float32)
    if mask_data.shape != func_data.shape[:3]:
        raise ValueError(
            f"ROI mask shape {mask_data.shape} does not match functional image shape {func_data.shape[:3]}"
        )

    mask = mask_data > 0
    if not np.any(mask):
        raise ValueError(f"ROI mask contains no voxels: {roi_mask_path}")

    roi_voxels = func_data[mask]
    return np.mean(roi_voxels, axis=0)


def save_roi_group_timeseries(
    subject_ts: dict[str, np.ndarray],
    roi_name: str,
    output_csv: Path,
    plot_path: Path | None = None,
) -> None:
    df = pd.DataFrame(subject_ts)
    df["group_mean"] = df.mean(axis=1)
    df.index.name = "timepoint"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=True)
    print(f"[OK] Saved ROI timecourse CSV: {output_csv}")

    if plot_path is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for subject_name in subject_ts:
            ax.plot(df.index, df[subject_name], alpha=0.5, label=subject_name)
        ax.plot(df.index, df["group_mean"], color="black", linewidth=2, label="group_mean")
        ax.set_title(f"ROI timecourse: {roi_name}")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Mean signal")
        ax.legend(fontsize="small", ncol=2, loc="best")
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved ROI plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coregister subject data to atlas and extract ROI timecourses."
    )
    parser.add_argument("--group-root", type=Path, default=CONFIG.get("group_root"))
    parser.add_argument("--subject-dir", type=Path, default=CONFIG.get("subject_dir"))
    parser.add_argument("--subject-glob", type=str, default=CONFIG.get("subject_glob", "*"))
    parser.add_argument("--atlas-reference", type=Path, default=CONFIG.get("atlas_reference"))
    parser.add_argument("--struct-pattern", type=str, default=CONFIG.get("struct_pattern", "*struct*.nii*"))
    parser.add_argument("--func-pattern", type=str, default=CONFIG.get("func_pattern", "*func*.nii*"))
    parser.add_argument("--roi-masks-dir", type=Path, default=CONFIG.get("roi_masks_dir"))
    parser.add_argument("--roi-mask-glob", type=str, default=CONFIG.get("roi_mask_glob", "*.nii*"))
    parser.add_argument("--output-dir", type=Path, default=CONFIG.get("output_dir"))
    parser.add_argument("--plot-dir", type=Path, default=CONFIG.get("plot_dir"))
    parser.add_argument(
        "--use-dialogs",
        action="store_true",
        help="Open file/folder selection dialogs for any missing paths.",
    )
    return parser.parse_args()

def get_incidence(r_data, r2_data):
    specific = 0
    non_specific = 0
    no_fc = 0
    for i in range(len(r_data)):
        if r_data[i] > 0.1 and r2_data[i] < 0.1:
            specific += 1
        if r_data[i] > 0.1 and r2_data[i] > 0.1:
            non_specific += 1
        if -0.1 < r_data[i] < 0.1 and -0.1 < r2_data[i] < 0.1:
            no_fc += 1
    spurious_fc = len(r_data) - (specific + non_specific + no_fc)
    return np.array([specific/len(r_data)*100, non_specific/len(r_data)*100, no_fc/len(r_data)*100, spurious_fc/len(r_data)*100])

def main() -> None:
    args = parse_args()
    
    # Print which paths are loaded from CONFIG
    print("=" * 70)
    print("PATH CONFIGURATION")
    print("=" * 70)
    config_paths = {
        "subject_dir": args.subject_dir,
        "group_root": args.group_root,
        "atlas_reference": args.atlas_reference,
        "roi_masks_dir": args.roi_masks_dir,
        "output_dir": args.output_dir,
        "plot_dir": args.plot_dir,
    }
    
    for key, value in config_paths.items():
        status = f"✓ {value}" if value else "✗ (will prompt)"
        print(f"  {key:20s}: {status}")
    print("=" * 70)

    use_dialogs = args.use_dialogs or (
        args.subject_dir is None and args.group_root is None
    ) or args.atlas_reference is None or args.roi_masks_dir is None

    if use_dialogs:
        print("\n[INFO] Prompting for missing paths via dialog...")
        if args.subject_dir is None and args.group_root is None:
            args.subject_dir = choose_folder_via_dialog("Select subject folder")
        if args.atlas_reference is None:
            args.atlas_reference = choose_file_via_dialog(
                "Select atlas reference file",
                [("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")],
            )
        if args.roi_masks_dir is None:
            args.roi_masks_dir = choose_folder_via_dialog("Select ROI masks folder")
        if args.output_dir is None:
            args.output_dir = choose_folder_via_dialog("Select output folder")
        if args.plot_dir is None:
            args.plot_dir = choose_folder_via_dialog("Select plot folder")

    results_dir = args.output_dir or (args.subject_dir or args.group_root) / "FC_QC_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.subject_dir is not None:
        if any(args.subject_dir.glob(args.struct_pattern)) or any(args.subject_dir.glob(args.func_pattern)):
            subject_dirs = [args.subject_dir]
        else:
            subject_dirs = find_subject_dirs(args.subject_dir, args.subject_glob)
    else:
        subject_dirs = find_subject_dirs(args.group_root, args.subject_glob)

    print(f"[OK] Found {len(subject_dirs)} subject directories")

    roi_masks = sorted(args.roi_masks_dir.glob(args.roi_mask_glob))
    if not roi_masks:
        raise FileNotFoundError(f"No ROI masks found in {args.roi_masks_dir} using {args.roi_mask_glob}")

    print(f"[OK] Found {len(roi_masks)} ROI masks")

    # Compute registration and get transform matrices
    subject_transforms: dict[str, tuple[Path, Path, Path]] = {}  # {subject_name: (struct2atlas_mat, struct2func_mat, func_path)}
    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        subject_output_dir = results_dir / subject_name
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        struct_path = find_single_file(subject_dir, args.struct_pattern, "structural image")
        func_path = find_single_file(subject_dir, args.func_pattern, "functional image")
        trimmed_func_path = trim_functional_to_first_n_volumes(
            func_path,
            subject_output_dir / f"{subject_name}_func_first600.nii.gz",
        )
        struct2atlas_mat, struct2func_mat, func_path_ret = register_subject_to_atlas(
            subject_name,
            struct_path,
            trimmed_func_path,
            args.atlas_reference,
            subject_output_dir,
        )
        subject_transforms[subject_name] = (struct2atlas_mat, struct2func_mat, func_path_ret)

    # Extract ROI timecourses
    for roi_mask_path in roi_masks:
        roi_name = roi_mask_path.stem.replace(".nii", "")

        for subject_name, (struct2atlas_mat, struct2func_mat, func_path) in subject_transforms.items():
            subject_output_dir = results_dir / subject_name
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            output_csv = subject_output_dir / f"roi_timeseries_{roi_name}.csv"
            plot_path = None
            if args.plot_dir is not None:
                plot_path = Path(args.plot_dir) / subject_name / f"roi_timeseries_{roi_name}.png"

            if output_csv.exists():
                print(f"[SKIP] ROI timeseries already extracted for {subject_name}/{roi_name}")
                continue

            resampled_mask = resample_mask_to_functional(
                roi_mask_path,
                func_path,
                args.atlas_reference,
                struct2atlas_mat,
                struct2func_mat,
                subject_output_dir,
                subject_name,
            )
            timecourse = extract_roi_timecourse(func_path, resampled_mask)
            save_roi_group_timeseries(
                {subject_name: timecourse},
                roi_name,
                output_csv,
                plot_path,
            )

    print(f"[OK] Completed ROI extraction for {len(roi_masks)} masks")
    print(f"[INFO] Processing complete. Working in functional space avoids memory issues from large atlas-space resampling.")


if __name__ == "__main__":
    main()
