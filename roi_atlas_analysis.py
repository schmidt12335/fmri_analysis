#!/usr/bin/env python3
"""
Extract per-ROI time courses from an already-preprocessed functional scan
(e.g. mc_stitched_func.nii.gz from electric_stim_stitching.py /
preprocess_stitched_scan.py) using a multi-label atlas (e.g. Allen Mouse
Brain Atlas), following the same approach as FC_QC_updated.py:

  1. Rigidly register the structural image directly to functional space
     (struct2func) and directly to the atlas template (struct2atlas).
  2. Compose those two rigid transforms (via cat_matvec) into a single
     functional -> atlas coordinate matrix - this avoids ever resampling/
     interpolating the functional data itself.
  3. Resample the ROI *label* atlas (not the functional data) onto the
     functional grid ONCE, using nearest-neighbour interpolation (so label
     integers are preserved exactly).
  4. For each ROI label present within the functional field of view, build a
     binary mask on the functional grid and extract its mean time course
     directly from the full 4D functional data with 3dmaskave (no resampling
     of the functional data, minimal RAM usage).

This differs from FC_QC_updated.py only in the ROI input: instead of a
directory of individual binary ROI mask files, it takes a single multi-label
atlas volume (one integer per region), which is what an Allen Mouse Brain
Atlas parcellation looks like.

Usage
-----
python roi_atlas_analysis.py \
    --stitched_dir  /path/to/stitched_EPI/<timestamp>_<user> \
    --struct_dir    /path/to/14anatomy \
    --atlas_template /path/to/allen_template.nii.gz \
    --atlas_labels   /path/to/allen_labels.nii.gz \
    --label_names_csv /path/to/allen_label_names.csv

Expected inputs
----------------
--stitched_dir must contain mc_stitched_func.nii.gz (the full 4D functional
    series to extract time courses from). If it also contains
    mean_mc_func.nii.gz (produced by preprocess_stitched_scan.py) that is
    reused as the functional registration target; otherwise a temporal mean
    of the first --func_reference_volumes volumes is computed.
--struct_dir must contain cleaned_struct.nii.gz (falls back to struct.nii.gz).
--atlas_template / --atlas_labels must be in the same space/grid as each
    other (a standard atlas template + matching label volume).
--label_names_csv (optional) - a CSV with columns "id,name" to give ROI
    outputs readable names instead of just their numeric label ID.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


REQUIRED_AFNI_PROGRAMS = ("3dAllineate", "3dTstat", "3dmaskave", "cat_matvec")


def run_command(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess:
    print("[COMMAND]", " ".join(command))
    try:
        return subprocess.run(command, check=True, text=True, capture_output=capture_output)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Command failed:\n  {' '.join(command)}\n\n"
            f"stdout:\n{error.stdout or ''}\n\nstderr:\n{error.stderr or ''}"
        ) from error


def check_required_programs() -> None:
    import shutil
    missing = [p for p in REQUIRED_AFNI_PROGRAMS if shutil.which(p) is None]
    if missing:
        raise RuntimeError("Required AFNI programs not found on PATH: " + ", ".join(missing))


def add_cmass_option(command: list[str], cmass_setting: str | None) -> None:
    if not cmass_setting:
        return
    cmass_setting = cmass_setting.strip()
    if cmass_setting == "cmass":
        command.append("-cmass")
    elif cmass_setting.startswith("+"):
        command.append(f"-cmass{cmass_setting}")
    else:
        raise ValueError("cmass setting must be None, '', 'cmass', or one of '+a','+xy','+xz','+yz'.")


def add_optional_registration_limits(command: list[str], max_rotation_deg, max_shift_mm) -> None:
    if max_rotation_deg is not None:
        command.extend(["-maxrot", str(float(max_rotation_deg))])
    if max_shift_mm is not None:
        command.extend(["-maxshf", str(float(max_shift_mm))])


def get_functional_reference(stitched_dir: Path, func_file: Path, n_volumes: int, force: bool) -> Path:
    existing = stitched_dir / "mean_mc_func.nii.gz"
    if existing.exists() and not force:
        print(f"[OK] Reusing existing functional reference: {existing}")
        return existing

    func_img = nib.load(str(func_file))
    if len(func_img.shape) != 4:
        raise ValueError(f"Functional image must be 4D: {func_file}")
    n_use = min(int(func_img.shape[3]), int(n_volumes))

    output_path = stitched_dir / f"func_mean_first{n_use}.nii.gz"
    if output_path.exists() and not force:
        print(f"[SKIP] Functional reference already exists: {output_path}")
        return output_path

    run_command(["3dTstat", "-mean", "-prefix", str(output_path), "-overwrite",
                 f"{func_file}[0..{n_use - 1}]"])
    print(f"[OK] Functional reference created: {output_path}")
    return output_path


def register_struct_to_functional(struct_file: Path, func_reference: Path, output_dir: Path,
                                   cost: str, cmass: str | None, twobest: str,
                                   max_rotation_deg, max_shift_mm, force: bool) -> tuple[Path, Path]:
    matrix_path = output_dir / "struct2func.aff12.1D"
    params_path = output_dir / "struct2func_params.1D"
    aligned_path = output_dir / "struct_in_func.nii.gz"

    if matrix_path.exists() and aligned_path.exists() and not force:
        print("[SKIP] struct->func registration already exists")
        return matrix_path, aligned_path

    command = [
        "3dAllineate", "-source", str(struct_file), "-base", str(func_reference),
        "-prefix", str(aligned_path), "-1Dmatrix_save", str(matrix_path),
        "-1Dparam_save", str(params_path), "-warp", "shift_rotate",
        "-cost", cost, "-source_automask+2", "-twopass", "-twobest", twobest,
        "-float", "-verb", "-overwrite",
    ]
    add_cmass_option(command, cmass)
    add_optional_registration_limits(command, max_rotation_deg, max_shift_mm)

    print("[INFO] Registering structural -> functional")
    run_command(command)
    print(f"[OK] Structural aligned to functional: {aligned_path}")
    return matrix_path, aligned_path


def register_struct_to_atlas(struct_file: Path, atlas_template: Path, output_dir: Path,
                              cost: str, cmass: str | None, twobest: str,
                              max_rotation_deg, max_shift_mm, force: bool) -> tuple[Path, Path]:
    matrix_path = output_dir / "struct2atlas.aff12.1D"
    params_path = output_dir / "struct2atlas_params.1D"
    aligned_path = output_dir / "struct_in_atlas.nii.gz"

    if matrix_path.exists() and aligned_path.exists() and not force:
        print("[SKIP] struct->atlas registration already exists")
        return matrix_path, aligned_path

    command = [
        "3dAllineate", "-source", str(struct_file), "-base", str(atlas_template),
        "-prefix", str(aligned_path), "-1Dmatrix_save", str(matrix_path),
        "-1Dparam_save", str(params_path), "-warp", "shift_rotate",
        "-cost", cost, "-source_automask+2", "-twopass", "-twobest", twobest,
        "-float", "-verb", "-overwrite",
    ]
    add_cmass_option(command, cmass)
    add_optional_registration_limits(command, max_rotation_deg, max_shift_mm)

    print("[INFO] Registering structural -> atlas")
    run_command(command)
    print(f"[OK] Structural aligned to atlas: {aligned_path}")
    return matrix_path, aligned_path


def convert_aff12_to_plain_matvec(matrix_path: Path, output_path: Path) -> Path:
    """Convert a single-row .aff12.1D matrix to cat_matvec's plain 3x4 format."""
    values: list[float] = []
    for line in matrix_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.extend(float(v) for v in stripped.split())

    if len(values) != 12:
        raise ValueError(f"Expected 12 values in {matrix_path}, found {len(values)}.")

    rows = [values[0:4], values[4:8], values[8:12]]
    output_path.write_text("\n".join(" ".join(f"{v:.12g}" for v in row) for row in rows) + "\n")
    return output_path


def compose_func_to_atlas_coordinate_matrix(struct2atlas_matrix: Path, struct2func_matrix: Path,
                                             output_dir: Path, force: bool) -> Path:
    """
    3dAllineate matrices are base -> source coordinate transforms:
        struct2func:  functional coordinates -> structural coordinates
        struct2atlas: atlas coordinates      -> structural coordinates
    Applying an atlas-space volume onto a functional master needs:
        functional coordinates -> atlas coordinates
            = inverse(struct2atlas) * struct2func
    cat_matvec applies the second transform after the first, so command-line
    order is: struct2func  struct2atlas -I
    """
    output_path = output_dir / "func_to_atlas_for_roi_apply.aff12.1D"
    if output_path.exists() and not force:
        return output_path

    plain_struct2func = convert_aff12_to_plain_matvec(struct2func_matrix, output_dir / "struct2func.matvec")
    plain_struct2atlas = convert_aff12_to_plain_matvec(struct2atlas_matrix, output_dir / "struct2atlas.matvec")

    result = run_command(
        ["cat_matvec", "-ONELINE", str(plain_struct2func), str(plain_struct2atlas), "-I"],
        capture_output=True,
    )
    output_path.write_text(result.stdout.strip() + "\n")
    print(f"[OK] Composite functional -> atlas coordinate matrix: {output_path}")
    return output_path


def resample_labels_to_functional(atlas_labels: Path, func_reference: Path,
                                   composite_matrix: Path, output_dir: Path, force: bool) -> Path:
    """Resample the WHOLE multi-label atlas onto the functional grid once, using NN
    interpolation so label integers are preserved exactly."""
    output_path = output_dir / "atlas_labels_funcspace.nii.gz"
    if output_path.exists() and not force:
        print(f"[SKIP] Resampled label atlas already exists: {output_path}")
        return output_path

    run_command([
        "3dAllineate", "-source", str(atlas_labels), "-master", str(func_reference),
        "-1Dmatrix_apply", str(composite_matrix), "-final", "NN",
        "-prefix", str(output_path), "-overwrite",
    ])
    print(f"[OK] Atlas labels resampled to functional space: {output_path}")
    return output_path


def load_label_names(csv_path: Path | None) -> dict[int, str]:
    if csv_path is None:
        return {}
    df = pd.read_csv(csv_path)
    if not {"id", "name"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must have columns 'id' and 'name'")
    return {int(row.id): str(row.name) for row in df.itertuples()}


def get_present_labels(label_img_path: Path) -> list[int]:
    data = np.asanyarray(nib.load(str(label_img_path)).dataobj)
    labels = sorted(int(v) for v in np.unique(data) if v != 0)
    return labels


def make_roi_mask_from_label(label_img: nib.Nifti1Image, label_id: int, out_path: Path) -> int:
    data = np.asanyarray(label_img.dataobj)
    mask = (data == label_id).astype(np.uint8)
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return 0
    nib.save(nib.Nifti1Image(mask, label_img.affine, label_img.header), str(out_path))
    return voxel_count


def extract_roi_timecourse(func_path: Path, roi_mask_path: Path) -> np.ndarray:
    result = run_command(["3dmaskave", "-quiet", "-mask", str(roi_mask_path), str(func_path)],
                          capture_output=True)
    timecourse = np.fromstring(result.stdout, sep=" ", dtype=float)
    if timecourse.size == 0:
        raise ValueError(f"3dmaskave returned an empty time course for {roi_mask_path}")
    if not np.all(np.isfinite(timecourse)):
        raise ValueError(f"Non-finite values in ROI time course for {roi_mask_path}")
    return timecourse


def main(args: argparse.Namespace) -> None:
    check_required_programs()

    stitched_dir = Path(args.stitched_dir).resolve()
    struct_dir = Path(args.struct_dir).resolve()
    atlas_template = Path(args.atlas_template).resolve()
    atlas_labels = Path(args.atlas_labels).resolve()

    func_file = stitched_dir / "mc_stitched_func.nii.gz"
    if not func_file.exists():
        raise FileNotFoundError(f"{func_file} not found.")

    cleaned_struct = struct_dir / "cleaned_struct.nii.gz"
    struct_file = cleaned_struct if cleaned_struct.exists() else struct_dir / "struct.nii.gz"
    if not struct_file.exists():
        raise FileNotFoundError(f"No structural image found in {struct_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else stitched_dir / "roi_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    func_reference = get_functional_reference(
        stitched_dir, func_file, args.func_reference_volumes, args.force
    )

    struct2func_matrix, _ = register_struct_to_functional(
        struct_file, func_reference, output_dir,
        cost=args.struct_to_func_cost, cmass=args.struct_to_func_cmass, twobest=args.twobest,
        max_rotation_deg=args.max_rotation_deg, max_shift_mm=args.max_shift_mm, force=args.force,
    )
    struct2atlas_matrix, _ = register_struct_to_atlas(
        struct_file, atlas_template, output_dir,
        cost=args.atlas_cost, cmass=args.atlas_cmass, twobest=args.twobest,
        max_rotation_deg=args.max_rotation_deg, max_shift_mm=args.max_shift_mm, force=args.force,
    )

    composite_matrix = compose_func_to_atlas_coordinate_matrix(
        struct2atlas_matrix, struct2func_matrix, output_dir, args.force
    )

    labels_funcspace = resample_labels_to_functional(
        atlas_labels, func_reference, composite_matrix, output_dir, args.force
    )

    label_names = load_label_names(Path(args.label_names_csv) if args.label_names_csv else None)
    present_labels = get_present_labels(labels_funcspace)
    print(f"[OK] {len(present_labels)} ROI labels present in the functional field of view")

    label_img = nib.load(str(labels_funcspace))
    masks_dir = output_dir / "roi_masks_funcspace"
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    timecourses: dict[str, np.ndarray] = {}

    for label_id in present_labels:
        roi_name = label_names.get(label_id, f"label_{label_id}")
        mask_path = masks_dir / f"roi_{roi_name}.nii.gz"

        voxel_count = make_roi_mask_from_label(label_img, label_id, mask_path)
        if voxel_count == 0:
            continue

        timecourse = extract_roi_timecourse(func_file, mask_path)
        timecourses[roi_name] = timecourse

        csv_path = output_dir / f"roi_timeseries_{roi_name}.csv"
        pd.DataFrame({"timepoint": np.arange(timecourse.size), "signal": timecourse}).to_csv(csv_path, index=False)

        manifest_rows.append({"label_id": label_id, "name": roi_name, "voxel_count": voxel_count,
                               "mask_file": str(mask_path), "timeseries_csv": str(csv_path)})
        print(f"[OK] {roi_name} (id={label_id}, {voxel_count} voxels) -> {csv_path}")

    manifest_path = output_dir / "roi_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"[OK] ROI manifest saved: {manifest_path}")

    if timecourses:
        max_len = max(ts.size for ts in timecourses.values())
        wide = pd.DataFrame({"timepoint": np.arange(max_len)})
        for name, ts in timecourses.items():
            padded = np.full(max_len, np.nan)
            padded[: ts.size] = ts
            wide[name] = padded
        wide_path = output_dir / "roi_timeseries_all.csv"
        wide.to_csv(wide_path, index=False)
        print(f"[OK] Combined ROI time series saved: {wide_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract per-ROI time courses from a preprocessed functional scan using a "
                    "multi-label atlas (e.g. Allen Mouse Brain Atlas), following FC_QC_updated.py's "
                    "struct->func / struct->atlas rigid registration + composite-matrix approach."
    )
    ap.add_argument("--stitched_dir", type=str, required=True,
                     help="Directory containing mc_stitched_func.nii.gz (and optionally mean_mc_func.nii.gz).")
    ap.add_argument("--struct_dir", type=str, required=True,
                     help="Directory containing cleaned_struct.nii.gz (falls back to struct.nii.gz).")
    ap.add_argument("--atlas_template", type=str, required=True,
                     help="Atlas structural reference image (same space as --atlas_labels).")
    ap.add_argument("--atlas_labels", type=str, required=True,
                     help="Multi-label ROI atlas volume (integers = ROI IDs), same grid as --atlas_template.")
    ap.add_argument("--label_names_csv", type=str, default=None,
                     help="Optional CSV with columns 'id,name' for readable ROI output names.")
    ap.add_argument("--output_dir", type=str, default=None,
                     help="Where to write registration + ROI outputs. Default: <stitched_dir>/roi_analysis.")
    ap.add_argument("--func_reference_volumes", type=int, default=600,
                     help="Number of initial volumes to average for the func registration target, "
                          "if mean_mc_func.nii.gz isn't already present in --stitched_dir.")
    ap.add_argument("--struct_to_func_cost", type=str, default="nmi")
    ap.add_argument("--struct_to_func_cmass", type=str, default=None,
                     help="Center-of-mass init for struct->func (usually unnecessary, same-session data).")
    ap.add_argument("--atlas_cost", type=str, default="lpa+ZZ")
    ap.add_argument("--atlas_cmass", type=str, default="+xz",
                     help="Center-of-mass init for struct->atlas. '+xz' suits a coronal slab with "
                          "limited A-P coverage (matches FC_QC_updated.py's convention).")
    ap.add_argument("--twobest", type=str, default="MAX")
    ap.add_argument("--max_rotation_deg", type=float, default=None)
    ap.add_argument("--max_shift_mm", type=float, default=None)
    ap.add_argument("--force", action="store_true",
                     help="Recompute registrations/resampled labels/time courses even if outputs exist.")
    args = ap.parse_args()

    main(args)
