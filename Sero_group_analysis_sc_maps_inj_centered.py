#!/usr/bin/env python3
"""
Group analysis script for average fMRI percent signal change (PSC) outputs,
aligned by injection site instead of atlas registration, with an optional
final atlas coregistration step.

WHAT IT DOES
------------
Two independent, optionally combined steps:

1. PSC map averaging (injection-centered, optionally atlas-registered)
   - Finds each subject's PSC/signal-change map (--map-pattern, default
     "*scm*.nii*") under --group-root (or --group-root/sc_maps if that
     subfolder exists).
   - For each subject, gets (or creates) an injection-site ROI mask
     (--roi-filename, default "inj_roi.nii.gz") in that subject's folder.
     If the mask doesn't exist yet, FSLeyes is launched with the subject's
     structural scan (--struct-pattern, default "*struct*.nii*") as an
     underlay and the PSC map overlaid on top -- draw/save the ROI there
     (see PROMPTS below).
   - Translates every subject's map so its ROI centroid overlaps the
     reference subject's (--reference-subject, defaults to the first
     subject alphabetically), then resamples everyone onto a shared
     isotropic bounding-box grid and averages them -> --map-output
     (default group_level_results/group_average_signal_change_map.nii.gz).
   - Optional atlas step: if an atlas/template is supplied
     (--atlas-reference), ONLY the reference subject's structural scan is
     registered to it (AFNI Allineate, once). That single transform is then
     applied to the group-average map above (since all subjects already
     share the reference subject's coordinate frame) -> --atlas-map-output
     (default group_level_results/group_average_signal_change_map_atlas.nii.gz).
     Pass --no-atlas to skip this step without being prompted.

2. ROI time series averaging (unrelated text-file averaging step)
   - Averages existing per-subject ROI time course .txt files
     (--roi-pattern, default "PSC_time_series_*.txt") found under each
     folder returned by --group-root/--subject-glob, and writes a CSV
     (--roi-output) plus an optional plot (--roi-plot).
   - NOTE: for this dataset's actual folder layout (timeseries files live
     under group_data/timeseries/<dose>/*.txt, not per-subject folders),
     this step currently finds no matches and will raise "No ROI time
     series files found for averaging." Use Sero_timeseries_group_analysis.py
     for dose-grouped timeseries averaging instead.

HOW TO RUN
----------
Minimal (map averaging only, injection-center alignment, no atlas, skip ROI step):

    python Sero_group_analysis_sc_maps_inj_centered.py \\
        --group-root /path/to/group_data \\
        --no-roi

With atlas coregistration of the reference subject and a specific reference:

    python Sero_group_analysis_sc_maps_inj_centered.py \\
        --group-root /path/to/group_data \\
        --no-roi \\
        --reference-subject sub_01 \\
        --atlas-reference /path/to/atlas_template.nii

Run `python Sero_group_analysis_sc_maps_inj_centered.py --help` for the full
list of options (patterns, output paths, cache directories, etc.).

PROMPTS / INTERACTIVE STEPS
----------------------------
- If --group-root is omitted, a folder-picker dialog opens to choose it.
- If neither --no-map nor --no-roi is passed, a dialog asks which step(s)
  to run (PSC maps only / ROI time series only / both).
- If map averaging runs and --no-atlas is not passed but --atlas-reference
  is omitted, a file-picker dialog opens to choose an atlas/template
  (cancel it to skip atlas coregistration for this run).
- For each subject missing its injection ROI mask, FSLeyes opens
  automatically (structural underlay + PSC map overlay, if a structural
  scan is found). In FSLeyes: enable Edit Mode, draw the injection-site
  ROI, File > Save As -> save it as exactly
  <subject_dir>/<--roi-filename> (default "inj_roi.nii.gz"), then close
  FSLeyes to continue to the next subject. Once a subject's ROI mask file
  exists, it is reused on future runs without reopening FSLeyes.
"""

import argparse
import glob
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.processing import resample_from_to
from nipype.interfaces import afni
import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


def find_subject_dirs(group_root: Path, subject_glob: str = "*") -> list[Path]:
    subjects = sorted([Path(p) for p in glob.glob(str(group_root / subject_glob)) if Path(p).is_dir()])
    if not subjects:
        raise FileNotFoundError(f"No subject directories found in {group_root} using pattern {subject_glob}")
    return subjects


def resolve_map_root(group_root: Path, map_subdir: str = "sc_maps") -> Path:
    """Return group_root/map_subdir if it exists, otherwise group_root itself.

    This lets --group-root point at the parent "group_data" folder (which also
    contains sibling folders like group_level_results/ and timeseries/) without
    needing to manually append /sc_maps for the PSC map averaging step.
    """
    candidate = group_root / map_subdir
    if candidate.is_dir():
        return candidate
    return group_root


def collect_files(subject_dirs: list[Path], pattern: str) -> dict[Path, list[Path]]:
    files_by_subject = {}
    for subject_dir in subject_dirs:
        matches = sorted(subject_dir.glob(pattern))
        files_by_subject[subject_dir] = matches
    return files_by_subject


def average_nifti_maps(nifti_paths: list[Path], output_path: Path) -> Path:
    if not nifti_paths:
        raise ValueError("No NIfTI files supplied for averaging.")

    first_img = nib.load(str(nifti_paths[0]))
    data_shape = first_img.shape
    affine = first_img.affine
    header = first_img.header.copy()

    all_data = np.zeros((len(nifti_paths),) + data_shape, dtype=np.float64)
    for idx, path in enumerate(nifti_paths):
        img = nib.load(str(path))
        if img.shape != data_shape:
            raise ValueError(
                f"Shape mismatch: {path} has shape {img.shape}, expected {data_shape}"
            )
        all_data[idx] = img.get_fdata(dtype=np.float64)

    mean_data = np.mean(all_data, axis=0)
    mean_img = nib.Nifti1Image(mean_data.astype(np.float32), affine, header=header)
    nib.save(mean_img, str(output_path))
    print(f"[OK] Saved group-average map: {output_path}")
    return output_path


def average_roi_timeseries(subject_dirs: list[Path], roi_pattern: str, output_csv: Path, plot_path: Path | None = None) -> Path:
    time_series_list = []
    subjects = []

    for subject_dir in subject_dirs:
        matches = sorted(subject_dir.glob(roi_pattern))
        if len(matches) == 0:
            print(f"[WARN] no ROI file found for subject {subject_dir} matching '{roi_pattern}'")
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Multiple matching ROI files found for subject {subject_dir}: {matches}"
            )

        subject_file = matches[0]
        values = np.loadtxt(subject_file)
        if values.ndim != 1:
            raise ValueError(f"ROI time series {subject_file} must be a 1D text file")

        time_series_list.append(values)
        subjects.append(subject_dir.name)
        print(f"[OK] loaded ROI time series from {subject_file}")

    if not time_series_list:
        raise RuntimeError("No ROI time series files found for averaging.")

    lengths = {len(ts) for ts in time_series_list}
    if len(lengths) > 1:
        raise ValueError("ROI time series lengths differ across subjects: " + ", ".join(map(str, sorted(lengths))))

    stacked = np.vstack(time_series_list)
    mean_ts = np.mean(stacked, axis=0)

    df = pd.DataFrame(stacked, index=subjects)
    df.loc["group_mean"] = mean_ts
    df.to_csv(output_csv, index=True)
    print(f"[OK] Saved ROI group average CSV: {output_csv}")

    if plot_path is not None:
        plt.figure(figsize=(10, 4))
        x = np.arange(mean_ts.shape[0])
        plt.plot(x, mean_ts, color="tab:blue", lw=2, label="Group mean")
        for subject_name, ts in zip(subjects, stacked):
            plt.plot(x, ts, color="gray", alpha=0.3, lw=1)
        plt.title("ROI percent signal change: group average")
        plt.xlabel("Time point")
        plt.ylabel("Signal change")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=300)
        print(f"[OK] Saved ROI group average plot: {plot_path}")

    return output_csv


def ensure_injection_roi(
    subject_dir: Path,
    map_path: Path,
    roi_filename: str,
    struct_path: Path | None = None,
) -> Path:
    """Get (or interactively create) a subject's injection-site ROI mask via FSLeyes.

    If subject_dir/roi_filename already exists, it is reused as-is. Otherwise FSLeyes is
    launched with the subject's anatomical scan as underlay (if found) and the PSC map
    loaded on top: enable Edit Mode, draw the injection-site ROI, then File > Save As ->
    save it as exactly subject_dir/roi_filename, and close FSLeyes to continue with the
    next subject.
    """
    roi_path = subject_dir / roi_filename
    if roi_path.exists():
        print(f"[OK] {subject_dir.name}: reusing existing ROI mask {roi_path}")
        return roi_path

    fsleyes_exe = shutil.which("fsleyes")
    if fsleyes_exe is None:
        raise RuntimeError(
            "fsleyes was not found on PATH. Please install/activate FSL, or manually create "
            f"'{roi_path}' and rerun."
        )

    fsleyes_cmd = [fsleyes_exe]
    if struct_path is not None:
        fsleyes_cmd += [str(struct_path)]
        fsleyes_cmd += [
            str(map_path),
            "-cm", "red-yellow",
            "-nc", "blue-lightblue",
            "-un",
            "-a", "80",
        ]
    else:
        fsleyes_cmd += [str(map_path)]

    print(
        f"[INFO] Opening FSLeyes for {subject_dir.name}. In FSLeyes: enable Edit Mode "
        f"(toolbar), draw the injection-site ROI, then File > Save As -> save it as exactly "
        f"'{roi_path}', then close FSLeyes to continue."
    )
    subprocess.run(fsleyes_cmd, check=False)

    if not roi_path.exists():
        raise FileNotFoundError(
            f"No ROI mask found at {roi_path} after closing FSLeyes for {subject_dir.name}. "
            "Make sure you saved it with the exact expected filename before closing."
        )

    print(f"[OK] {subject_dir.name}: created ROI mask {roi_path}")
    return roi_path


def find_single_file(subject_dir: Path, pattern: str, description: str) -> Path:
    matches = sorted(subject_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {description} found in {subject_dir} using pattern '{pattern}'"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {description} files found in {subject_dir}: {matches}"
        )
    return matches[0]


def resolve_reference_subject(all_map_files: dict[Path, Path], reference_subject: str | None) -> Path:
    if reference_subject is not None:
        ref_dir = next((s for s in all_map_files if s.name == reference_subject), None)
        if ref_dir is None:
            raise ValueError(f"--reference-subject '{reference_subject}' not found among subject directories.")
        return ref_dir
    ref_dir = sorted(all_map_files.keys())[0]
    print(f"[OK] No --reference-subject given; using '{ref_dir.name}' as reference.")
    return ref_dir


def resample_atlas_to_anatomical_grid(
    atlas_path: Path,
    anatomical_path: Path,
    cache_dir: Path,
) -> Path:
    """Resample atlas to match the reference subject's voxel size to improve registration.

    Handles anisotropic voxel mismatch (e.g., atlas 0.15³ mm vs anatomy 0.1x0.7x0.1 mm)
    by resampling the atlas to the anatomical grid. Result is cached and reused.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    resampled_atlas = cache_dir / "atlas_resampled_to_anat_grid.nii.gz"

    if resampled_atlas.exists():
        print(f"[OK] Using cached resampled atlas: {resampled_atlas}")
        return resampled_atlas

    anat_img = nib.load(str(anatomical_path))
    anat_voxel_size = np.abs(np.diag(anat_img.affine[:3, :3]))
    print(f"[INFO] Reference structural voxel size: {anat_voxel_size} mm")

    print("[INFO] Resampling atlas to reference structural grid...")
    resample = afni.Resample()
    resample.inputs.in_file = str(atlas_path)
    resample.inputs.voxel_size = tuple(anat_voxel_size)
    resample.inputs.outputtype = "NIFTI_GZ"
    resample.inputs.out_file = str(resampled_atlas)
    resample.run()

    print(f"[OK] Resampled atlas saved: {resampled_atlas}")
    return resampled_atlas


def register_reference_to_atlas(
    subject_name: str,
    subject_struct: Path,
    atlas_reference: Path,
    output_dir: Path,
) -> Path:
    """Coregister only the reference subject's structural scan to the atlas.

    Returns the resulting affine transform (.aff12.1D), which can then be applied to any
    map that already shares the reference subject's world coordinate system (e.g. the
    injection-centered group-average signal change map).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    struct_aligned = output_dir / f"{subject_name}_struct_atlas.nii.gz"
    affine_mat = output_dir / f"{subject_name}_struct2atlas.aff12.1D"
    params_file = output_dir / f"{subject_name}_params.1D"

    print(f"[INFO] Registering reference subject '{subject_name}' structural scan to atlas {atlas_reference}")
    coreg = afni.Allineate()
    coreg.inputs.in_file = str(subject_struct)
    coreg.inputs.reference = str(atlas_reference)
    coreg.inputs.out_matrix = str(affine_mat)
    coreg.inputs.out_file = str(struct_aligned)
    coreg.inputs.out_param_file = str(params_file)
    coreg.inputs.warp_type = "shift_rotate"
    coreg.inputs.cost = "nmi"
    coreg.inputs.two_pass = True
    coreg.inputs.verbose = True
    coreg.inputs.overwrite = True
    coreg.run()
    print(f"[OK] Reference structural scan aligned to atlas → {struct_aligned}")

    return affine_mat


def apply_atlas_transform_to_map(
    map_path: Path,
    atlas_reference: Path,
    affine_mat: Path,
    output_path: Path,
) -> Path:
    """Apply the reference subject's struct→atlas affine transform to a signal change map."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Applying reference→atlas transform to {map_path}")
    apply_xfm = afni.Allineate()
    apply_xfm.inputs.in_file = str(map_path)
    apply_xfm.inputs.reference = str(atlas_reference)
    apply_xfm.inputs.in_matrix = str(affine_mat)
    apply_xfm.inputs.master = str(atlas_reference)
    apply_xfm.inputs.final_interpolation = "linear"
    apply_xfm.inputs.verbose = True
    apply_xfm.inputs.overwrite = True
    apply_xfm.inputs.out_file = str(output_path)
    apply_xfm.run()
    print(f"[OK] Atlas-space group-average signal change map saved → {output_path}")

    return output_path


def get_roi_mask_center(mask_path: Path, subject_name: str) -> np.ndarray:
    """Return a subject's injection-site ROI mask centroid in world (mm) coordinates.

    The mask can live on any grid (functional, structural, etc.) since the centroid is
    converted to world/scanner coordinates using the mask's own affine, which is directly
    comparable to the PSC map's world coordinates for the same subject/session.
    """
    img = nib.load(str(mask_path))
    data = img.get_fdata(dtype=np.float64)

    voxel_indices = np.argwhere(data > 0)
    if voxel_indices.size == 0:
        raise ValueError(f"ROI mask {mask_path} contains no nonzero voxels.")

    center_ijk = voxel_indices.mean(axis=0)
    world = apply_affine(img.affine, center_ijk)
    print(
        f"[OK] {subject_name}: ROI mask {mask_path.name} centroid = {world} mm "
        f"({len(voxel_indices)} voxels)"
    )
    return world


def build_common_grid(images: list[nib.Nifti1Image]) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Compute an isotropic output affine/shape covering the bounding box of all input images."""
    voxel_sizes = []
    corners_world = []

    for img in images:
        affine = img.affine
        shape = img.shape
        voxel_sizes.append(np.abs(np.diag(affine[:3, :3])))
        for corner in itertools.product([0, shape[0] - 1], [0, shape[1] - 1], [0, shape[2] - 1]):
            corners_world.append(apply_affine(affine, corner))

    voxel_size = float(np.min(np.concatenate(voxel_sizes)))
    corners_world = np.array(corners_world)
    bbox_min = corners_world.min(axis=0)
    bbox_max = corners_world.max(axis=0)

    shape = tuple(int(np.ceil((bbox_max[i] - bbox_min[i]) / voxel_size)) + 1 for i in range(3))
    out_affine = np.eye(4)
    out_affine[0, 0] = voxel_size
    out_affine[1, 1] = voxel_size
    out_affine[2, 2] = voxel_size
    out_affine[:3, 3] = bbox_min

    return out_affine, shape


def align_maps_by_injection_center(
    all_map_files: dict[Path, Path],
    output_dir: Path,
    roi_filename: str,
    struct_pattern: str,
    ref_dir: Path,
) -> list[Path]:
    """Translate each subject's PSC map so an injection-site ROI (drawn via FSLeyes)
    overlaps across all subjects, then resample everyone onto a shared bounding-box grid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    centers = {}
    for subject_dir, map_path in all_map_files.items():
        try:
            struct_path = find_single_file(subject_dir, struct_pattern, "structural scan")
        except (FileNotFoundError, ValueError) as exc:
            print(f"[WARN] {subject_dir.name}: no anatomical underlay found ({exc}); opening PSC map alone.")
            struct_path = None
        roi_path = ensure_injection_roi(subject_dir, map_path, roi_filename, struct_path)
        centers[subject_dir] = get_roi_mask_center(roi_path, subject_dir.name)

    reference_center = centers[ref_dir]

    shifted_images = {}
    for subject_dir, map_path in all_map_files.items():
        img = nib.load(str(map_path))
        delta = reference_center - centers[subject_dir]
        shifted_affine = img.affine.copy()
        shifted_affine[:3, 3] += delta
        shifted_images[subject_dir] = nib.Nifti1Image(
            img.get_fdata(dtype=np.float64), shifted_affine, header=img.header
        )
        print(f"[OK] {subject_dir.name}: shifting by {delta} mm to align with reference '{ref_dir.name}'")

    common_affine, common_shape = build_common_grid(list(shifted_images.values()))
    reference_grid = nib.Nifti1Image(np.zeros(common_shape, dtype=np.float32), common_affine)

    aligned_paths = []
    for subject_dir, shifted_img in shifted_images.items():
        resampled_img = resample_from_to(shifted_img, reference_grid, order=1)
        out_path = output_dir / f"{subject_dir.name}_injection_centered.nii.gz"
        nib.save(resampled_img, str(out_path))
        aligned_paths.append(out_path)
        print(f"[OK] {subject_dir.name}: resampled onto common grid → {out_path}")

    return aligned_paths


def choose_group_root_via_dialog() -> Path:
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass --group-root manually.")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select group root folder containing subject data")
    root.destroy()

    if not folder:
        raise RuntimeError("No folder selected. Please run the script again and choose a folder.")

    return Path(folder)


def choose_atlas_reference_via_dialog() -> Path | None:
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass --atlas-reference manually.")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select atlas/template anatomical image (or cancel to skip atlas coregistration)",
        filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*")],
    )
    root.destroy()

    if not file_path:
        return None
    return Path(file_path)


def choose_steps_via_dialog() -> tuple[bool, bool]:
    if tk is None:
        raise RuntimeError("tkinter is not available on this system, cannot show step selection dialog.")

    selection = {
        "option": "both"
    }

    def on_select():
        selection["option"] = option_var.get()
        dialog.destroy()

    dialog = tk.Tk()
    dialog.title("Select analysis steps")
    dialog.geometry("360x180")
    dialog.resizable(False, False)

    label = tk.Label(dialog, text="Choose which group analysis steps to run:", pady=10)
    label.pack()

    option_var = tk.StringVar(dialog, value="both")
    tk.Radiobutton(dialog, text="PSC maps only", variable=option_var, value="maps").pack(anchor="w", padx=20)
    tk.Radiobutton(dialog, text="ROI time series only", variable=option_var, value="rois").pack(anchor="w", padx=20)
    tk.Radiobutton(dialog, text="Both PSC maps and ROI time series", variable=option_var, value="both").pack(anchor="w", padx=20)

    button_frame = tk.Frame(dialog, pady=10)
    button_frame.pack()
    tk.Button(button_frame, text="OK", width=10, command=on_select).pack(side="left", padx=10)
    tk.Button(button_frame, text="Cancel", width=10, command=dialog.destroy).pack(side="right", padx=10)

    dialog.mainloop()

    option = selection["option"]
    if option == "maps":
        return True, False
    if option == "rois":
        return False, True
    return True, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average group-level signal change maps or ROI time courses from subject outputs."
    )
    parser.add_argument(
        "--group-root",
        type=Path,
        required=False,
        help="Directory containing one folder per subject. If omitted, a folder picker dialog will open.",
    )
    parser.add_argument(
        "--subject-glob",
        type=str,
        default="*",
        help="Glob pattern for subject folders under group root.",
    )
    parser.add_argument(
        "--map-pattern",
        type=str,
        default="*scm*.nii*",
        help="Glob pattern for subject-level PSC map files.",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        help="Output filename for the average group PSC map.",
    )
    parser.add_argument(
        "--roi-filename",
        type=str,
        default="inj_roi.nii.gz",
        help=(
            "Filename (not a glob) for the injection-site ROI mask inside each subject folder. "
            "If it doesn't exist yet, FSLeyes is launched so you can draw and save it under "
            "this exact name."
        ),
    )
    parser.add_argument(
        "--struct-pattern",
        type=str,
        default="*struct*.nii*",
        help="Glob pattern (relative to each subject folder) for the anatomical scan to show as underlay in FSLeyes.",
    )
    parser.add_argument(
        "--reference-subject",
        type=str,
        help=(
            "Subject directory name to use as the alignment reference when centering maps "
            "on the injection site (defaults to the first subject alphabetically)."
        ),
    )
    parser.add_argument(
        "--injection-cache-dir",
        type=Path,
        help="Directory to write per-subject injection-centered maps.",
    )
    parser.add_argument(
        "--atlas-reference",
        type=Path,
        help=(
            "Atlas/template anatomical image to coregister the reference subject to. Only the "
            "reference subject's structural scan is registered to this atlas; the resulting "
            "transform is then applied to the group-average signal change map. If omitted, a "
            "file picker dialog will open (cancel it to skip atlas coregistration)."
        ),
    )
    parser.add_argument(
        "--atlas-cache-dir",
        type=Path,
        help="Directory to write the atlas-resampled template and reference registration files.",
    )
    parser.add_argument(
        "--atlas-map-output",
        type=Path,
        help="Output filename for the atlas-space group-average signal change map.",
    )
    parser.add_argument(
        "--no-atlas",
        action="store_true",
        help="Skip atlas coregistration entirely (no dialog prompt either).",
    )
    parser.add_argument(
        "--roi-pattern",
        type=str,
        default="PSC_time_series_*.txt",
        help="Glob pattern for subject-level ROI PSC time series files.",
    )
    parser.add_argument(
        "--roi-output",
        type=Path,
        help="Output CSV filename for average ROI time series.",
    )
    parser.add_argument(
        "--roi-plot",
        type=Path,
        help="Optional plot output for the ROI group average time series.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip map averaging.",
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Skip ROI averaging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.group_root is None:
        args.group_root = choose_group_root_via_dialog()
        print(f"Selected group root: {args.group_root}")

    results_dir = args.group_root / "group_level_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Results folder: {results_dir}")

    if not args.no_map and not args.no_roi:
        if tk is not None:
            do_map, do_roi = choose_steps_via_dialog()
            args.no_map = not do_map
            args.no_roi = not do_roi
            print(f"Selected steps: PSC maps = {do_map}, ROI time series = {do_roi}")
        else:
            print("tkinter not available, running both map and ROI steps by default.")

    if not args.no_map and not args.no_atlas and args.atlas_reference is None:
        if tk is not None:
            atlas_file = choose_atlas_reference_via_dialog()
            if atlas_file is not None:
                args.atlas_reference = atlas_file
                print(f"Selected atlas reference: {args.atlas_reference}")
            else:
                print("No atlas reference selected; atlas coregistration will be skipped.")
        else:
            print("tkinter not available, atlas coregistration requires --atlas-reference if desired.")

    subject_dirs = find_subject_dirs(args.group_root, args.subject_glob)
    print(f"Found {len(subject_dirs)} subject directories.")

    if not args.no_map:
        map_root = resolve_map_root(args.group_root)
        if map_root != args.group_root:
            print(f"[OK] Using PSC map subject root: {map_root}")
        map_subject_dirs = find_subject_dirs(map_root, args.subject_glob)

        if args.map_output is None:
            args.map_output = results_dir / "group_average_signal_change_map.nii.gz"
        all_map_files = {}
        for subject_dir in map_subject_dirs:
            matches = sorted(subject_dir.glob(args.map_pattern))
            if len(matches) == 1:
                all_map_files[subject_dir] = matches[0]
            elif len(matches) == 0:
                print(f"[WARN] no map found for {subject_dir} using pattern {args.map_pattern}")
            else:
                raise ValueError(
                    f"Multiple map files found in {subject_dir}: {matches}"
                )

        if not all_map_files:
            raise RuntimeError("No PSC map files found for averaging.")

        ref_dir = resolve_reference_subject(all_map_files, args.reference_subject)

        injection_cache_dir = args.injection_cache_dir or results_dir / "injection_centered_maps"
        aligned_maps = align_maps_by_injection_center(
            all_map_files, injection_cache_dir, args.roi_filename, args.struct_pattern, ref_dir
        )
        average_nifti_maps(aligned_maps, args.map_output)

        if not args.no_atlas and args.atlas_reference is not None:
            atlas_cache_dir = args.atlas_cache_dir or results_dir / "atlas_registered_maps"
            ref_struct = find_single_file(ref_dir, args.struct_pattern, "structural scan")
            atlas_reference = resample_atlas_to_anatomical_grid(args.atlas_reference, ref_struct, atlas_cache_dir)
            affine_mat = register_reference_to_atlas(ref_dir.name, ref_struct, atlas_reference, atlas_cache_dir)

            atlas_map_output = args.atlas_map_output or results_dir / "group_average_signal_change_map_atlas.nii.gz"
            apply_atlas_transform_to_map(args.map_output, atlas_reference, affine_mat, atlas_map_output)

    if not args.no_roi:
        if args.roi_output is None:
            args.roi_output = results_dir / "group_average_roi_timeseries.csv"
        average_roi_timeseries(subject_dirs, args.roi_pattern, args.roi_output, args.roi_plot)


if __name__ == "__main__":
    main()


