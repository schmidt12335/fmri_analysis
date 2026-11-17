#!/usr/bin/env python3
"""Group analysis script for average fMRI percent signal change outputs."""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces import afni

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


def resample_atlas_to_anatomical_grid(
    atlas_path: Path,
    anatomical_path: Path,
    cache_dir: Path,
) -> Path:
    """
    Resample atlas to match anatomical voxel size to improve registration.
    
    Handles anisotropic voxel mismatch (e.g., atlas 0.15³ mm vs anatomy 0.1x0.7x0.1 mm)
    by resampling the atlas to the anatomical grid.
    
    Args:
        atlas_path: Path to atlas/template image.
        anatomical_path: Path to subject anatomical image to use as voxel reference.
        cache_dir: Directory to write resampled atlas.
    
    Returns:
        Path to resampled atlas (created once and reused across subjects).
    """
    resampled_atlas = cache_dir / "atlas_resampled_to_anat_grid.nii.gz"
    
    # Check if we already resampled this atlas for this anatomical grid
    if resampled_atlas.exists():
        print(f"[OK] Using cached resampled atlas: {resampled_atlas}")
        return resampled_atlas
    
    # Load images to extract voxel size from anatomical
    anat_img = nib.load(str(anatomical_path))
    anat_affine = anat_img.affine
    anat_voxel_size = np.abs(np.diag(anat_affine[:3, :3]))
    print(f"[INFO] Anatomical voxel size: {anat_voxel_size} mm")
    
    # Resample atlas to anatomical voxel size using AFNI's Resample
    print(f"[INFO] Resampling atlas to anatomical grid...")
    resample = afni.Resample()
    resample.inputs.in_file = str(atlas_path)
    resample.inputs.voxel_size = tuple(anat_voxel_size)
    resample.inputs.outputtype = "NIFTI_GZ"
    resample.inputs.out_file = str(resampled_atlas)
    resample.run()
    
    print(f"[OK] Resampled atlas saved: {resampled_atlas}")
    return resampled_atlas


def register_subject_to_atlas(
    subject_name: str,
    subject_struct: Path,
    subject_map: Path,
    atlas_reference: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    struct_aligned = output_dir / f"{subject_name}_struct_atlas.nii.gz"
    map_aligned = output_dir / f"{subject_name}_psc_atlas.nii.gz"
    affine_mat = output_dir / f"{subject_name}_struct2atlas.aff12.1D"
    params_file = output_dir / f"{subject_name}_params.1D"

    print(f"[INFO] Registering {subject_name} structural image to atlas {atlas_reference}")
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
    #coreg.inputs.center_of_mass = "cm"
    coreg.run()
    print(f"[OK] Structural scan aligned to atlas → {struct_aligned}")

    print(f"[INFO] Applying transform to PSC map {subject_map}")
    apply_affine = afni.Allineate()
    apply_affine.inputs.in_file = str(subject_map)
    apply_affine.inputs.reference = str(atlas_reference)
    apply_affine.inputs.in_matrix = str(affine_mat)
    apply_affine.inputs.master = str(atlas_reference)
    apply_affine.inputs.final_interpolation = "linear"
    apply_affine.inputs.verbose = True
    apply_affine.inputs.overwrite = True
    #apply_affine.inputs.center_of_mass = "cm"
    apply_affine.inputs.out_file = str(map_aligned)
    apply_affine.run()
    print(f"[OK] PSC map aligned to atlas → {map_aligned}")

    return map_aligned


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


def choose_atlas_reference_via_dialog() -> Path | None:
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass --atlas-reference manually.")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select atlas/template anatomical image",
        filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*")],
    )
    root.destroy()

    if not file_path:
        return None
    return Path(file_path)


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
        "--atlas-reference",
        type=Path,
        help="Atlas/template anatomical image to register subject maps into.",
    )
    parser.add_argument(
        "--struct-pattern",
        type=str,
        default="*struct*.nii*",
        help="Glob pattern for subject anatomical scans used for atlas registration.",
    )
    parser.add_argument(
        "--atlas-cache-dir",
        type=Path,
        help="Directory to write per-subject atlas-registered PSC maps.",
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

    if not args.no_map and args.atlas_reference is None:
        if tk is not None:
            atlas_file = choose_atlas_reference_via_dialog()
            if atlas_file is not None:
                args.atlas_reference = atlas_file
                print(f"Selected atlas reference: {args.atlas_reference}")
            else:
                print("No atlas reference selected; atlas registration will be skipped.")
        else:
            print("tkinter not available, atlas registration requires --atlas-reference if desired.")

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

        if args.atlas_reference is not None:
            atlas_cache_dir = args.atlas_cache_dir or results_dir / "atlas_registered_maps"
            atlas_cache_dir.mkdir(parents=True, exist_ok=True)
            atlas_maps = []
            atlas_reference = args.atlas_reference
            # Resample atlas to first subject's anatomical grid once, then reuse
            first_subj_dir = list(all_map_files.keys())[0]
            first_struct = find_single_file(first_subj_dir, args.struct_pattern, "structural scan")
            atlas_reference = resample_atlas_to_anatomical_grid(atlas_reference, first_struct, atlas_cache_dir)
            print(f"[OK] Using resampled atlas for all registrations: {atlas_reference}")
            for subject_dir, subject_map in all_map_files.items():
                try:
                    subject_struct = find_single_file(subject_dir, args.struct_pattern, "structural scan")
                except (FileNotFoundError, ValueError) as exc:
                    raise RuntimeError(
                        f"Could not find structural scan for atlas registration in {subject_dir}: {exc}"
                    )

                atlas_map = register_subject_to_atlas(
                    subject_dir.name,
                    subject_struct,
                    subject_map,
                    atlas_reference,
                    atlas_cache_dir,
                )
                atlas_maps.append(atlas_map)

            if not atlas_maps:
                raise RuntimeError("No atlas-registered PSC maps were generated.")
            average_nifti_maps(atlas_maps, args.map_output)
        else:
            average_nifti_maps(list(all_map_files.values()), args.map_output)

    if not args.no_roi:
        if args.roi_output is None:
            args.roi_output = results_dir / "group_average_roi_timeseries.csv"
        average_roi_timeseries(subject_dirs, args.roi_pattern, args.roi_output, args.roi_plot)


if __name__ == "__main__":
    main()


