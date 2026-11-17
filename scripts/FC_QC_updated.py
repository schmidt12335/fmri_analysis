#!/usr/bin/env python3
"""
Rigidly coregister partial-coverage subject data to an atlas, transform atlas ROIs
into functional space, extract ROI time courses, and generate a simple FC-QC plot.

Key design choices
------------------
1. Structural -> functional and structural -> atlas are both rigid-body only
   (3 translations + 3 rotations; no scaling or shearing).
2. The full 4D functional image is retained for time-course extraction.
   A 3D temporal mean of the first N volumes is used only as the registration target.
3. Structural -> atlas uses a restricted center-of-mass initialization suitable for
   partial coverage. For a coronal slab (limited A-P coverage), use ATLAS_CMASS="+xz".
4. Atlas ROI transforms are concatenated using AFNI's base-to-source matrix convention.
5. ROI time courses are extracted with 3dmaskave, avoiding loading the full 4D image
   into RAM.

Run examples
------------
    python FC_QC_updated.py
    python FC_QC_updated.py --force

Use --force after changing registration settings, otherwise existing outputs are reused.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Subject data
    "subject_dir": None,
    "group_root": "/home/pschmidt/fmri_analysis/QC/QC_in/data",
    "subject_glob": "sub*",

    # Input image patterns
    "struct_pattern": "*struct*.nii*",
    "func_pattern": "*func*.nii*",

    # Atlas and ROI masks
    "atlas_reference": (
        "/home/pschmidt/fmri_analysis/QC/QC_in/atlas/"
        "SIGMA_InVivo_Brain_Template_Masked.nii"
    ),
    "roi_masks_dir": "/home/pschmidt/fmri_analysis/QC/QC_in/rois",
    "roi_mask_glob": "*.nii*",

    # Outputs
    "output_dir": "/home/pschmidt/fmri_analysis/QC/QC_out_2",
    "plot_dir": "/home/pschmidt/fmri_analysis/QC/QC_out_2",

    # Functional reference used for structural -> functional registration
    "func_reference_volumes": 600,

    # Registration settings
    "struct_to_func_cost": "nmi",
    "atlas_cost": "lpa+ZZ",
    "twobest": "MAX",

    # Center-of-mass initialization for structural -> atlas.
    # AFNI coordinates are physical DICOM axes:
    #   x = left-right, y = anterior-posterior, z = inferior-superior
    # For a CORONAL slab with limited A-P coverage, use "+xz".
    # For an AXIAL slab with limited I-S coverage, use "+xy".
    # For a SAGITTAL slab with limited L-R coverage, use "+yz".
    "atlas_cmass": "+xz",

    # Usually unnecessary if structural and functional were acquired in the same session.
    # Valid examples: None, "", "+a", "+xy", "+xz", "+yz".
    "struct_to_func_cmass": None,

    # Optional search limits. Leave as None to use AFNI defaults.
    # These restrict the optimizer's search range but do not add degrees of freedom.
    "max_rotation_deg": None,
    "max_shift_mm": None,

    # Optional FC-QC plot. These names must match ROI filenames without .nii/.nii.gz.
    "make_fc_qc_plot": True,
    "qc_seed_roi": "S1bf_l",
    "qc_target_roi": "S1bf_r",
    "qc_control_roi": "ACA_l",
    "fc_threshold": 0.1,
}
# =============================================================================


REQUIRED_AFNI_PROGRAMS = (
    "3dAllineate",
    "3dTstat",
    "3dmaskave",
    "cat_matvec",
)


def normalize_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def nifti_stem(path: Path) -> str:
    """Return filename without .nii or .nii.gz."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def run_command(
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Print and run a command with consistent error reporting."""
    print("[COMMAND]", " ".join(command))
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(command)}\n\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{stderr}"
        ) from error


def check_required_programs() -> None:
    missing = [program for program in REQUIRED_AFNI_PROGRAMS if shutil.which(program) is None]
    if missing:
        raise RuntimeError(
            "Required AFNI programs were not found on PATH: " + ", ".join(missing)
        )


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
    return Path(folder).resolve()


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
    return Path(file_path).resolve()


def find_subject_dirs(group_root: Path, subject_glob: str = "*") -> list[Path]:
    direct_subject_paths = sorted(
        Path(path)
        for path in glob.glob(str(group_root / subject_glob))
        if Path(path).is_dir()
    )
    if direct_subject_paths:
        return direct_subject_paths

    nested_subject_paths = sorted(
        path
        for path in group_root.iterdir()
        if path.is_dir()
        and (any(path.glob("*struct*.nii*")) or any(path.glob("*func*.nii*")))
    )
    if nested_subject_paths:
        return nested_subject_paths

    raise FileNotFoundError(
        f"No subject directories found in {group_root} using pattern {subject_glob}"
    )


def find_single_file(subject_dir: Path, pattern: str, description: str) -> Path:
    matches = sorted(subject_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {description} found in {subject_dir} using pattern {pattern}"
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple {description} files found in {subject_dir} using pattern "
            f"{pattern}:\n" + "\n".join(f"  {path}" for path in matches)
        )
    return matches[0].resolve()


def add_optional_registration_limits(command: list[str]) -> None:
    max_rotation = CONFIG.get("max_rotation_deg")
    max_shift = CONFIG.get("max_shift_mm")

    if max_rotation is not None:
        command.extend(["-maxrot", str(float(max_rotation))])
    if max_shift is not None:
        command.extend(["-maxshf", str(float(max_shift))])


def add_cmass_option(command: list[str], cmass_setting: str | None) -> None:
    if cmass_setting is None:
        return

    cmass_setting = str(cmass_setting).strip()
    if not cmass_setting:
        return

    if cmass_setting == "cmass":
        command.append("-cmass")
    elif cmass_setting.startswith("+"):
        command.append(f"-cmass{cmass_setting}")
    else:
        raise ValueError(
            "cmass setting must be None, an empty string, 'cmass', or one of "
            "'+a', '+xy', '+xz', '+yz'."
        )


def create_functional_reference(
    subject_name: str,
    func_path: Path,
    output_dir: Path,
    max_volumes: int,
    force: bool,
) -> Path:
    """Create a 3D temporal mean from the first max_volumes functional volumes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    func_img = nib.load(str(func_path))
    if len(func_img.shape) != 4:
        raise ValueError(f"Functional image must be 4D: {func_path}")

    n_volumes = min(int(func_img.shape[3]), int(max_volumes))
    if n_volumes < 1:
        raise ValueError(f"Functional image contains no volumes: {func_path}")

    output_path = output_dir / f"{subject_name}_func_mean_first{n_volumes}.nii.gz"

    if output_path.exists() and not force:
        print(f"[SKIP] Functional reference already exists: {output_path}")
        return output_path

    selector = f"{func_path}[0..{n_volumes - 1}]"
    command = [
        "3dTstat",
        "-mean",
        "-prefix",
        str(output_path),
        "-overwrite",
        selector,
    ]
    run_command(command)
    print(f"[OK] Functional reference created: {output_path}")
    return output_path


def register_struct_to_functional(
    subject_name: str,
    subject_struct: Path,
    functional_reference: Path,
    output_dir: Path,
    force: bool,
) -> tuple[Path, Path]:
    """Rigidly register structural data to the 3D functional mean."""
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / f"{subject_name}_struct2func.aff12.1D"
    params_path = output_dir / f"{subject_name}_struct2func_params.1D"
    aligned_path = output_dir / f"{subject_name}_struct_func.nii.gz"

    if matrix_path.exists() and aligned_path.exists() and not force:
        print(f"[SKIP] Structural-to-functional registration exists for {subject_name}")
        return matrix_path, aligned_path

    command = [
        "3dAllineate",
        "-source",
        str(subject_struct),
        "-base",
        str(functional_reference),
        "-prefix",
        str(aligned_path),
        "-1Dmatrix_save",
        str(matrix_path),
        "-1Dparam_save",
        str(params_path),
        "-warp",
        "shift_rotate",
        "-cost",
        str(CONFIG["struct_to_func_cost"]),
        "-source_automask+2",
        "-twopass",
        "-twobest",
        str(CONFIG["twobest"]),
        "-float",
        "-verb",
        "-overwrite",
    ]

    add_cmass_option(command, CONFIG.get("struct_to_func_cmass"))
    add_optional_registration_limits(command)

    print(f"[INFO] Registering structural -> functional for {subject_name}")
    run_command(command)
    print(f"[OK] Structural aligned to functional: {aligned_path}")
    return matrix_path, aligned_path


def register_struct_to_atlas(
    subject_name: str,
    subject_struct: Path,
    atlas_reference: Path,
    output_dir: Path,
    force: bool,
) -> tuple[Path, Path]:
    """Rigidly register the original structural image directly to the atlas."""
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / f"{subject_name}_struct2atlas.aff12.1D"
    params_path = output_dir / f"{subject_name}_struct2atlas_params.1D"
    aligned_path = output_dir / f"{subject_name}_struct_atlas.nii.gz"

    if matrix_path.exists() and aligned_path.exists() and not force:
        print(f"[SKIP] Structural-to-atlas registration exists for {subject_name}")
        return matrix_path, aligned_path

    command = [
        "3dAllineate",
        "-source",
        str(subject_struct),
        "-base",
        str(atlas_reference),
        "-prefix",
        str(aligned_path),
        "-1Dmatrix_save",
        str(matrix_path),
        "-1Dparam_save",
        str(params_path),
        "-warp",
        "shift_rotate",
        "-cost",
        str(CONFIG["atlas_cost"]),
        "-source_automask+2",
        "-twopass",
        "-twobest",
        str(CONFIG["twobest"]),
        "-float",
        "-verb",
        "-overwrite",
    ]

    add_cmass_option(command, CONFIG.get("atlas_cmass"))
    add_optional_registration_limits(command)

    print(f"[INFO] Registering structural -> atlas for {subject_name}")
    run_command(command)
    print(f"[OK] Structural aligned to atlas: {aligned_path}")
    return matrix_path, aligned_path


def register_subject(
    subject_name: str,
    subject_struct: Path,
    subject_func: Path,
    atlas_reference: Path,
    output_dir: Path,
    force: bool,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """
    Register one subject.

    Returns
    -------
    struct2atlas_matrix, struct2func_matrix, full_func, func_reference,
    struct_in_atlas, struct_in_func
    """
    functional_reference = create_functional_reference(
        subject_name=subject_name,
        func_path=subject_func,
        output_dir=output_dir,
        max_volumes=int(CONFIG["func_reference_volumes"]),
        force=force,
    )

    struct2func_matrix, struct_in_func = register_struct_to_functional(
        subject_name=subject_name,
        subject_struct=subject_struct,
        functional_reference=functional_reference,
        output_dir=output_dir,
        force=force,
    )

    struct2atlas_matrix, struct_in_atlas = register_struct_to_atlas(
        subject_name=subject_name,
        subject_struct=subject_struct,
        atlas_reference=atlas_reference,
        output_dir=output_dir,
        force=force,
    )

    return (
        struct2atlas_matrix,
        struct2func_matrix,
        subject_func,
        functional_reference,
        struct_in_atlas,
        struct_in_func,
    )


def convert_aff12_to_plain_matvec(matrix_path: Path, output_path: Path) -> Path:
    """
    Convert a single-row .aff12.1D matrix to cat_matvec's plain 3x4 format.

    cat_matvec permits at most one multi-row .aff12.1D input. Converting both
    single transforms to plain 3x4 files avoids that ambiguity entirely.
    """
    values: list[float] = []
    for line in matrix_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.extend(float(value) for value in stripped.split())

    if len(values) != 12:
        raise ValueError(
            f"Expected exactly 12 values in single-transform matrix {matrix_path}, "
            f"but found {len(values)}."
        )

    rows = [values[0:4], values[4:8], values[8:12]]
    output_path.write_text(
        "\n".join(" ".join(f"{value:.12g}" for value in row) for row in rows)
        + "\n"
    )
    return output_path


def compose_func_to_atlas_coordinate_matrix(
    subject_name: str,
    struct2atlas_matrix: Path,
    struct2func_matrix: Path,
    output_dir: Path,
    force: bool,
) -> Path:
    """
    Build the matrix needed to resample an atlas-space ROI onto the functional grid.

    Matrices saved by 3dAllineate are base -> source coordinate transforms:
        struct2func:  functional coordinates -> structural coordinates
        struct2atlas: atlas coordinates      -> structural coordinates

    3dAllineate applying an atlas ROI onto a functional master needs:
        functional coordinates -> atlas coordinates

    Therefore:
        M_func_to_atlas = inverse(M_struct2atlas) * M_struct2func

    cat_matvec applies the second command-line transform after the first, so the
    command-line order is:
        struct2func  struct2atlas -I
    """
    output_path = output_dir / f"{subject_name}_func_to_atlas_for_roi_apply.aff12.1D"

    if output_path.exists() and not force:
        return output_path

    plain_struct2func = convert_aff12_to_plain_matvec(
        struct2func_matrix,
        output_dir / f"{subject_name}_struct2func.matvec",
    )
    plain_struct2atlas = convert_aff12_to_plain_matvec(
        struct2atlas_matrix,
        output_dir / f"{subject_name}_struct2atlas.matvec",
    )

    command = [
        "cat_matvec",
        "-ONELINE",
        str(plain_struct2func),
        str(plain_struct2atlas),
        "-I",
    ]
    result = run_command(command, capture_output=True)
    output_path.write_text(result.stdout.strip() + "\n")
    print(f"[OK] Composite functional -> atlas coordinate matrix: {output_path}")
    return output_path


def resample_mask_to_functional(
    roi_mask_path: Path,
    functional_reference: Path,
    composite_func_to_atlas_matrix: Path,
    output_dir: Path,
    subject_name: str,
    force: bool,
) -> Path:
    """Resample one atlas-space ROI mask onto the functional grid using NN."""
    roi_name = nifti_stem(roi_mask_path)
    output_path = output_dir / f"{subject_name}_{roi_name}_funcspace.nii.gz"

    if output_path.exists() and not force:
        return output_path

    command = [
        "3dAllineate",
        "-source",
        str(roi_mask_path),
        "-master",
        str(functional_reference),
        "-1Dmatrix_apply",
        str(composite_func_to_atlas_matrix),
        "-final",
        "NN",
        "-prefix",
        str(output_path),
        "-overwrite",
    ]
    run_command(command)

    mask_data = np.asanyarray(nib.load(str(output_path)).dataobj)
    voxel_count = int(np.count_nonzero(mask_data > 0))
    if voxel_count == 0:
        raise ValueError(
            f"Resampled ROI is empty for {subject_name}/{roi_name}: {output_path}"
        )

    print(f"[OK] ROI in functional space: {output_path} ({voxel_count} voxels)")
    return output_path


def extract_roi_timecourse(func_path: Path, roi_mask_path: Path) -> np.ndarray:
    """Extract a mean ROI time course with AFNI 3dmaskave."""
    command = [
        "3dmaskave",
        "-quiet",
        "-mask",
        str(roi_mask_path),
        str(func_path),
    ]
    result = run_command(command, capture_output=True)
    timecourse = np.fromstring(result.stdout, sep=" ", dtype=float)

    if timecourse.size == 0:
        raise ValueError(
            f"3dmaskave returned an empty time course for mask {roi_mask_path}"
        )
    if not np.all(np.isfinite(timecourse)):
        raise ValueError(
            f"Non-finite values found in ROI time course for mask {roi_mask_path}"
        )
    return timecourse


def save_subject_roi_timeseries(
    subject_name: str,
    roi_name: str,
    timecourse: np.ndarray,
    output_csv: Path,
    plot_path: Path | None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(
        {
            "timepoint": np.arange(timecourse.size, dtype=int),
            "signal": timecourse,
        }
    )
    dataframe.to_csv(output_csv, index=False)
    print(f"[OK] Saved ROI time course: {output_csv}")

    if plot_path is None:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(dataframe["timepoint"], dataframe["signal"], linewidth=1)
    axis.set_title(f"{subject_name}: {roi_name}")
    axis.set_xlabel("Timepoint")
    axis.set_ylabel("Mean signal")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f"[OK] Saved ROI plot: {plot_path}")


def load_saved_timecourse(csv_path: Path) -> np.ndarray:
    dataframe = pd.read_csv(csv_path)

    if "signal" in dataframe.columns:
        values = dataframe["signal"].to_numpy(dtype=float)
    else:
        # Backward compatibility with the old CSV format:
        # timepoint index, subject column, group_mean column.
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        candidate_columns = [column for column in numeric_columns if column != "timepoint"]
        if not candidate_columns:
            raise ValueError(f"No numeric signal column found in {csv_path}")
        values = dataframe[candidate_columns[0]].to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError(f"Empty time course in {csv_path}")
    return values


def calculate_fc_categories(
    target_correlations: np.ndarray,
    control_correlations: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return percentages: specific, non-specific, no FC, spurious FC."""
    if target_correlations.size != control_correlations.size:
        raise ValueError("Correlation arrays must have equal lengths.")
    if target_correlations.size == 0:
        return np.zeros(4, dtype=float)

    specific = (target_correlations > threshold) & (control_correlations < threshold)
    non_specific = (
        (target_correlations > threshold) & (control_correlations > threshold)
    )
    no_fc = (
        (np.abs(target_correlations) < threshold)
        & (np.abs(control_correlations) < threshold)
    )
    spurious = ~(specific | non_specific | no_fc)

    counts = np.array(
        [specific.sum(), non_specific.sum(), no_fc.sum(), spurious.sum()],
        dtype=float,
    )
    return counts / target_correlations.size * 100.0


def make_fc_qc_plot(
    timecourses: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    seed_roi = str(CONFIG["qc_seed_roi"])
    target_roi = str(CONFIG["qc_target_roi"])
    control_roi = str(CONFIG["qc_control_roi"])
    threshold = float(CONFIG["fc_threshold"])

    subject_names: list[str] = []
    target_correlations: list[float] = []
    control_correlations: list[float] = []

    for subject_name, subject_timecourses in timecourses.items():
        required = (seed_roi, target_roi, control_roi)
        if not all(roi_name in subject_timecourses for roi_name in required):
            print(
                f"[WARN] Skipping FC-QC for {subject_name}; missing one of: "
                f"{seed_roi}, {target_roi}, {control_roi}"
            )
            continue

        seed = subject_timecourses[seed_roi]
        target = subject_timecourses[target_roi]
        control = subject_timecourses[control_roi]

        min_length = min(seed.size, target.size, control.size)
        if min_length < 2:
            print(f"[WARN] Skipping FC-QC for {subject_name}; time course too short")
            continue

        seed = seed[:min_length]
        target = target[:min_length]
        control = control[:min_length]

        r_target = float(np.corrcoef(seed, target)[0, 1])
        r_control = float(np.corrcoef(seed, control)[0, 1])

        if not np.isfinite(r_target) or not np.isfinite(r_control):
            print(f"[WARN] Skipping FC-QC for {subject_name}; invalid correlation")
            continue

        subject_names.append(subject_name)
        target_correlations.append(r_target)
        control_correlations.append(r_control)

    if not subject_names:
        print("[WARN] No valid subjects available for FC-QC plot")
        return

    target_array = np.asarray(target_correlations, dtype=float)
    control_array = np.asarray(control_correlations, dtype=float)
    category_percentages = calculate_fc_categories(
        target_array,
        control_array,
        threshold,
    )

    category_names = ["Specific FC", "Non-specific FC", "No FC", "Spurious FC"]
    print("[INFO] FC categories:")
    for name, value in zip(category_names, category_percentages):
        print(f"  {name:16s}: {value:6.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(target_array, control_array, label="Subjects")

    for subject_name, x_value, y_value in zip(
        subject_names,
        target_array,
        control_array,
    ):
        axis.annotate(
            subject_name,
            xy=(x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    axis.axvline(threshold, linestyle="--", linewidth=1, color="black")
    axis.axvline(-threshold, linestyle="--", linewidth=1, color="black")
    axis.axhline(threshold, linestyle="--", linewidth=1, color="black")
    axis.axhline(-threshold, linestyle="--", linewidth=1, color="black")

    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_xlabel(f"Correlation {seed_roi} to {target_roi} (r)")
    axis.set_ylabel(f"Correlation {seed_roi} to {control_roi} (r)")
    axis.set_title("Functional-connectivity QC")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)

    summary_text = "\n".join(
        f"{name}: {value:.1f}%"
        for name, value in zip(category_names, category_percentages)
    )
    axis.text(
        1.02,
        0.98,
        summary_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"[OK] Saved FC-QC plot: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rigidly coregister partial-coverage subject data to an atlas, "
            "transform atlas ROIs to functional space, and extract time courses."
        )
    )
    parser.add_argument("--group-root", type=Path, default=CONFIG.get("group_root"))
    parser.add_argument("--subject-dir", type=Path, default=CONFIG.get("subject_dir"))
    parser.add_argument(
        "--subject-glob",
        type=str,
        default=CONFIG.get("subject_glob", "*"),
    )
    parser.add_argument(
        "--atlas-reference",
        type=Path,
        default=CONFIG.get("atlas_reference"),
    )
    parser.add_argument(
        "--struct-pattern",
        type=str,
        default=CONFIG.get("struct_pattern", "*struct*.nii*"),
    )
    parser.add_argument(
        "--func-pattern",
        type=str,
        default=CONFIG.get("func_pattern", "*func*.nii*"),
    )
    parser.add_argument(
        "--roi-masks-dir",
        type=Path,
        default=CONFIG.get("roi_masks_dir"),
    )
    parser.add_argument(
        "--roi-mask-glob",
        type=str,
        default=CONFIG.get("roi_mask_glob", "*.nii*"),
    )
    parser.add_argument("--output-dir", type=Path, default=CONFIG.get("output_dir"))
    parser.add_argument("--plot-dir", type=Path, default=CONFIG.get("plot_dir"))
    parser.add_argument(
        "--use-dialogs",
        action="store_true",
        help="Open dialogs for missing paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute registrations, transformed masks, CSVs, and plots even "
            "when outputs already exist."
        ),
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.group_root = normalize_optional_path(args.group_root)
    args.subject_dir = normalize_optional_path(args.subject_dir)
    args.atlas_reference = normalize_optional_path(args.atlas_reference)
    args.roi_masks_dir = normalize_optional_path(args.roi_masks_dir)
    args.output_dir = normalize_optional_path(args.output_dir)
    args.plot_dir = normalize_optional_path(args.plot_dir)

    use_dialogs = args.use_dialogs or (
        args.subject_dir is None and args.group_root is None
    ) or args.atlas_reference is None or args.roi_masks_dir is None

    if use_dialogs:
        print("[INFO] Prompting for missing paths")
        if args.subject_dir is None and args.group_root is None:
            args.subject_dir = choose_folder_via_dialog("Select subject folder")
        if args.atlas_reference is None:
            args.atlas_reference = choose_file_via_dialog(
                "Select atlas reference",
                [("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")],
            )
        if args.roi_masks_dir is None:
            args.roi_masks_dir = choose_folder_via_dialog("Select ROI-mask folder")
        if args.output_dir is None:
            args.output_dir = choose_folder_via_dialog("Select output folder")
        if args.plot_dir is None:
            args.plot_dir = choose_folder_via_dialog("Select plot folder")

    if args.atlas_reference is None:
        raise ValueError("Atlas reference is required.")
    if args.roi_masks_dir is None:
        raise ValueError("ROI-mask directory is required.")
    if args.subject_dir is None and args.group_root is None:
        raise ValueError("Provide either --subject-dir or --group-root.")

    if args.output_dir is None:
        root = args.subject_dir if args.subject_dir is not None else args.group_root
        if root is None:
            raise ValueError("Cannot infer output directory.")
        args.output_dir = root / "FC_QC_results"

    return args


def main() -> None:
    args = resolve_paths(parse_args())
    check_required_programs()

    print("=" * 80)
    print("PATH CONFIGURATION")
    print("=" * 80)
    for key, value in {
        "subject_dir": args.subject_dir,
        "group_root": args.group_root,
        "atlas_reference": args.atlas_reference,
        "roi_masks_dir": args.roi_masks_dir,
        "output_dir": args.output_dir,
        "plot_dir": args.plot_dir,
    }.items():
        print(f"  {key:20s}: {value}")
    print(f"  force               : {args.force}")
    print(f"  atlas_cmass         : {CONFIG['atlas_cmass']}")
    print(f"  atlas_cost          : {CONFIG['atlas_cost']}")
    print(f"  warp_type           : shift_rotate (rigid, 6 parameters)")
    print("=" * 80)

    if args.subject_dir is not None:
        if (
            any(args.subject_dir.glob(args.struct_pattern))
            or any(args.subject_dir.glob(args.func_pattern))
        ):
            subject_dirs = [args.subject_dir]
        else:
            subject_dirs = find_subject_dirs(args.subject_dir, args.subject_glob)
    else:
        if args.group_root is None:
            raise ValueError("Group root is missing.")
        subject_dirs = find_subject_dirs(args.group_root, args.subject_glob)

    print(f"[OK] Found {len(subject_dirs)} subject directories")

    roi_masks = sorted(args.roi_masks_dir.glob(args.roi_mask_glob))
    if not roi_masks:
        raise FileNotFoundError(
            f"No ROI masks found in {args.roi_masks_dir} using {args.roi_mask_glob}"
        )
    print(f"[OK] Found {len(roi_masks)} ROI masks")

    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    subject_transforms: dict[
        str,
        tuple[Path, Path, Path, Path, Path, Path],
    ] = {}

    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        subject_output_dir = results_dir / subject_name
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        struct_path = find_single_file(
            subject_dir,
            args.struct_pattern,
            "structural image",
        )
        func_path = find_single_file(
            subject_dir,
            args.func_pattern,
            "functional image",
        )

        subject_transforms[subject_name] = register_subject(
            subject_name=subject_name,
            subject_struct=struct_path,
            subject_func=func_path,
            atlas_reference=args.atlas_reference,
            output_dir=subject_output_dir,
            force=args.force,
        )

    all_timecourses: dict[str, dict[str, np.ndarray]] = {}

    for subject_name, transform_data in subject_transforms.items():
        (
            struct2atlas_matrix,
            struct2func_matrix,
            full_func_path,
            functional_reference,
            _struct_in_atlas,
            _struct_in_func,
        ) = transform_data

        subject_output_dir = results_dir / subject_name
        all_timecourses[subject_name] = {}

        composite_matrix = compose_func_to_atlas_coordinate_matrix(
            subject_name=subject_name,
            struct2atlas_matrix=struct2atlas_matrix,
            struct2func_matrix=struct2func_matrix,
            output_dir=subject_output_dir,
            force=args.force,
        )

        for roi_mask_path in roi_masks:
            roi_name = nifti_stem(roi_mask_path)
            output_csv = subject_output_dir / f"roi_timeseries_{roi_name}.csv"

            plot_path = None
            if args.plot_dir is not None:
                plot_path = (
                    args.plot_dir
                    / subject_name
                    / f"roi_timeseries_{roi_name}.png"
                )

            if output_csv.exists() and not args.force:
                print(f"[SKIP] Time course exists for {subject_name}/{roi_name}")
                timecourse = load_saved_timecourse(output_csv)
                all_timecourses[subject_name][roi_name] = timecourse
                continue

            functional_mask = resample_mask_to_functional(
                roi_mask_path=roi_mask_path,
                functional_reference=functional_reference,
                composite_func_to_atlas_matrix=composite_matrix,
                output_dir=subject_output_dir,
                subject_name=subject_name,
                force=args.force,
            )

            timecourse = extract_roi_timecourse(
                func_path=full_func_path,
                roi_mask_path=functional_mask,
            )
            all_timecourses[subject_name][roi_name] = timecourse

            save_subject_roi_timeseries(
                subject_name=subject_name,
                roi_name=roi_name,
                timecourse=timecourse,
                output_csv=output_csv,
                plot_path=plot_path,
            )

    if bool(CONFIG.get("make_fc_qc_plot", True)):
        qc_plot_dir = args.plot_dir if args.plot_dir is not None else results_dir
        make_fc_qc_plot(
            timecourses=all_timecourses,
            output_path=qc_plot_dir / "FC_QC_scatter.png",
        )

    print("[OK] Processing complete")
    print(
        "[IMPORTANT] Registration outputs must still be inspected visually, "
        "especially because the subject structural image has partial coverage."
    )


if __name__ == "__main__":
    main()
