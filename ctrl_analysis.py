#!/usr/bin/env python3
"""Group-vs-control timeseries analysis (BLuSH ctrl_analysis).

What it does
------------
1. Loads per-subject percent-signal-change (PSC) timeseries .txt files for a
   control group and one or more experimental groups (any file with
   "PSC_time_series" in its name inside a folder is picked up automatically).
2. Averages subjects within each group and computes an asymmetric
   (semi) standard deviation band around each group mean.
3. For every experimental group, plots its mean timeseries against the
   control's mean timeseries (optionally including the group-minus-control
   difference curve), shades the baseline/injection/signal windows, and adds
   an inset bar plot of subject-wise signal changes with a Welch's t-test and
   significance stars.
4. After each plot is shown, asks whether to save it (300 dpi PNG) into that
   group's own data folder.

How to configure it
--------------------
Edit the constants below before running:
- BASELINE_WINDOW / INJECTION_WINDOW / SIGNAL_WINDOW: frame ranges (1 frame
  = 1 second) used for the shaded regions and the baseline-vs-signal change
  used in the stats/inset plot.
- SMOOTH_WINDOW: moving-average window (in frames) applied to the mean and
  std traces before plotting; 1 disables smoothing.
- CONTROL_NAME / CONTROL_FOLDER: display name and data folder for the
  control group. Leave CONTROL_FOLDER = "" to be prompted for a folder.
- GROUP_FOLDERS: dict of {group name: folder path}. Leave as {} to be
  prompted interactively (you can add as many groups as you like, one at a
  time, until you cancel the name prompt).
- CONTROL_INSET_LABEL / GROUP_INSET_LABELS: optional short labels used only
  for the inset bar plot's x-axis ticks (falls back to the names above).

Running it
----------
Just run the script (`python ctrl_analysis.py`). If CONTROL_FOLDER/
GROUP_FOLDERS are filled in, no dialogs are needed except the yes/no
"plot the difference curve?" and "save this plot?" prompts. Otherwise,
folder-picker/name dialogs guide you through selecting the control and each
group.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 110,
})

# Fixed analysis windows (frames), edit here to change them everywhere.
BASELINE_WINDOW = (180, 480)
INJECTION_WINDOW = (600, 1200)
SIGNAL_WINDOW = (1800, 2100)
SMOOTH_WINDOW = 60

# Optional: point these at folders to skip the file-picker dialogs. Every file
# with "PSC_time_series" in its name inside a folder is picked up automatically,
# so the number of subject files per group/control is not fixed.
# Leave CONTROL_FOLDER empty and/or GROUP_FOLDERS empty ({}) to be prompted instead.
CONTROL_NAME = "Ctrl-AAV + 1mM Sero"
CONTROL_FOLDER: str = "/Volumes/pschmidt/fmri_analysis/AnalysedData/Sero/group_data/timeseries/Ctrl"
EXP_NAME = "Sero-AAV + 1mM Sero"
GROUP_FOLDERS: dict[str, str] = {
        "Sero-AAV + 1mM Sero": "/Volumes/pschmidt/fmri_analysis/AnalysedData/Sero/group_data/timeseries/1mM",
}

# Optional: short labels for the inset bar plot's x-axis ticks. Falls back to
# CONTROL_NAME / the group name above when left blank.
CONTROL_INSET_LABEL = "Ctrl\nn=1"
GROUP_INSET_LABELS: dict[str, str] = {
        "Sero-AAV + 1mM Sero": "Sero\nn=3",
}


def select_group_folder(root: tk.Tk, title: str) -> Path | None:
    """Show a folder-picker dialog and return the chosen path, or None if cancelled."""
    folder = filedialog.askdirectory(title=title, message=title, parent=root)
    return Path(folder) if folder else None


def find_psc_files(folder: str | Path) -> list[Path]:
    """Recursively find subject timeseries files ("*PSC_time_series*") in a folder.

    Skips macOS AppleDouble sidecar files (names starting with "._") that can
    appear on SMB/exFAT network shares and are not valid text data.
    """
    return sorted(
        p for p in Path(folder).rglob("*PSC_time_series*")
        if p.is_file() and not p.name.startswith("._")
    )


def load_group(paths: list[Path]) -> np.ndarray:
    """Load each subject's timeseries .txt file into a (subjects, timepoints) array.

    Files are trimmed to the shortest file's length if they differ.
    """
    series = [np.loadtxt(path) for path in paths]
    min_len = min(len(s) for s in series)
    if any(len(s) != min_len for s in series):
        print(f"[WARN] Trimming series to shared length {min_len} timepoints.")
    return np.array([s[:min_len] for s in series])


def semi_std(data: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric (semi) standard deviation: upper_std from above-mean samples,
    lower_std from below-mean samples."""
    mean = np.mean(data, axis=axis, keepdims=True)
    diff = data - mean
    upper_sq = np.where(diff > 0, diff ** 2, np.nan)
    lower_sq = np.where(diff < 0, diff ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        upper_std = np.sqrt(np.nanmean(upper_sq, axis=axis))
        lower_std = np.sqrt(np.nanmean(lower_sq, axis=axis))
    return np.nan_to_num(upper_std, nan=0.0), np.nan_to_num(lower_std, nan=0.0)


def p_to_stars(p: float) -> str:
    """Convert a p-value to a conventional significance annotation (***/**/*/n.s.)."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def signal_changes(data: np.ndarray, baseline: tuple[int, int], signal: tuple[int, int]) -> np.ndarray:
    """Per-subject (signal window mean - baseline window mean), used for stats/inset plot."""
    baseline_mean = np.mean(data[:, baseline[0]:baseline[1]], axis=1)
    signal_mean = np.mean(data[:, signal[0]:signal[1]], axis=1)
    return signal_mean - baseline_mean


def plot_group_vs_control(
    name: str,
    group_data: np.ndarray,
    control_data: np.ndarray,
    control_name: str,
    group_inset_label: str,
    control_inset_label: str,
    baseline: tuple[int, int],
    injection: tuple[int, int],
    signal: tuple[int, int],
    smooth: int,
    show_diff: bool,
    save_folder: Path,
    root: tk.Tk,
) -> None:
    """Plot one experimental group's mean timeseries against the control's.

    Draws the group and control mean +/- semi-std bands (optionally with the
    group-minus-control difference curve), shades the baseline/injection/
    signal windows, and adds an inset bar plot of subject-wise signal changes
    with a Welch's t-test and significance stars. After the figure is shown,
    prompts whether to save it (300 dpi PNG) into `save_folder`.
    """
    length = min(group_data.shape[1], control_data.shape[1])
    group_data = group_data[:, :length]
    control_data = control_data[:, :length]

    mean_group = np.mean(group_data, axis=0)
    std_upper_group, std_lower_group = semi_std(group_data, axis=0)
    mean_control = np.mean(control_data, axis=0)
    std_upper_control, std_lower_control = semi_std(control_data, axis=0)

    if smooth > 1:
        mean_group, std_upper_group, std_lower_group, mean_control, std_upper_control, std_lower_control = (
            uniform_filter1d(series, size=smooth, mode="nearest")
            for series in (mean_group, std_upper_group, std_lower_group, mean_control, std_upper_control, std_lower_control)
        )

    group_change = signal_changes(group_data, baseline, signal)
    control_change = signal_changes(control_data, baseline, signal)
    t_stat, p_value = stats.ttest_ind(group_change, control_change, equal_var=False)
    print(f"[{name} vs {control_name}] t = {t_stat:.3f}, p = {p_value:.4g}")

    means = [np.mean(group_change), np.mean(control_change)]
    sems = [
        np.std(group_change, ddof=1) / np.sqrt(len(group_change)),
        np.std(control_change, ddof=1) / np.sqrt(len(control_change)),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    time = np.arange(length) / 60  # frames are 1 second each
    ax.plot(time, mean_group, label=name, color="C0", linewidth=2, zorder=10)
    ax.plot(time, mean_control, label=control_name, color="C1", linewidth=2, zorder=10)
    ax.fill_between(time, mean_group - std_lower_group, mean_group + std_upper_group, color="C0", alpha=0.2, zorder=9)
    ax.fill_between(time, mean_control - std_lower_control, mean_control + std_upper_control, color="C1", alpha=0.2, zorder=8)
    if show_diff:
        ax.plot(time, mean_group - mean_control, label=f"{name} - {control_name}", color="C2", linewidth=2, zorder=10)
    ax.axvspan(baseline[0] / 60, baseline[1] / 60, color="#9e9e9e", alpha=0.18, zorder=1)
    ax.axvspan(injection[0] / 60, injection[1] / 60, color="#e07a5f", alpha=0.18, zorder=1)
    ax.axvspan(signal[0] / 60, signal[1] / 60, color="#3d5a80", alpha=0.18, zorder=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("PSC [%]")
    ax.legend(loc="upper center")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for (start, end), text in ((baseline, "Baseline"), (injection, "Injection"), (signal, "Signal")):
        ax.text((start / 60 + end / 60) / 2, 0.02, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)

    x_inset = np.arange(2)
    ax_inset = ax.inset_axes([0.075, 0.7, 0.125, 0.25])
    ax_inset.set_facecolor("#f5f5f5")
    ax_inset.bar(x_inset, means, yerr=sems, capsize=4, color=["C0", "C1"], edgecolor="black", linewidth=1, alpha=0.85, zorder=2)
    for i, changes in enumerate((group_change, control_change)):
        ax_inset.scatter(np.full(len(changes), x_inset[i]), changes, color="black", s=18, zorder=3)

    stars = p_to_stars(p_value)
    all_values = np.concatenate([group_change, control_change])
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = (y_max - y_min) or 1
    y_sig = y_max + 0.20 * y_range
    h = 0.08 * y_range
    ax_inset.set_ylim(y_min - 0.15 * y_range, y_sig + h + 0.25 * y_range)
    ax_inset.plot([x_inset[0], x_inset[0], x_inset[1], x_inset[1]], [y_sig, y_sig + h, y_sig + h, y_sig], color="black", linewidth=1, zorder=10)
    ax_inset.text(np.mean(x_inset), y_sig + h, stars, ha="center", va="bottom", fontsize=10, zorder=10)
    ax_inset.axhline(0, color="black", linewidth=0.8)
    ax_inset.set_xticks(x_inset)
    ax_inset.set_xticklabels([group_inset_label, control_inset_label], fontsize=8)
    ax_inset.set_ylabel("ΔPSC [%]", fontsize=8)
    ax_inset.tick_params(axis="both", labelsize=8)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    if messagebox.askyesno("Save plot", f"Save the '{name}' plot to {save_folder}?", parent=root):
        suffix = "group_analysis" if show_diff else "group_analysis_no_diff"
        target = save_folder / f"{name}_vs_{control_name}_{suffix}.png"
        fig.savefig(target, dpi=300)
        print(f"[OK] Saved {target}")


def run() -> None:
    """Load the control and experimental groups (from constants or dialogs) and
    plot each experimental group against the control."""
    root = tk.Tk()
    # Setting -topmost before withdraw avoids macOS briefly showing an empty root window.
    root.attributes("-topmost", True)
    root.withdraw()

    if CONTROL_FOLDER:
        control_paths = find_psc_files(CONTROL_FOLDER)
    else:
        control_folder = select_group_folder(root, "Select control group folder")
        control_paths = find_psc_files(control_folder) if control_folder else []
    if not control_paths:
        messagebox.showerror("Ctrl analysis", "No control files found.", parent=root)
        root.destroy()
        return
    control_data = load_group(control_paths)

    groups: dict[str, tuple[np.ndarray, Path]] = {}
    if GROUP_FOLDERS:
        for name, folder in GROUP_FOLDERS.items():
            paths = find_psc_files(folder)
            if not paths:
                print(f"[WARN] No PSC_time_series files found in {folder} for group '{name}'.")
                continue
            groups[name] = (load_group(paths), Path(folder))
    else:
        while True:
            name = simpledialog.askstring("Experimental group", "Name for this group (Cancel to finish)", parent=root)
            if not name:
                break
            folder = select_group_folder(root, f"Select folder for '{name}'")
            if not folder:
                continue
            paths = find_psc_files(folder)
            if not paths:
                print(f"[WARN] No PSC_time_series files found in {folder} for group '{name}'.")
                continue
            groups[name] = (load_group(paths), folder)

    if not groups:
        root.destroy()
        print("[WARN] No experimental groups selected; nothing to plot.")
        return

    show_diff = messagebox.askyesno(
        "Subtracted timeseries",
        "Also plot the subtracted (group - control) timeseries?",
        parent=root,
    )

    baseline = BASELINE_WINDOW
    injection = INJECTION_WINDOW
    signal = SIGNAL_WINDOW
    smooth = SMOOTH_WINDOW

    # Detailed group-analysis plot (timecourse + signal-change stats) per group vs control.
    control_inset_label = CONTROL_INSET_LABEL or CONTROL_NAME
    for name, (data, folder) in groups.items():
        group_inset_label = GROUP_INSET_LABELS.get(name) or name
        plot_group_vs_control(
            name, data, control_data, CONTROL_NAME, group_inset_label, control_inset_label,
            baseline, injection, signal, smooth, show_diff, folder, root,
        )

    root.destroy()


if __name__ == "__main__":
    run()

