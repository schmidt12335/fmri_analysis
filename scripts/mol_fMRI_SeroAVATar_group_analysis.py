#!/usr/bin/env python3
"""Group-vs-control timeseries analysis (BLuSH ctrl_analysis).

What it does
------------
1. Loads per-subject percent-signal-change (PSC) timeseries .txt files for a
   control group and one or more experimental groups (any file with
   "PSC_time_series" in its name inside a folder is picked up automatically).
2. Averages subjects within each group and computes the standard error of
   the mean (SEM) band around each group mean.
3. For every experimental group, plots its mean timeseries against the
   control's mean timeseries (optionally including the group-minus-control
   difference curve) and shades the baseline/injection/signal windows, and
   as a separate figure, a bar plot of subject-wise signal changes with a
   Welch's t-test and significance stars.
4. After each pair of plots is shown, asks whether to save both (300 dpi PNG)
   into that group's own data folder.

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
  for the bar plot's x-axis ticks (falls back to the names above).

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
import matplotlib.transforms as mtransforms
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
SMOOTH_WINDOW = 120

# Optional: point these at folders to skip the file-picker dialogs. Every file
# with "PSC_time_series" in its name inside a folder is picked up automatically,
# so the number of subject files per group/control is not fixed.
# Leave CONTROL_FOLDER empty and/or GROUP_FOLDERS empty ({}) to be prompted instead.
CONTROL_NAME = "Ctrl-AAV + 150uM Sero"
CONTROL_FOLDER: str = "/Volumes/pschmidt/fmri_analysis/AnalysedData/Sero/group_data/timeseries/Ctrl_150uM"
EXP_NAME = "Sero-AAV + 150uM Sero"
GROUP_FOLDERS: dict[str, str] = {
        "Sero-AAV + 150uM Sero": "/Volumes/pschmidt/fmri_analysis/AnalysedData/Sero/group_data/timeseries/150uM",
}

# Optional: short name prefixes for the bar plot's x-axis ticks (falls back
# to CONTROL_NAME / the group name above when left blank). The "n=..." subject
# count is always appended automatically from the number of files actually loaded.
CONTROL_INSET_LABEL = "Ctrl-AAV + Sero"
GROUP_INSET_LABELS: dict[str, str] = {
        "Sero-AAV + 150uM Sero": "Sero-AVATar + Sero",
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


def sem(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standard error of the mean along `axis`."""
    return np.std(data, axis=axis, ddof=1) / np.sqrt(data.shape[axis])


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
    """Per-subject (signal window mean - baseline window mean), used for stats/bar plot."""
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

    Draws the group and control mean +/- SEM bands (optionally with the
    group-minus-control difference curve) and shades the baseline/injection/
    signal windows in one figure, and a separate bar plot of subject-wise
    signal changes with a Welch's t-test and significance stars in another.
    After both figures are shown, prompts whether to save them (300 dpi PNG)
    into `save_folder`.
    """
    length = min(group_data.shape[1], control_data.shape[1])
    group_data = group_data[:, :length]
    control_data = control_data[:, :length]

    mean_group = np.mean(group_data, axis=0)
    sem_group = sem(group_data, axis=0)
    mean_control = np.mean(control_data, axis=0)
    sem_control = sem(control_data, axis=0)

    if smooth > 1:
        mean_group, sem_group, mean_control, sem_control = (
            uniform_filter1d(series, size=smooth, mode="nearest")
            for series in (mean_group, sem_group, mean_control, sem_control)
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

    # Short names (drop the "\nn=..." suffix) keep the legend compact.
    group_label = group_inset_label.split("\n")[0]
    control_label = control_inset_label.split("\n")[0]
    group_color, control_color, diff_color = "tab:red", "tab:green", "tab:blue"

    fig, ax = plt.subplots(figsize=(10, 6))
    time = np.arange(length) / 60  # frames are 1 second each
    ax.plot(time, mean_group, label=group_label, color=group_color, linewidth=2, zorder=10)
    ax.plot(time, mean_control, label=control_label, color=control_color, linewidth=2, zorder=10)
    ax.fill_between(time, mean_group - sem_group, mean_group + sem_group, color=group_color, alpha=0.2, zorder=9)
    ax.fill_between(time, mean_control - sem_control, mean_control + sem_control, color=control_color, alpha=0.2, zorder=8)
    if show_diff:
        ax.plot(time, mean_group - mean_control, label="Difference", color=diff_color, linewidth=2, zorder=10)
    ax.axvspan(injection[0] / 60, injection[1] / 60, color="#e07a5f", alpha=0.18, zorder=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time [min]", fontweight="bold")
    ax.set_ylabel("Signal Change [%]", fontweight="bold")
    # Anchor the legend at x=15 min (data coords) but keep y near the top of the axes.
    legend_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.legend(loc="upper left", ncol=1)
    legend = ax.legend(loc="upper left", ncol=1)
    plt.setp(legend.get_texts(), fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontweight="bold")

    ax.text(np.mean(injection) / 60, 0.01, "Injection", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()

    # Signal-change bar plot, as its own standalone figure styled to match the timecourse plot.
    x_bar = np.arange(2)
    bar_width = 0.6
    rng = np.random.default_rng(0)
    fig_bar, ax_bar = plt.subplots(figsize=(4, 5))
    ax_bar.bar(
        x_bar, means, yerr=sems, width=bar_width, capsize=5, error_kw={"elinewidth": 1.2, "ecolor": "black"},
        color=[group_color, control_color], edgecolor=["darkred", "darkgreen"], linewidth=1.2, alpha=0.85, zorder=2,
    )
    for i, changes in enumerate((group_change, control_change)):
        jitter = rng.uniform(-0.12, 0.12, size=len(changes))
        ax_bar.scatter(x_bar[i] + jitter, changes, color="black", edgecolor="white", linewidth=0.5, s=32, alpha=0.8, zorder=3)

    stars = p_to_stars(p_value)
    all_values = np.concatenate([group_change, control_change])
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = (y_max - y_min) or 1
    y_sig = y_max + 0.20 * y_range
    h = 0.08 * y_range
    ax_bar.set_ylim(y_min - 0.15 * y_range, y_sig + h + 0.25 * y_range)
    ax_bar.set_xlim(x_bar[0] - 0.6, x_bar[1] + 0.6)
    ax_bar.plot([x_bar[0], x_bar[0], x_bar[1], x_bar[1]], [y_sig, y_sig + h, y_sig + h, y_sig], color="black", linewidth=2.2, zorder=10)
    ax_bar.text(np.mean(x_bar), y_sig + h, stars, ha="center", va="bottom", fontsize=16, fontweight="bold", zorder=10)
    ax_bar.axhline(0, color="black", linewidth=0.8)
    n_labels = [label.split("\n")[-1] for label in (group_inset_label, control_inset_label)]
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(n_labels)
    ax_bar.set_ylabel("Signal Change [%]", fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    plt.setp(ax_bar.get_xticklabels() + ax_bar.get_yticklabels(), fontweight="bold")
    fig_bar.tight_layout()

    plt.show()

    if messagebox.askyesno("Save plot", f"Save the '{name}' plots to {save_folder}?", parent=root):
        suffix = "group_analysis" if show_diff else "group_analysis_no_diff"
        target = save_folder / f"{name}_vs_{control_name}_{suffix}.png"
        fig.savefig(target, dpi=300)
        print(f"[OK] Saved {target}")
        bar_target = save_folder / f"{name}_vs_{control_name}_signal_change_bar.png"
        fig_bar.savefig(bar_target, dpi=300)
        print(f"[OK] Saved {bar_target}")


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
    control_inset_label = f"{CONTROL_INSET_LABEL or CONTROL_NAME}\nn={control_data.shape[0]}"
    for name, (data, folder) in groups.items():
        group_inset_label = f"{GROUP_INSET_LABELS.get(name) or name}\nn={data.shape[0]}"
        plot_group_vs_control(
            name, data, control_data, CONTROL_NAME, group_inset_label, control_inset_label,
            baseline, injection, signal, smooth, show_diff, folder, root,
        )

    root.destroy()


if __name__ == "__main__":
    run()

