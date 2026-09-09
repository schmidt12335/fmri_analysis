from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("/home/pschmidt/fmri_analysis/AnalysedData/pacap")
OUTPUT_DIR = Path("/home/pschmidt/fmri_analysis/AnalysedData/pacap/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_series(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 2:
        if data.shape[1] >= 2:
            data = data[:, 1]
        else:
            data = data[:, 0]
    return data.astype(float)


def is_time_series_file(path):
    try:
        sample = np.loadtxt(path, delimiter=",", skiprows=1, max_rows=3)
    except Exception:
        return False

    if sample.ndim == 0:
        return False
    if sample.ndim == 1:
        return sample.size > 0
    return sample.shape[1] >= 2 or sample.shape[0] > 0


def smooth_series(values, smooth=10):
    if smooth <= 1:
        return values
    return np.convolve(values, np.ones(smooth) / smooth, mode="valid")


def find_roi_series_files(subject_dir):
    candidates = [
        path
        for path in subject_dir.rglob("*.txt")
        if path.is_file() and is_time_series_file(path) and "header" not in path.name.lower()
    ]

    direct_candidates = [path for path in candidates if path.parent == subject_dir]
    if direct_candidates:
        candidates = direct_candidates

    preferred_candidates = [
        path for path in candidates if "psc_time_series_roi" in path.name.lower() or "time_series_roi" in path.name.lower()
    ]
    if preferred_candidates:
        candidates = preferred_candidates

    return sorted(candidates)


def plot_subject(subject_dir):
    roi_paths = find_roi_series_files(subject_dir)

    if not roi_paths:
        print(f"Skipping {subject_dir.name}: missing time-series files")
        return

    pacap_path = next((path for path in roi_paths if "pacap" in path.name.lower()), None)
    other_paths = [path for path in roi_paths if path != pacap_path and "pacap" not in path.name.lower()]
    pbs_path = other_paths[0] if other_paths else None

    series = []
    if pacap_path is not None:
        series.append((pacap_path.name, load_series(pacap_path)))
    if pbs_path is not None:
        series.append((pbs_path.name, load_series(pbs_path)))

    if not series:
        print(f"Skipping {subject_dir.name}: no usable series")
        return

    smoothed_series = [(name, smooth_series(values, smooth=1)) for name, values in series]
    time = np.arange(len(smoothed_series[0][1]))

    if len(time) > 500:
        dur = 200
        base_start = 300
        base_end = base_start + dur
        sig_start = 2000
        sig_end = sig_start + dur
        inj_dur = 600
        inj = 600
    else:
        dur = 34
        base_start = 26
        base_end = base_start + dur
        sig_start = 250
        sig_end = sig_start + dur
        inj_dur = 75
        inj = 75

    means = []
    sems = []
    labels = []
    for name, values in smoothed_series:
        mean_value = np.mean(values[sig_start:sig_end]) - np.mean(values[base_start:base_end])
        sem_value = np.std(values[base_start:base_end]) / np.sqrt(len(values[base_start:base_end]))
        means.append(mean_value)
        sems.append(sem_value)
        labels.append("PACAP" if "pacap" in name.lower() else "PBS")

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (_, values) in zip(labels, smoothed_series):
        ax.plot(time, values, label=label)

    if means:
        x_inset = np.arange(len(means))
        ax_inset = ax.inset_axes([0.04, 0.7, 0.125, 0.25])
        ax_inset.bar(
            x_inset,
            means,
            yerr=sems,
            capsize=4,
            color=["C0", "C1"][: len(means)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.75,
            zorder=2,
        )

    ax.axvspan(base_start, base_end, color="green", alpha=0.3)
    ax.axvspan(sig_start, sig_end, color="blue", alpha=0.3)
    ax.axvspan(inj, inj + inj_dur, color="red", alpha=0.3)
    ax.set_title(f"{subject_dir.name} - Time Series")
    #ax.set_ylim(-2, 5)
    ax.set_xlabel("Time Points (Volumes)")
    ax.set_ylabel("Signal Change (%)")
    ax.legend(loc="upper center")

    output_path = OUTPUT_DIR / f"{subject_dir.name}_timeseries.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


subject_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
for subject_dir in subject_dirs:
    plot_subject(subject_dir)
