#!/usr/bin/env python3
"""BLuSH filtering, QC, SCM, and decomposition workflow.

The original shell workflow expected cleaned_mc_func.nii.gz and an optional
mask in the current directory.  This version chooses those files with native
file dialogs and performs the numerical operations in Python.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import platform
import subprocess
import webbrowser
from pathlib import Path
from typing import Callable

import matplotlib
# The macosx backend and Tkinter each run their own native Cocoa event loop;
# mixing them in one process causes intermittent segfaults, so force TkAgg.
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.signal import butter, detrend as scipy_detrend, filtfilt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog, messagebox, simpledialog


TIMESTAMP = "%Y%m%d_%H%M%S"


def stamp() -> str:
    return dt.datetime.now().strftime(TIMESTAMP)


def nifti_name(path: Path) -> str:
    return path.name.removesuffix(".nii.gz").removesuffix(".nii")


def ask_number(root: tk.Tk, title: str, prompt: str, default: float = 0) -> float | None:
    value = simpledialog.askstring(title, prompt, initialvalue=str(default), parent=root)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        messagebox.showerror(title, f"Enter a number, not {value!r}.", parent=root)
        return ask_number(root, title, prompt, default)


def ask_integer(root: tk.Tk, title: str, prompt: str, default: int = 0) -> int | None:
    value = ask_number(root, title, prompt, default)
    return None if value is None else int(value)


def choose_nifti(root: tk.Tk, title: str, candidates: list[Path] | None = None) -> Path | None:
    if candidates:
        candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
        labels = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(candidates))
        choice = ask_integer(root, title, f"Choose a file number:\n\n{labels}", 1)
        if choice is not None and 1 <= choice <= len(candidates):
            return candidates[choice - 1]
        return None
    selected = filedialog.askopenfilename(
        title=title,
        message=title,
        filetypes=[("NIfTI image (*.nii)", "*.nii"), ("Compressed NIfTI image (*.gz)", "*.gz"), ("All files", "*.*")],
        parent=root,
    )
    return Path(selected) if selected else None


def make_dirs(output: Path) -> dict[str, Path]:
    folders = {
        "root": output / "soner_pipeline",
        "filtering": output / "soner_pipeline" / "filtering",
        "scm": output / "soner_pipeline" / "scm_outputs",
        "detrended": output / "soner_pipeline" / "detrended",
        "pca": output / "soner_pipeline" / "pca_outputs",
        "global": output / "soner_pipeline" / "qc" / "global",
        "grid": output / "soner_pipeline" / "qc" / "grid",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    for category in ("raw", "hp", "lp", "bp", "notch", "detrended", "trim", "smoothed"):
        (folders["pca"] / category).mkdir(parents=True, exist_ok=True)
    return folders


def load(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    return image, image.get_fdata(dtype=np.float32)


def save_like(data: np.ndarray, image: nib.Nifti1Image, path: Path) -> Path:
    nib.save(nib.Nifti1Image(data.astype(np.float32), image.affine, image.header), path)
    print(f"[OK] Saved: {path}")
    return path


def tr_for(image: nib.Nifti1Image) -> float:
    zooms = image.header.get_zooms()
    return float(zooms[3]) if len(zooms) > 3 and zooms[3] > 0 else 1.0


def global_signal(path: Path, mask_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    image, data = load(path)
    if data.ndim == 3:
        return np.array([float(np.nanmean(data))]), np.array([0.0])
    values = data.reshape(-1, data.shape[-1])
    if mask_path:
        try:
            _, mask = load(mask_path)
            values = data[mask > 0]
        except (OSError, ValueError):
            print("[WARN] Mask could not be applied; using the whole volume.")
    return np.nanmean(values, axis=0), np.arange(data.shape[-1]) * tr_for(image)


def save_global_plot(path: Path, mask: Path | None, output: Path, label: str) -> Path:
    signal, times = global_signal(path, mask)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(times, signal, color="black", linewidth=1)
    axis.set(title=f"Global Mean - {path.name}", xlabel="Time (s)", ylabel="Mean intensity")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    target = output / "qc" / "global" / f"global_{label}_{stamp()}.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    print(f"[OK] Global plot: {target}")
    return target


def show_global(path: Path, mask: Path | None) -> None:
    signal, times = global_signal(path, mask)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(times, signal, color="black")
    axis.set(title=f"Global Mean Signal - {path.name}", xlabel="Time (s)", ylabel="Intensity")
    axis.grid(alpha=0.3)
    plt.show()


def trim_file(path: Path, start: int, end: int, folders: dict[str, Path]) -> Path:
    image, data = load(path)
    if data.ndim == 3:
        print("[WARN] 3D data cannot be trimmed.")
        return path
    stop = data.shape[-1] - end if end else data.shape[-1]
    if start < 0 or end < 0 or start >= stop:
        raise ValueError("Trim values leave no volumes in the dataset.")
    target = folders["root"] / f"trimmed_{start}-{end}_{nifti_name(path)}_{stamp()}.nii.gz"
    return save_like(data[..., start:stop], image, target)


def smooth_file(path: Path, temporal: float, spatial: float, tr: float, folders: dict[str, Path]) -> Path:
    image, data = load(path)
    result = data
    if temporal and data.ndim == 4:
        window = max(1, int(round(temporal / tr)))
        result = uniform_filter1d(result, size=window, axis=-1, mode="nearest")
        target = folders["root"] / f"temporal_smoothed_{temporal:g}s_{path.name}"
        path = save_like(result, image, target)
    elif temporal:
        print("[WARN] 3D data: temporal smoothing skipped.")
    if spatial:
        image, result = load(path)
        sigma = [spatial / zoom for zoom in image.header.get_zooms()[:3]]
        result = gaussian_filter(result, sigma=sigma + ([0] if result.ndim == 4 else []))
        target = folders["root"] / f"spatial_smoothed_{spatial:g}mm_{nifti_name(path)}_{stamp()}.nii.gz"
        path = save_like(result, image, target)
    return path


def detrend_file(path: Path, order: int, folders: dict[str, Path]) -> Path:
    image, data = load(path)
    if data.ndim != 4:
        print("[WARN] 3D data: detrending skipped.")
        return path
    result = scipy_detrend(data, axis=-1, type="linear" if order == 1 else "constant")
    if order > 1:
        time = np.linspace(-1, 1, data.shape[-1])
        design = np.stack([time**degree for degree in range(order + 1)], axis=1)
        flat = data.reshape(-1, data.shape[-1])
        coefficients = np.linalg.lstsq(design, flat.T, rcond=None)[0]
        result = (flat.T - design @ coefficients).T.reshape(data.shape)
    target = folders["detrended"] / f"detrended_{nifti_name(path)}_pol{order}_{stamp()}.nii.gz"
    return save_like(result, image, target)


def filter_file(path: Path, kind: str, values: tuple[float, ...], tr: float, folders: dict[str, Path]) -> Path | None:
    image, data = load(path)
    if data.ndim != 4:
        print("[WARN] 3D data cannot be temporally filtered.")
        return None
    nyquist = 0.5 / tr
    if kind == "High-pass":
        cutoff = values[0]
        frequencies, btype, label = cutoff / nyquist, "highpass", f"hp_{cutoff:g}Hz"
    elif kind == "Low-pass":
        cutoff = values[0]
        frequencies, btype, label = cutoff / nyquist, "lowpass", f"lp_{cutoff:g}Hz"
    elif kind == "Band-pass":
        low, high = values
        frequencies, btype, label = [low / nyquist, high / nyquist], "bandpass", f"bp_{low:g}-{high:g}Hz"
    else:
        center, width = values
        frequencies, btype, label = [(center - width / 2) / nyquist, (center + width / 2) / nyquist], "bandstop", f"notch_{center:g}Hz"
    if np.any(np.asarray(frequencies) <= 0) or np.any(np.asarray(frequencies) >= 1):
        raise ValueError("Filter frequencies must be between 0 and the Nyquist frequency.")
    b, a = butter(4, frequencies, btype=btype)
    flat = data.reshape(-1, data.shape[-1])
    good = np.isfinite(flat).all(axis=1) & (flat.std(axis=1) > 0)
    result = flat.copy()
    if good.any():
        result[good] = filtfilt(b, a, flat[good], axis=1, method="gust")
    result = result.reshape(data.shape)
    result[~np.isfinite(result)] = data[~np.isfinite(result)]
    target = folders["filtering"] / f"{label}_{nifti_name(path)}_{stamp()}.nii.gz"
    return save_like(result, image, target)


def scm(path: Path, baseline: tuple[int, int], signal: tuple[int, int], mask: Path | None, folders: dict[str, Path]) -> Path:
    image, data = load(path)
    base = np.mean(data[..., baseline[0]:baseline[1] + 1], axis=-1) if data.ndim == 4 else data
    response = np.mean(data[..., signal[0]:signal[1] + 1], axis=-1) if data.ndim == 4 else data
    change = (response - base) / np.where(base == 0, np.nan, base) * 100
    normalized = (data - base[..., None]) / np.where(base[..., None] == 0, np.nan, base[..., None]) * 100 if data.ndim == 4 else change
    if mask:
        _, mask_data = load(mask)
        change = np.where(mask_data > 0, change, 0)
        normalized = np.where(mask_data[..., None] > 0, normalized, 0) if normalized.ndim == 4 else normalized
    folder = folders["scm"] / f"{nifti_name(path)}_B{baseline[0]}-{baseline[1]}_S{signal[0]}-{signal[1]}_{stamp()}"
    folder.mkdir(parents=True, exist_ok=True)
    save_like(base, image, folder / f"baseline_{baseline[0]}_{baseline[1]}.nii.gz")
    save_like(response, image, folder / f"signal_{signal[0]}_{signal[1]}.nii.gz")
    save_like(change, image, folder / f"signal_change_map_{baseline[0]}_{baseline[1]}_{signal[0]}_{signal[1]}.nii.gz")
    save_like(normalized, image, folder / f"norm_func_{baseline[0]}_{baseline[1]}.nii.gz")
    return folder


def component_gui(components: np.ndarray, title: str, target: Path) -> list[int]:
    shown = min(components.shape[1], 25)
    figure, axes = plt.subplots(int(np.ceil(shown / 5)), 5, figsize=(18, 9), squeeze=False)
    axes = axes.ravel()
    dropped: set[int] = set()
    figure.suptitle(f"{title} | click=drop | CTRL+click=marker | ENTER=save")
    for index in range(shown):
        signal = components[:, index]
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        axes[index].plot(signal, linewidth=0.8)
        axes[index].set_title(f"C{index + 1}")
    for axis in axes[shown:]:
        axis.set_visible(False)

    def click(event):
        if event.inaxes not in axes[:shown]:
            return
        index = int(np.where(axes == event.inaxes)[0][0])
        if event.key in ("control", "ctrl") and event.xdata is not None:
            time = int(round(event.xdata))
            if 0 <= time < components.shape[0]:
                print(f"[MARK] C{index + 1}: t={time}, value={components[time, index]:.4f}")
            return
        if index in dropped:
            dropped.remove(index)
        else:
            dropped.add(index)
        event.inaxes.spines["top"].set_color("red" if index in dropped else "black")
        event.inaxes.set_title(f"C{index + 1}" + (" x" if index in dropped else ""))
        figure.canvas.draw_idle()

    def key(event):
        if event.key == "enter":
            figure.savefig(target, dpi=160)
            print(f"[OK] Component grid: {target}")
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", click)
    figure.canvas.mpl_connect("key_press_event", key)
    plt.show()
    return sorted(dropped)


def decompose(path: Path, method: str, category: str, regressor: tuple[int, int], folders: dict[str, Path], mask: Path | None) -> Path | None:
    image, data = load(path)
    if data.ndim != 4:
        print("[WARN] Decomposition requires 4D data.")
        return None
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    if method == "PCA":
        model = PCA(n_components=min(50, shape[-1]))
        components = model.fit_transform(flat.T)
        reconstruct = lambda selected: model.inverse_transform(selected).T
    elif method == "Temporal ICA":
        model = FastICA(n_components=min(25, shape[-1]), random_state=0, max_iter=500)
        components = model.fit_transform(flat.T)
        reconstruct = lambda selected: (selected @ model.mixing_.T).T
    elif method == "Spatial ICA":
        centered = flat - flat.mean(axis=1, keepdims=True)
        transformed = StandardScaler(with_mean=False).fit_transform(centered.T)
        model = FastICA(n_components=min(25, shape[-1]), random_state=0, max_iter=500, whiten="unit-variance")
        components = model.fit_transform(transformed)
        reconstruct = lambda selected: (selected @ model.mixing_.T).T
    else:
        target = np.zeros((shape[-1], 1))
        start, end = regressor
        if 0 <= start <= end < shape[-1]:
            target[start:end + 1] = 1
        model = PLSRegression(n_components=min(25, shape[-1]))
        model.fit(flat.T, target)
        components = model.x_scores_
        reconstruct = lambda selected: (selected @ model.x_loadings_.T).T
    grid = folders["grid"] / f"{method.lower().replace(' ', '_')}_grid_{stamp()}.png"
    dropped = component_gui(components, f"{method} Components", grid)
    components[:, dropped] = 0
    result = reconstruct(components).reshape(shape)
    target = folders["pca"] / category / f"{category}_{method.lower().replace(' ', '_')}_func_denoised_{stamp()}.nii.gz"
    output = save_like(result, image, target)
    save_global_plot(output, mask, folders["root"], f"{method.lower()}_after")
    return output


def candidates(output: Path) -> list[Path]:
    patterns = ["*.nii", "*.nii.gz"]
    excluded = {"soner_pipeline"}
    return [p for pattern in patterns for p in output.glob(pattern) if p.name not in excluded]


def make_summary(folders: dict[str, Path]) -> Path:
    global_plots = sorted(folders["global"].glob("*.png"))
    grids = sorted(folders["grid"].glob("*.png"))
    def section(title: str, files: list[Path]) -> str:
        if not files:
            return f"<h2>{title}</h2><p>No images found.</p>"
        rows = "".join(f"<tr><td>{html.escape(p.name)}</td><td><img src='{html.escape(str(p))}' width='360'></td></tr>" for p in files)
        return f"<h2>{title}</h2><table>{rows}</table>"
    target = folders["root"] / f"qc_summary_{stamp()}.html"
    document = f"<html><head><meta charset='utf-8'><title>BLuSH QC Summary</title><style>body{{background:#0f1116;color:#eaeef2;font-family:Arial;padding:24px}}h1{{color:#8bd3ff}}h2{{color:#62b0ff}}td{{border:1px solid #2a2f3a;padding:6px}}table{{border-collapse:collapse}}</style></head><body><h1>BLuSH QC Summary</h1><p>Generated: {dt.datetime.now():%Y-%m-%d %H:%M}</p>{section('Global Mean Plots', global_plots)}{section('Component Grid Plots', grids)}</body></html>"
    target.write_text(document, encoding="utf-8")
    print(f"[OK] Summary HTML: {target}")
    webbrowser.open(target.as_uri())
    return target


def run() -> None:
    root = tk.Tk()
    root.withdraw()
    # Without this, dialogs can open behind other apps on macOS, making it look
    # like the same prompt keeps reappearing when a hidden one is still waiting.
    root.attributes("-topmost", True)
    root.title("BLuSH v24")
    try:
        functional = choose_nifti(root, "Choose cleaned functional NIfTI")
        if not functional:
            return
        mask_name = filedialog.askopenfilename(
            title="Choose a mask (Cancel for no mask)",
            message="Choose a mask (Cancel for no mask)",
            filetypes=[("NIfTI image (*.nii)", "*.nii"), ("Compressed NIfTI image (*.gz)", "*.gz"), ("All files", "*.*")],
            parent=root,
        )
        mask = Path(mask_name) if mask_name else None
        output_name = filedialog.askdirectory(
            title="Choose output folder",
            message="Choose output folder",
            initialdir=str(functional.parent),
            parent=root,
        )
        if not output_name:
            return
        folders = make_dirs(Path(output_name))
        image, data = load(functional)
        tr = tr_for(image)
        print(f"[INFO] Input: {functional}\n[INFO] Shape: {data.shape}\n[INFO] TR: {tr:g} seconds")
        save_global_plot(functional, mask, folders["root"], "input")
        current = functional

        start = ask_integer(root, "Trimming", "Volumes to trim from the start", 0)
        end = ask_integer(root, "Trimming", "Volumes to trim from the end", 0)
        if start is None or end is None:
            return
        if start or end:
            current = trim_file(current, start, end, folders)
            save_global_plot(current, mask, folders["root"], "trim")

        if messagebox.askyesno("Smoothing", "Apply temporal or spatial smoothing?", parent=root):
            temporal = ask_number(root, "Smoothing", "Temporal window in seconds (0 to skip)", 0)
            spatial = ask_number(root, "Smoothing", "Spatial sigma in mm (0 to skip)", 0)
            if temporal is not None and spatial is not None:
                current = smooth_file(current, temporal, spatial, tr, folders)
                save_global_plot(current, mask, folders["root"], "smooth")

        if messagebox.askyesno("Detrending", "Apply polynomial detrending?", parent=root):
            order = ask_integer(root, "Detrending", "Polynomial order (1 = linear)", 1)
            if order is not None:
                show_global(current, mask)
                current = detrend_file(current, order, folders)
                save_global_plot(current, mask, folders["root"], f"detrended_pol{order}")

        if messagebox.askyesno("SCM QC", "Run baseline/signal percent-change QC?", parent=root):
            baseline_start = ask_integer(root, "SCM QC", "Baseline start volume", 0)
            baseline_end = ask_integer(root, "SCM QC", "Baseline end volume", 0)
            signal_start = ask_integer(root, "SCM QC", "Signal start volume", 1)
            signal_end = ask_integer(root, "SCM QC", "Signal end volume", 1)
            if None not in (baseline_start, baseline_end, signal_start, signal_end):
                scm(current, (baseline_start, baseline_end), (signal_start, signal_end), mask, folders)

        if messagebox.askyesno("Filtering", "Apply a temporal filter?", parent=root):
            kind = simpledialog.askstring("Filtering", "Type: High-pass, Low-pass, Band-pass, or Notch", initialvalue="High-pass", parent=root)
            if kind:
                kind = kind.strip().title()
                if kind in ("High-Pass", "Low-Pass"):
                    value = ask_number(root, "Filtering", "Cutoff frequency in Hz", 0.01)
                    values = (value,) if value is not None else ()
                elif kind == "Band-Pass":
                    low = ask_number(root, "Filtering", "Low cutoff in Hz", 0.01)
                    high = ask_number(root, "Filtering", "High cutoff in Hz", 0.1)
                    values = (low, high) if low is not None and high is not None else ()
                elif kind == "Notch":
                    center = ask_number(root, "Filtering", "Notch center in Hz", 0.05)
                    width = ask_number(root, "Filtering", "Notch width in Hz", 0.01)
                    values = (center, width) if center is not None and width is not None else ()
                else:
                    values = ()
                if values:
                    filtered = filter_file(current, kind, values, tr, folders)
                    if filtered:
                        current = filtered
                        save_global_plot(current, mask, folders["root"], "filter")
                        show_global(current, mask)

        while messagebox.askyesno("Decomposition", "Run PCA, ICA, or PLS decomposition?", parent=root):
            method = simpledialog.askstring("Decomposition", "Method: PCA, Temporal ICA, Spatial ICA, or PLS", initialvalue="PCA", parent=root)
            if not method:
                break
            method = method.strip().title()
            if method not in ("Pca", "Temporal Ica", "Spatial Ica", "Pls"):
                messagebox.showerror("Decomposition", "Unknown decomposition method.", parent=root)
                continue
            method = {"Pca": "PCA", "Temporal Ica": "Temporal ICA", "Spatial Ica": "Spatial ICA", "Pls": "PLS"}[method]
            regressor = (0, 0)
            if method == "PLS":
                start_reg = ask_integer(root, "PLS", "Regressor start volume", 0)
                end_reg = ask_integer(root, "PLS", "Regressor end volume", 0)
                if start_reg is None or end_reg is None:
                    continue
                regressor = (start_reg, end_reg)
            category = "raw"
            name = current.name.lower()
            for token, selected in (("hp_", "hp"), ("lp_", "lp"), ("bp_", "bp"), ("notch_", "notch"), ("detrended", "detrended"), ("trimmed", "trim"), ("smoothed", "smoothed")):
                if token in name:
                    category = selected
            save_global_plot(current, mask, folders["root"], f"{method.lower()}_before")
            output = decompose(current, method, category, regressor, folders, mask)
            if output and messagebox.askyesno("SCM QC", "Run SCM QC on the denoised output?", parent=root):
                b_start = ask_integer(root, "SCM QC", "Baseline start volume", 0)
                b_end = ask_integer(root, "SCM QC", "Baseline end volume", 0)
                s_start = ask_integer(root, "SCM QC", "Signal start volume", 1)
                s_end = ask_integer(root, "SCM QC", "Signal end volume", 1)
                if None not in (b_start, b_end, s_start, s_end):
                    scm(output, (b_start, b_end), (s_start, s_end), mask, folders)
        make_summary(folders)
        messagebox.showinfo("BLuSH complete", f"Pipeline finished.\n\nOutput: {folders['root']}", parent=root)
    except Exception as error:
        messagebox.showerror("BLuSH error", str(error), parent=root)
        raise
    finally:
        root.destroy()


if __name__ == "__main__":
    run()