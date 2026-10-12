"""Signal change maps on an atlas as a stack of thick, spaced slices seen at an angle (common figure in fMRI papers).

Every slab is `thickness` mm of the atlas template with `gap` mm of empty space to the next one. The camera is rotated
by `azimuth` (to the side) and `elevation` (from above), so the faces, the tops and the sides of the slabs are visible.
The map is drawn on the face of each slab (mean or maximum over the slab thickness); the sides of a slab are rendered
from the actual sub-slices (darkened), so the thickness follows the outline of the brain.

Run without arguments (`python plot_slice_stack.py`) to get a form that asks for the files and all settings.

Command line (maps from the atlas_space folder of the electric stimulation notebooks, same grid as the template):
    python plot_slice_stack.py atlas_space/stim_diff_psc_atlasspace.nii.gz template_registration_0.1mm.nii.gz stack.png \
        --threshold 0.5 --vmax 3 --label "Signal change (%)"
In a notebook:
    from plot_slice_stack import plot_slice_stack
    plot_slice_stack(map_file, template_file, "stack.png", threshold=2, vmax=6, label="t")
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Affine2D
from nibabel.processing import resample_from_to
from scipy import ndimage

# Per slice direction: (index of the slice axis, viewing direction, "up" direction on screen) in RAS world axes
# (x = right, y = anterior, z = superior). Coronal is seen from posterior, sagittal from the left, axial from above.
AXES = {
    "coronal": (1, np.array([0, 1, 0]), np.array([0, 0, 1])),
    "sagittal": (0, np.array([1, 0, 0]), np.array([0, 0, 1])),
    "axial": (2, np.array([0, 0, -1]), np.array([0, 1, 0])),
}
SIDE_GRAY = (0.28, 0.5)  # brightness of the slab sides at the back and next to the face
EDGE_COLOR = np.array([0.12, 0.12, 0.12])
SETTINGS_FILE = Path(__file__).resolve().parent / "plot_slice_stack_last.json"
NEG_CMAP = LinearSegmentedColormap.from_list("neg", ["#00ffff", "#0030ff"])


def _load_canonical(path_or_img, reference=None):
    img = nib.as_closest_canonical(nib.load(str(path_or_img)) if not isinstance(path_or_img, nib.Nifti1Image) else path_or_img)
    if reference is not None and (img.shape[:3] != reference.shape[:3] or not np.allclose(img.affine, reference.affine, atol=1e-3)):
        img = resample_from_to(img, (reference.shape[:3], reference.affine), order=1)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    return img, (data[..., 0] if data.ndim == 4 else data)


def _fill_holes(mask, slice_axis):
    out = np.zeros_like(mask)
    for i in range(mask.shape[slice_axis]):
        idx = [slice(None)] * 3
        idx[slice_axis] = i
        out[tuple(idx)] = ndimage.binary_fill_holes(mask[tuple(idx)])
    return out


def _overlay_colors(values, cmap_pos, cmap_neg, threshold, vmax):
    """RGBA for the overlay values; voxels with |value| <= threshold are transparent."""
    span = max(vmax - threshold, 1e-9)
    frac = np.clip((np.abs(values) - threshold) / span, 0, 1)
    rgba = np.where((values >= 0)[..., None], cmap_pos(frac), cmap_neg(frac))
    rgba[..., 3] = (np.abs(values) > threshold).astype(np.float32)
    return rgba


def _face_rgba(under, over, mask, cmaps, norm_under, threshold, vmax, alpha, shade, edge):
    gray = np.clip((under - norm_under[0]) / max(norm_under[1] - norm_under[0], 1e-9), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    ov = _overlay_colors(over, *cmaps, threshold, vmax)
    a = (ov[..., 3] * alpha)[..., None]
    rgb = rgb * (1 - a) + ov[..., :3] * a
    rgb = rgb * shade
    if edge:
        border = mask & ~ndimage.binary_erosion(mask, iterations=2)
        rgb[border] = EDGE_COLOR * (0.5 + 0.5 * shade)
    return np.dstack([rgb, mask.astype(np.float32)])


def _upsample(arr, factor, binary=False):
    if factor == 1:
        return arr
    out = ndimage.zoom(arr.astype(np.float32), factor, order=1)
    return out > 0.5 if binary else out


def plot_slice_stack(map_file, underlay_file, out_file=None, *, axis="coronal", thickness=2.5, gap=2.0, azimuth=35.0,
                     elevation=25.0, slice_range=None, n_slices=None, cover="data", threshold=0.0, vmax=None,
                     cmap="autumn", alpha=1.0, projection="mean", brain_mask=None, under_threshold=0.02,
                     upsample=3, flip_view=False, slice_labels=True, label_origin=0.0, label="", title="",
                     colorbar=True, background="white", dpi=300, figsize=(9, 7), show=False):
    """Draws the stack and saves it to `out_file` (png/pdf/svg). Returns the matplotlib figure.

    thickness, gap  mm of atlas per slab / of empty space between slabs
    azimuth         degrees the camera is turned to the side (0 = looking straight onto the slices)
    elevation       degrees the camera is raised above the slices (0 = no view onto the tops)
    slice_range     (start, end) in mm along the slice axis (atlas world coordinates); default: `cover`
    cover           "data" = extent of the map (default), "brain" = whole brain
    n_slices        use only this many slabs, centred in the range
    threshold, vmax values <= threshold are not shown, vmax is the end of the colour scale (default: 99.5th percentile)
    cmap            colormap of positive values (negative values, if present, are shown in cyan-blue)
    projection      "mean" or "max" of the map over the slab thickness
    flip_view       look at the slices from the other side
    label_origin    mm that are subtracted from the slice position in the labels (e.g. the y coordinate of bregma)
    """
    slice_axis, view_dir, up = AXES[axis]
    view_dir = -view_dir if flip_view else view_dir
    right = np.cross(view_dir, up)

    under_img, under = _load_canonical(underlay_file)
    _, over = _load_canonical(map_file, reference=under_img)
    over = np.nan_to_num(over)
    if brain_mask is not None:
        _, mask3d = _load_canonical(brain_mask, reference=under_img)
        mask3d = mask3d > 0
    else:
        mask3d = _fill_holes(under > under_threshold * np.percentile(under[under > 0], 99), slice_axis)
    over = np.where(mask3d, over, 0)

    in_brain = under[mask3d]
    norm_under = (np.percentile(in_brain, 1), np.percentile(in_brain, 99.5))
    if vmax is None:
        shown = np.abs(over[np.abs(over) > threshold])
        vmax = float(np.percentile(shown, 99.5)) if shown.size else threshold + 1
    cmap_pos = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    cmaps = (cmap_pos, NEG_CMAP)

    # World coordinate of every voxel along the three axes (canonical RAS: the affine is diagonal)
    aff = under_img.affine
    world = [aff[a, a] * np.arange(under.shape[a]) + aff[a, 3] for a in range(3)]
    h_axis, v_axis = int(np.argmax(np.abs(right))), int(np.argmax(np.abs(up)))
    sign_h, sign_v, sign_s = right[h_axis], up[v_axis], view_dir[slice_axis]
    voxel_depth = abs(aff[slice_axis, slice_axis])

    # Slab positions along the slice axis
    support = mask3d if cover == "brain" else (np.abs(over) > threshold)
    others = tuple(a for a in range(3) if a != slice_axis)
    present = np.where(support.any(axis=others))[0]
    if slice_range is None:
        if present.size == 0:
            raise ValueError("Nothing to show: the map has no voxels above the threshold inside the brain.")
        lo, hi = world[slice_axis][present].min(), world[slice_axis][present].max()
    else:
        lo, hi = sorted(slice_range)
    pitch = thickness + gap
    n_fit = int(np.floor((hi - lo) / pitch)) + 1
    n = min(n_slices, n_fit) if n_slices else n_fit
    first = (lo + hi) / 2 - (n - 1) * pitch / 2
    centers = first + pitch * np.arange(n)

    az, el = np.radians(azimuth), np.radians(elevation)

    def projection_transform(depth):
        return Affine2D.from_values(np.cos(az), -np.sin(az) * np.sin(el), 0, np.cos(el),
                                    depth * np.sin(az), depth * np.cos(az) * np.sin(el))

    def plane(vol, index):
        return _plane_of_reduced(vol.take(index, axis=slice_axis), slice_axis, h_axis, v_axis, sign_h, sign_v, aff)

    h_vals = sign_h * world[h_axis]
    v_vals = sign_v * world[v_axis]
    dh, dv = abs(aff[h_axis, h_axis]), abs(aff[v_axis, v_axis])
    extent = (h_vals.min() - dh / 2, h_vals.max() + dh / 2, v_vals.min() - dv / 2, v_vals.max() + dv / 2)

    fig, ax = plt.subplots(figsize=figsize, facecolor="none" if background == "transparent" else background)
    if background != "transparent":
        ax.set_facecolor(background)
    corners = []
    slabs = []
    for c in centers:
        members = np.where(np.abs(world[slice_axis] - c) <= thickness / 2 + 1e-6)[0]
        if members.size == 0:
            continue
        slabs.append((sign_s * c, members))
    # Far slabs first, within a slab far sub-slices first
    for _, members in sorted(slabs, key=lambda s: -s[0]):
        depths = sign_s * world[slice_axis][members]
        order = np.argsort(-depths)
        members, depths = members[order], depths[order]
        stack = over.take(members, axis=slice_axis)
        slab_over = np.moveaxis(stack, slice_axis, 0)
        slab_over = slab_over.max(axis=0) if projection == "max" else slab_over.mean(axis=0)
        slab_over = _upsample(_plane_of_reduced(slab_over, slice_axis, h_axis, v_axis, sign_h, sign_v, aff), upsample)
        face_mask = _upsample(plane(mask3d, members[-1]), upsample, binary=True)
        for k, (index, depth) in enumerate(zip(members, depths)):
            if k == len(members) - 1:
                rgba = _face_rgba(_upsample(plane(under, index), upsample), slab_over, face_mask, cmaps, norm_under,
                                  threshold, vmax, alpha, 1.0, edge=True)
            else:
                # Sides: the face outline extruded as a prism, darker towards the back
                shade = SIDE_GRAY[0] + (SIDE_GRAY[1] - SIDE_GRAY[0]) * k / max(len(members) - 1, 1)
                rgba = np.dstack([np.full(face_mask.shape + (3,), shade, dtype=np.float32), face_mask.astype(np.float32)])
            transform = projection_transform(depth)
            im = ax.imshow(rgba, extent=extent, origin="lower", interpolation="bilinear", zorder=2)
            im.set_transform(transform + ax.transData)
            for hx in extent[:2]:
                for vy in extent[2:]:
                    corners.append(transform.transform((hx, vy)))
        if slice_labels:
            label_depth = depths[-1]
            x, y = projection_transform(label_depth).transform((extent[1], extent[2]))
            ax.text(x, y - 0.35, f"{world[slice_axis][members].mean() - label_origin:+.1f}", ha="right", va="top", fontsize=8)

    corners = np.array(corners)
    pad = 0.04 * np.ptp(corners, axis=0).max()
    ax.set_xlim(corners[:, 0].min() - pad, corners[:, 0].max() + pad)
    ax.set_ylim(corners[:, 1].min() - pad - (0.6 if slice_labels else 0), corners[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)

    if colorbar:
        negative = bool((over < -threshold).any())
        if negative:
            colors = np.vstack([NEG_CMAP(np.linspace(1, 0, 256)), cmap_pos(np.linspace(0, 1, 256))])
            mappable = ScalarMappable(Normalize(-vmax, vmax), ListedColormap(colors))
        else:
            mappable = ScalarMappable(Normalize(threshold, vmax), cmap_pos)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.02, aspect=18)
        cbar.set_label(label)

    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight", transparent=background == "transparent")
        print(f"[OK] Slice stack saved: {out_file}  ({len(slabs)} slabs of {thickness} mm, gap {gap} mm)")
    if show:
        plt.show()
    return fig


def _plane_of_reduced(arr2d, slice_axis, h_axis, v_axis, sign_h, sign_v, aff):
    """Orient a slab-reduced 2D array (axes in increasing RAS order without the slice axis) like `plane` does."""
    order = [a for a in range(3) if a != slice_axis]
    if order != [h_axis, v_axis]:
        arr2d = arr2d.T
    if sign_h * aff[h_axis, h_axis] < 0:
        arr2d = arr2d[::-1]
    if sign_v * aff[v_axis, v_axis] < 0:
        arr2d = arr2d[:, ::-1]
    return arr2d.T


def _guess_underlay(map_file):
    for folder in list(Path(map_file).resolve().parents)[:3]:
        found = sorted(folder.glob("template_registration_*mm.nii.gz"))
        if found:
            return str(found[0])
    return ""


def _open_file(path):
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    elif sys.platform.startswith("win"):
        os.startfile(str(path))
    else:
        subprocess.run(["xdg-open", str(path)])


# (key, label, default, kind, choices): "file" fields get a Browse button, "choice" a drop-down, "check" a checkbox
GUI_FIELDS = [
    ("map", "Map in atlas space (.nii/.nii.gz)", "", "file", None),
    ("underlay", "Atlas template (same grid as the map)", "", "file", None),
    ("out", "Output image (.png/.pdf/.svg)", "", "save", None),
    ("axis", "Slice direction", "coronal", "choice", list(AXES)),
    ("thickness", "Slab thickness (mm)", "2.5", "text", None),
    ("gap", "Gap between slabs (mm)", "2.0", "text", None),
    ("azimuth", "Camera: turn to the side (deg)", "35", "text", None),
    ("elevation", "Camera: raise above the slices (deg)", "25", "text", None),
    ("cover", "Slabs cover", "data", "choice", ["data", "brain"]),
    ("range_start", "Range start (mm, empty = automatic)", "", "text", None),
    ("range_end", "Range end (mm, empty = automatic)", "", "text", None),
    ("n_slices", "Max. number of slabs (empty = automatic)", "", "text", None),
    ("threshold", "Threshold (values below are hidden)", "0", "text", None),
    ("vmax", "Colour scale maximum (empty = automatic)", "", "text", None),
    ("cmap", "Colormap (matplotlib name)", "autumn", "text", None),
    ("projection", "Map over slab thickness", "mean", "choice", ["mean", "max"]),
    ("brain_mask", "Brain mask (optional)", "", "file", None),
    ("label", "Colorbar label", "", "text", None),
    ("title", "Title", "", "text", None),
    ("label_origin", "Slice labels: subtract (mm, e.g. bregma)", "0", "text", None),
    ("slice_labels", "Show slice position labels", True, "check", None),
    ("colorbar", "Show colorbar", True, "check", None),
    ("flip_view", "View from the other side", False, "check", None),
    ("transparent", "Transparent background", False, "check", None),
    ("dpi", "Resolution (dpi)", "300", "text", None),
]


def _gui_values_to_kwargs(v):
    def number(key, cast=float, optional=False):
        text = str(v[key]).strip()
        if not text:
            if optional:
                return None
            raise ValueError(f"'{key}' is required")
        try:
            return cast(text)
        except ValueError:
            raise ValueError(f"'{key}' must be a number, got '{text}'") from None

    for key, name in (("map", "map"), ("underlay", "atlas template"), ("out", "output file")):
        if not str(v[key]).strip():
            raise ValueError(f"Please select the {name}.")
    for key in ("map", "underlay"):
        if not Path(v[key]).is_file():
            raise ValueError(f"File not found: {v[key]}")
    if v["brain_mask"].strip() and not Path(v["brain_mask"]).is_file():
        raise ValueError(f"File not found: {v['brain_mask']}")
    start, end = number("range_start", optional=True), number("range_end", optional=True)
    if (start is None) != (end is None):
        raise ValueError("Enter both range start and range end, or leave both empty.")
    return dict(
        map_file=v["map"], underlay_file=v["underlay"], out_file=v["out"], axis=v["axis"],
        thickness=number("thickness"), gap=number("gap"), azimuth=number("azimuth"), elevation=number("elevation"),
        slice_range=None if start is None else (start, end), n_slices=number("n_slices", int, optional=True),
        cover=v["cover"], threshold=number("threshold"), vmax=number("vmax", optional=True), cmap=v["cmap"].strip() or "autumn",
        projection=v["projection"], brain_mask=v["brain_mask"].strip() or None, flip_view=bool(v["flip_view"]),
        slice_labels=bool(v["slice_labels"]), label_origin=number("label_origin"), label=v["label"], title=v["title"],
        colorbar=bool(v["colorbar"]), background="transparent" if v["transparent"] else "white", dpi=number("dpi", int))


def run_gui():
    """Form for all settings; 'Plot' saves the figure and opens it, the form stays open to adjust and plot again.
    The entries are remembered in plot_slice_stack_last.json next to this script."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        from fmri_paths import get_fmri_root
        start_dir = get_fmri_root()
    except Exception:
        start_dir = str(Path.home())
    saved = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.is_file() else {}

    root = tk.Tk()
    root.title("Slice stack plot")
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    frame = ttk.Frame(root, padding=12)
    frame.grid(sticky="nsew")
    variables = {}

    def browse(key, kind):
        current = variables[key].get()
        options = {"initialdir": str(Path(current).parent) if current else start_dir}
        if kind == "save":
            chosen = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image", "*.png *.pdf *.svg")], **options)
        else:
            chosen = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.gz"), ("All files", "*")], **options)  # "*.nii.gz" crashes Tk on macOS
        if chosen:
            variables[key].set(chosen)

    for row, (key, label, default, kind, choices) in enumerate(GUI_FIELDS):
        value = saved.get(key, default)
        if kind == "check":
            variables[key] = tk.BooleanVar(value=bool(value))
            ttk.Checkbutton(frame, text=label, variable=variables[key]).grid(row=row, column=1, sticky="w", pady=1)
            continue
        variables[key] = tk.StringVar(value=str(value))
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=1)
        if kind == "choice":
            ttk.Combobox(frame, textvariable=variables[key], values=choices, state="readonly", width=12).grid(row=row, column=1, sticky="w")
        else:
            ttk.Entry(frame, textvariable=variables[key], width=70 if kind in ("file", "save") else 14).grid(row=row, column=1, sticky="w")
            if kind in ("file", "save"):
                ttk.Button(frame, text="Browse...", command=lambda k=key, t=kind: browse(k, t)).grid(row=row, column=2, padx=6)

    def on_map_changed(*_):
        map_file = variables["map"].get().strip()
        if not map_file or not Path(map_file).is_file():
            return
        if not variables["underlay"].get().strip() or not Path(variables["underlay"].get()).is_file():
            variables["underlay"].set(_guess_underlay(map_file))
        if not variables["out"].get().strip():
            variables["out"].set(str(Path(map_file).parent / (Path(map_file).name.split(".")[0] + "_slice_stack.png")))

    variables["map"].trace_add("write", on_map_changed)
    status = tk.StringVar()

    def plot():
        values = {k: var.get() for k, var in variables.items()}
        try:
            kwargs = _gui_values_to_kwargs(values)
        except ValueError as err:
            messagebox.showerror("Invalid input", str(err))
            return
        SETTINGS_FILE.write_text(json.dumps(values, indent=1))
        status.set("Plotting ...")
        root.update_idletasks()
        try:
            fig = plot_slice_stack(**kwargs)
            plt.close(fig)
        except Exception as err:
            status.set("")
            messagebox.showerror("Plotting failed", f"{type(err).__name__}: {err}")
            return
        status.set(f"Saved: {kwargs['out_file']}")
        _open_file(kwargs["out_file"])

    buttons = ttk.Frame(frame)
    buttons.grid(row=len(GUI_FIELDS), column=0, columnspan=3, pady=(12, 0), sticky="ew")
    ttk.Button(buttons, text="Plot", command=plot).pack(side="left")
    ttk.Button(buttons, text="Close", command=root.destroy).pack(side="left", padx=8)
    ttk.Label(buttons, textvariable=status).pack(side="left", padx=8)
    on_map_changed()
    root.mainloop()


def main():
    plt.switch_backend("Agg")  # run as a script, figures are only saved (the form opens them with the system viewer)
    if len(sys.argv) == 1:
        run_gui()
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map", help="Signal change / t-stat map in atlas space (NIfTI)")
    p.add_argument("underlay", help="Atlas template on the same grid (NIfTI)")
    p.add_argument("out", help="Output image (.png, .pdf, .svg)")
    p.add_argument("--axis", choices=AXES, default="coronal")
    p.add_argument("--thickness", type=float, default=2.5, help="slab thickness in mm")
    p.add_argument("--gap", type=float, default=2.0, help="space between slabs in mm")
    p.add_argument("--azimuth", type=float, default=35.0)
    p.add_argument("--elevation", type=float, default=25.0)
    p.add_argument("--range", nargs=2, type=float, metavar=("START", "END"), dest="slice_range",
                   help="range along the slice axis in atlas mm (default: extent of the map)")
    p.add_argument("--cover", choices=("data", "brain"), default="data")
    p.add_argument("--n-slices", type=int)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--vmax", type=float)
    p.add_argument("--cmap", default="autumn")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--projection", choices=("mean", "max"), default="mean")
    p.add_argument("--brain-mask")
    p.add_argument("--upsample", type=int, default=3)
    p.add_argument("--flip-view", action="store_true", help="look at the slices from the other side")
    p.add_argument("--no-labels", action="store_true", help="no slice position labels")
    p.add_argument("--label-origin", type=float, default=0.0, help="mm subtracted from the slice labels (e.g. bregma)")
    p.add_argument("--label", default="", help="colorbar label")
    p.add_argument("--title", default="")
    p.add_argument("--no-colorbar", action="store_true")
    p.add_argument("--background", default="white", help="colour or 'transparent'")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()
    plot_slice_stack(a.map, a.underlay, a.out, axis=a.axis, thickness=a.thickness, gap=a.gap, azimuth=a.azimuth,
                     elevation=a.elevation, slice_range=a.slice_range, n_slices=a.n_slices, cover=a.cover,
                     threshold=a.threshold, vmax=a.vmax, cmap=a.cmap, alpha=a.alpha, projection=a.projection,
                     brain_mask=a.brain_mask, upsample=a.upsample, flip_view=a.flip_view,
                     slice_labels=not a.no_labels, label_origin=a.label_origin, label=a.label, title=a.title,
                     colorbar=not a.no_colorbar, background=a.background, dpi=a.dpi)


if __name__ == "__main__":
    main()
