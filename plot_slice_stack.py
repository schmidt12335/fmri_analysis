"""3D figures of signal change maps on an atlas: slice stack, volume rendering and surface rendering.

view="stack" (default)   the map on thick slices of the atlas template. Two layouts:
                           layout="tiled"    flat tiles side by side in rows (like the slice stacks in fUSI / fMRI papers), seen from a
                                             slightly oblique angle with an orthographic camera, so every slice is fully visible
                           layout="physical" the slabs stand behind each other, `thickness` mm of atlas with `gap` mm of space in between
view="volume"            volume rendering: translucent brain with the map as glowing volume inside
view="surface"           surface rendering: translucent brain surface with nested iso-surfaces of the map

Every view can be saved as a figure (png/pdf/svg, with colorbar), opened in an interactive window that can be rotated with the mouse
and written as a rotating video (mp4). The rendering is done with pyvista / VTK (`pip install pyvista imageio-ffmpeg`).

Run without arguments (`python plot_slice_stack.py`) to get a form that asks for the files and all settings.

Command line (maps from the atlas_space folder of the electric stimulation notebooks, same grid as the template):
    python plot_slice_stack.py atlas_space/stim_diff_psc_atlasspace.nii.gz template_registration_0.1mm.nii.gz stack.png \
        --threshold 0.5 --vmax 3 --label "Signal change (%)"
    python plot_slice_stack.py map.nii.gz template.nii.gz vol.png --view volume --movie vol.mp4
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
from types import SimpleNamespace

import matplotlib

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from nibabel.processing import resample_from_to
from scipy import ndimage

# Per slice direction: (index of the slice axis, viewing direction, "up" direction on screen) in RAS world axes
# (x = right, y = anterior, z = superior). Coronal is seen from posterior, sagittal from the left, axial from above.
AXES = {
    "coronal": (1, np.array([0, 1, 0]), np.array([0, 0, 1])),
    "sagittal": (0, np.array([1, 0, 0]), np.array([0, 0, 1])),
    "axial": (2, np.array([0, 0, -1]), np.array([0, 1, 0])),
}
# camera presets of the tiled layout: (azimuth, elevation) in degrees
TILE_VIEWS = {"paper oblique": (14.0, 10.0), "shallow oblique": (7.0, 5.0), "face-on": (0.0, 0.0)}
VIEWS = ("stack", "volume", "surface")
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


def _alpha_ramp(frac, alpha_min, alpha_max):
    """Opacity of the map: alpha_min just above the threshold, alpha_max at the colour scale maximum. The square root lifts weak
    values so that they are still clearly visible."""
    return alpha_min + (alpha_max - alpha_min) * np.sqrt(np.clip(frac, 0, 1))


def _overlay_colors(values, cmap_pos, cmap_neg, threshold, vmax, alpha_min=1.0, alpha_max=1.0):
    """RGBA for the overlay values; voxels with |value| <= threshold are transparent."""
    span = max(vmax - threshold, 1e-9)
    frac = np.clip((np.abs(values) - threshold) / span, 0, 1)
    rgba = np.where((values >= 0)[..., None], cmap_pos(frac), cmap_neg(frac))
    rgba[..., 3] = np.where(np.abs(values) > threshold, _alpha_ramp(frac, alpha_min, alpha_max), 0)
    return rgba


def _pyvista():
    """Imports pyvista (with a helpful message if it is missing)."""
    try:
        import pyvista as pv
    except ImportError as err:
        raise ImportError("This script needs pyvista: pip install pyvista imageio-ffmpeg") from err
    return pv


def _upsample_inplane(arr, factor, slice_axis, order=1):
    """Smooth enlargement of a 3D array in the two in-plane directions (the slice direction is kept).
    The first and last sample stay where they are, so the spacing becomes (n - 1) * spacing / (m - 1)."""
    if factor == 1:
        return arr
    zoom = [1.0 if a == slice_axis else float(factor) for a in range(3)]
    return ndimage.zoom(arr, zoom, order=order)


def _prepare(map_file, underlay_file, axis, flip_view, brain_mask, under_threshold, threshold, vmax, cmap):
    """Loads and normalises the data; everything the views share."""
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
    aff = under_img.affine
    if np.abs(aff[:3, :3] - np.diag(np.diag(aff[:3, :3]))).max() > 1e-4:
        raise ValueError("The images must be in standard (non-oblique) orientation.")
    in_brain = under[mask3d]
    norm_under = (np.percentile(in_brain, 1), np.percentile(in_brain, 99.5))
    if vmax is None:
        shown = np.abs(over[np.abs(over) > threshold])
        vmax = float(np.percentile(shown, 99.5)) if shown.size else threshold + 1
    cmap_pos = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    world = [aff[a, a] * np.arange(under.shape[a]) + aff[a, 3] for a in range(3)]
    extent = []
    for a in range(3):
        present = np.where(mask3d.any(axis=tuple(b for b in range(3) if b != a)))[0]
        extent.append((world[a][present].min(), world[a][present].max()))
    return SimpleNamespace(
        slice_axis=slice_axis, view_dir=view_dir, up=up.astype(float), right=right.astype(float), under=under, over=over, mask=mask3d,
        spacing=np.abs(np.diag(aff)[:3]), origin=aff[:3, 3], world=world, norm_under=norm_under, vmax=vmax, threshold=threshold,
        cmap_pos=cmap_pos, cmaps=(cmap_pos, NEG_CMAP), extent=np.array(extent), negative=bool((over < -threshold).any()))


def _gray(c, values):
    lo, hi = c.norm_under
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)


def _direction(c, azimuth, elevation):
    """Unit vector from the scene towards the camera."""
    az, el = np.radians(azimuth), np.radians(elevation)
    return np.cos(el) * (np.cos(az) * (-c.view_dir) + np.sin(az) * c.right) + np.sin(el) * c.up


def _slab_centers(c, layout, thickness, gap, n_slices, slice_range, cover, reverse_order):
    support = c.mask if cover == "brain" else (np.abs(c.over) > c.threshold)
    present = np.where(support.any(axis=tuple(a for a in range(3) if a != c.slice_axis)))[0]
    if slice_range is None:
        if present.size == 0:
            raise ValueError("Nothing to show: the map has no voxels above the threshold inside the brain.")
        lo, hi = c.world[c.slice_axis][present].min(), c.world[c.slice_axis][present].max()
    else:
        lo, hi = sorted(slice_range)
    if layout == "tiled":
        n = int(n_slices or 8)
        centers = np.linspace(lo, hi, n) if n > 1 else np.array([(lo + hi) / 2])
        if thickness is None:
            thickness = float(np.clip((hi - lo) / max(n - 1, 1), c.spacing[c.slice_axis], 1.0))
    else:
        thickness = 2.5 if thickness is None else thickness
        pitch = thickness + gap
        n_fit = int(np.floor((hi - lo) / pitch)) + 1
        n = min(n_slices, n_fit) if n_slices else n_fit
        centers = (lo + hi) / 2 - (n - 1) * pitch / 2 + pitch * np.arange(n)
    descending = c.slice_axis != 0  # coronal: anterior first, axial: dorsal first, sagittal: left to right
    if descending != bool(reverse_order):
        centers = centers[::-1]
    return centers, float(thickness)


def _build_slab(pv, c, center, thickness, direction, *, projection, upsample, alpha, alpha_min, alpha_max, outline, side_color,
                edge_voxels, tissue_threshold, both_caps):
    """One slab as closed 3D solid in world coordinates. Returns None if the slab has no brain."""
    slice_axis = c.slice_axis
    members = np.where(np.abs(c.world[slice_axis] - center) <= thickness / 2 + 1e-6)[0]
    if members.size == 0:
        members = np.array([int(np.argmin(np.abs(c.world[slice_axis] - center)))])
    sub_under = c.under.take(members, axis=slice_axis)
    sub_mask = c.mask.take(members, axis=slice_axis).astype(np.float32)
    sub_over = c.over.take(members, axis=slice_axis)
    slab_over = np.expand_dims(sub_over.max(axis=slice_axis) if projection == "max" else sub_over.mean(axis=slice_axis), slice_axis)

    # Outline of the slab = outline of its front layer (the one that faces the camera), extruded through the slab. The brain tapers, so the
    # average over the layers would be larger than the front layer and show background at the rim of the front face. The outline is the
    # brain mask restricted to bright tissue (the templates have a dark rim of dura/CSF inside the mask), holes (ventricles) filled,
    # largest piece only.
    front_layer = -1 if direction[slice_axis] > 0 else 0
    front_gray = _gray(c, sub_under.take(front_layer, axis=slice_axis))
    outline_2d = (sub_mask.take(front_layer, axis=slice_axis) > 0.5) & (ndimage.gaussian_filter(front_gray, 0.7) > tissue_threshold)
    outline_2d = ndimage.binary_fill_holes(ndimage.binary_opening(outline_2d))
    pieces, n_pieces = ndimage.label(outline_2d)
    if n_pieces > 1:
        outline_2d = pieces == 1 + int(np.argmax(ndimage.sum(outline_2d, pieces, range(1, n_pieces + 1))))
    if not outline_2d.any():
        return None

    # Everything in the two in-plane directions at `upsample` times the resolution
    prism = _upsample_inplane(np.expand_dims(outline_2d.astype(np.float32), slice_axis), upsample, slice_axis)
    sigma = [0.0 if a == slice_axis else 1.3 * upsample for a in range(3)]
    prism = np.repeat(ndimage.gaussian_filter(prism, sigma), len(members), axis=slice_axis)
    under_up = _upsample_inplane(sub_under, upsample, slice_axis, order=3)
    over_up = _upsample_inplane(slab_over, upsample, slice_axis)
    gray = _gray(c, under_up)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    ov = _overlay_colors(np.repeat(over_up, len(members), axis=slice_axis), *c.cmaps, c.threshold, c.vmax, alpha_min, alpha_max)
    a = (ov[..., 3] * alpha)[..., None]
    rgb = rgb * (1 - a) + ov[..., :3] * a
    rgb_sides = np.array([0.62, 0.62, 0.62], dtype=np.float32) * (1 - a) + ov[..., :3] * a  # neutral tissue gray, map colour where active

    # The template is 0 outside the brain: take the colour of the nearest voxel well inside the outline, so that neither the sides
    # nor the rim of the front face turn black
    footprint = np.ones([1 if a_ == slice_axis else 3 for a_ in range(3)], dtype=bool)
    interior = ndimage.binary_erosion(prism > 0.5, structure=footprint, iterations=max(1, int(round(edge_voxels * upsample))))
    if interior.any():
        nearest = tuple(ndimage.distance_transform_edt(~interior, return_distances=False, return_indices=True))
        rgb = rgb[nearest]
        rgb_sides = rgb_sides[nearest]
    if side_color == "anatomy":
        rgb_sides = rgb

    # Pad (closed solid; the colours continue outwards so that nothing darkens at the borders)
    pad = 2
    pad_width = [(pad, pad)] * 3
    pad_width[slice_axis] = (1, 1)
    prism = np.pad(prism, pad_width)
    rgb = np.pad(rgb, pad_width + [(0, 0)], mode="edge")
    rgb_sides = np.pad(rgb_sides, pad_width + [(0, 0)], mode="edge")

    orig_dims, up_dims = np.array(sub_under.shape), np.array(under_up.shape)
    step = np.where(up_dims > 1, (orig_dims - 1) * c.spacing / np.maximum(up_dims - 1, 1), c.spacing)
    origin = np.array([c.world[a_][members[0]] if a_ == slice_axis else c.world[a_][0] for a_ in range(3)])
    origin = origin - np.array([1 if a_ == slice_axis else pad for a_ in range(3)]) * step
    grid = pv.ImageData(dimensions=tuple(int(d) for d in prism.shape), spacing=tuple(float(s) for s in step), origin=tuple(float(o) for o in origin))
    grid.point_data["prism"] = prism.ravel(order="F")
    grid.point_data["rgb"] = np.stack([rgb[..., k].ravel(order="F") for k in range(3)], axis=1).astype(np.float32)
    grid.point_data["rgb_sides"] = np.stack([rgb_sides[..., k].ravel(order="F") for k in range(3)], axis=1).astype(np.float32)

    surface = grid.contour([0.5], scalars="prism", method="marching_cubes").clean()
    if surface.n_points == 0:
        return None
    surface = surface.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True, consistent_normals=True)
    surface = surface.sample(grid)
    normal_along = surface.cell_data["Normals"][:, slice_axis]
    is_cap = np.abs(normal_along) > 0.95
    # only the face turned to the camera is drawn (the other one is never visible), unless the scene can be turned around
    keep = is_cap if both_caps else is_cap & (normal_along * np.sign(direction[slice_axis] or 1.0) > 0)
    caps = surface.extract_cells(np.where(keep)[0]).extract_surface(algorithm=None)
    body = surface.extract_cells(np.where(~is_cap)[0]).extract_surface(algorithm=None)
    caps.point_data["color"] = caps.point_data["rgb"]
    body.point_data["color"] = body.point_data["rgb_sides"]
    edge = caps.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False) if outline and caps.n_points else None
    return SimpleNamespace(caps=caps, body=body, edge=edge, position=float(c.world[slice_axis][members].mean()))


def _place(mesh, c, position, shift, depth_gain):
    """Tiled layout: the slab is moved to the tile (shift = mm along `right` and `up`), its thickness exaggerated for display."""
    out = mesh.copy()
    pts = np.asarray(out.points, dtype=float).copy()
    centre = c.centre.copy()
    centre[c.slice_axis] = position
    pts = pts - centre
    pts[:, c.slice_axis] *= depth_gain
    out.points = pts + c.right * shift[0] + c.up * shift[1]
    return out


def _tile_columns(n, width, height, gap, rise, figsize):
    """Number of columns that makes the tiles as large as possible on a figure of the given proportions."""
    best, best_scale = 1, 0.0
    for cols in range(1, min(n, 10) + 1):
        rows = int(np.ceil(n / cols))
        w = cols * width + (cols - 1) * gap * width
        h = (rows - 1) * height * (1.2 + (cols - 1) * rise) + height * (1 + (cols - 1) * rise)
        scale = min(figsize[0] / w, figsize[1] / h)
        if scale > best_scale + 1e-12:
            best, best_scale = cols, scale
    return best


def _bounds_corners(meshes):
    out = []
    for m in meshes:
        b = m.bounds
        out.extend([(x, y, z) for x in b[0:2] for y in b[2:4] for z in b[4:6]])
    return np.array(out)


def _basis(c, direction):
    """Screen axes (right, up) for a camera that looks from `direction` at the scene."""
    view = -direction
    right = np.cross(view, c.up)
    right /= np.linalg.norm(right)
    return right, np.cross(right, view)


def _add_scalar_bar(pl, pv, bar_cmap, bar_limits, label, text_color, anchor):
    """Colorbar in the 3D window: an invisible point carries the colour scale."""
    marker = pv.PolyData(np.array([anchor]))
    marker["scale"] = np.array([bar_limits[0]])
    pl.add_mesh(marker, scalars="scale", cmap=bar_cmap, clim=bar_limits, opacity=0.0, show_scalar_bar=True,
                scalar_bar_args={"title": label or " ", "vertical": True, "position_x": 0.9, "position_y": 0.25, "height": 0.5, "width": 0.05,
                                 "color": text_color, "n_labels": 5, "fmt": "%.2g"})


def _iso_surface(pv, c, volume, level, sigma, smooth_iterations=25):
    """Smooth closed iso-surface of a 3D array (padded, so that it is also closed at the border of the image)."""
    vol = np.pad(ndimage.gaussian_filter(volume.astype(np.float32), sigma) if sigma else volume.astype(np.float32), 2)
    grid = pv.ImageData(dimensions=vol.shape, spacing=tuple(float(s) for s in c.spacing), origin=tuple(float(o) for o in c.origin - 2 * c.spacing))
    grid.point_data["v"] = vol.ravel(order="F")
    surf = grid.contour([level], scalars="v", method="marching_cubes").clean()
    if surf.n_points == 0:
        return None
    surf = surf.extract_surface(algorithm=None)
    if smooth_iterations:
        surf = surf.smooth_taubin(n_iter=smooth_iterations, pass_band=0.1)
    return surf.compute_normals(point_normals=True, cell_normals=False, split_vertices=False)


def plot_slice_stack(map_file, underlay_file, out_file=None, *, view="stack", layout="tiled", axis="coronal", thickness=None, gap=2.0,
                     azimuth=35.0, elevation=25.0, tile_view="paper oblique", n_slices=None, depth_gain=3.0, tile_gap=0.15, tile_rise=0.2,
                     columns=None, reverse_order=False, slice_range=None, cover="data", threshold=0.0, vmax=None, cmap="autumn", alpha=1.0,
                     alpha_min=0.55, alpha_max=1.0, projection="mean", brain_mask=None, under_threshold=0.02, upsample=3, flip_view=False,
                     slice_labels=True, label_origin=0.0, label="", title="", colorbar=True, background="white", dpi=300, figsize=(9, 7),
                     show=False, view_angle=25.0, outline=False, side_color="gray", edge_voxels=1.5, tissue_threshold=0.15,
                     surface_levels=3, volume_opacity=1.0, interactive=False, view_file=None, movie_file=None, movie_seconds=8.0, movie_fps=20,
                     _screenshot_instead=None):
    """Draws the 3D figure and saves it to `out_file` (png/pdf/svg). Returns the matplotlib figure (None if only the viewer/movie are made).

    view            "stack" (slices), "volume" (volume rendering) or "surface" (surface rendering)
    layout          stack only: "tiled" (default, flat tiles in rows, orthographic) or "physical" (slabs behind each other with gaps)
    axis            slice direction: "coronal" (default), "sagittal" or "axial"; in the volume and surface views the direction the scene is seen from
    n_slices        tiled: number of tiles (default 8), evenly spread over the range; physical: use only this many slabs
    thickness       mm of atlas averaged for each slice (tiled: default = distance between the slices, at most 1; physical: 2.5)
    slice_range     (start, end) in mm along the slice axis (atlas world coordinates); default: `cover`
    cover           "data" = extent of the map (default), "brain" = whole brain
    tile_view       tiled camera: "paper oblique" (default), "shallow oblique", "face-on" or "custom" (uses azimuth and elevation)
    depth_gain      tiled: the slice thickness is exaggerated by this factor in the display, so the sides of the tiles are visible
    tile_gap, tile_rise  tiled: space between the tiles as fraction of their width and the step down of each next tile in a row as fraction of its height
    columns         tiled: tiles per row (default: automatic)
    reverse_order   tiles in the other order
    gap             physical: mm of empty space between the slabs
    azimuth         degrees the camera is turned to the side (0 = looking straight onto the slices)
    elevation       degrees the camera is raised above the slices
    view_angle      perspective opening angle in degrees, 0 = parallel projection (not used by the tiled layout, which is always parallel)
    threshold, vmax values <= threshold are not shown, vmax is the end of the colour scale (default: 99.5th percentile)
    cmap            colormap of positive values (negative values, if present, are shown in cyan-blue)
    alpha_min, alpha_max  opacity of the map just above the threshold and at vmax (weak values stay translucent so the anatomy shows through)
    alpha           overall opacity factor of the map on the slices
    projection      "mean" or "max" of the map over the slice thickness ("max" is the better choice for small effects)
    flip_view       look at the slices from the other side
    label_origin    mm that are subtracted from the slice positions in the labels (e.g. the y coordinate of bregma)
    outline         thin dark line around the faces of the slices (off by default)
    side_color      "gray" (default: sides in neutral tissue gray, shaded, with the map where it is active) or "anatomy"
    tissue_threshold  the outline of a slab follows the tissue brighter than this fraction of the template's tissue range (0 = the brain
                    mask as it is); removes the dark rim at the edge of the brain
    edge_voxels     width (in voxels) of the border of the brain that takes the colour of the tissue next to it; hides the dark band of the
                    template at the brain edge
    surface_levels  surface view: number of nested iso-surfaces of the map
    volume_opacity  volume view: factor for the opacity of the brain (1 = default, lower = more transparent)
    interactive     open the scene in a window that can be rotated with the mouse (drag = rotate, scroll = zoom, shift + drag = pan,
                    `s` = save the current view as PNG to `view_file`, `q` = close). The window blocks until it is closed. The figure
                    file `out_file` is still written first if it is given (it may be omitted in this mode).
    movie_file      write a video (.mp4): the tiled and physical stacks sway to both sides, the volume and surface views turn around once
    """
    pv = _pyvista()
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    if layout not in ("tiled", "physical"):
        raise ValueError("layout must be 'tiled' or 'physical'")
    tiled = view == "stack" and layout == "tiled"
    c = _prepare(map_file, underlay_file, axis, flip_view, brain_mask, under_threshold, threshold, vmax, cmap)
    vmax = c.vmax
    c.centre = c.extent.mean(axis=1)
    if tiled and tile_view != "custom":
        azimuth, elevation = TILE_VIEWS[tile_view]
    direction = _direction(c, azimuth, elevation)

    bg = "white" if background == "transparent" else background
    if c.negative:
        bar_cmap, bar_limits = ListedColormap(np.vstack([NEG_CMAP(np.linspace(1, 0, 256)), c.cmap_pos(np.linspace(0, 1, 256))])), (-vmax, vmax)
    else:
        bar_cmap, bar_limits = c.cmap_pos, (threshold, vmax)

    scene = SimpleNamespace(parallel=False, sway=0.0, labels=[], caption="")
    stack_options = dict(projection=projection, upsample=upsample, alpha=alpha, alpha_min=alpha_min, alpha_max=alpha_max, outline=outline,
                         side_color=side_color, edge_voxels=edge_voxels, tissue_threshold=tissue_threshold, both_caps=interactive)
    slabs, volume_items, surface_items = [], [], []

    if view == "stack":
        centers, thickness = _slab_centers(c, layout, thickness, gap, n_slices, slice_range, cover, reverse_order)
        for center in centers:
            slab = _build_slab(pv, c, center, thickness, direction, **stack_options)
            if slab is not None:
                slabs.append(slab)
        if not slabs:
            raise ValueError("No slab contains brain voxels.")
        scene.sway = 25.0
        if tiled:
            r_axis, u_axis = int(np.argmax(np.abs(c.right))), int(np.argmax(np.abs(c.up)))
            width, height = c.extent[r_axis][1] - c.extent[r_axis][0], c.extent[u_axis][1] - c.extent[u_axis][0]
            n = len(slabs)
            cols = int(columns) if columns else _tile_columns(n, width, height, tile_gap, tile_rise, figsize)
            cols = max(1, min(cols, n))
            row_step = height * (1.2 + (cols - 1) * tile_rise)
            for i, slab in enumerate(slabs):
                row, col = divmod(i, cols)
                shift = (col * width * (1 + tile_gap), -row * row_step - col * tile_rise * height)
                slab.caps = _place(slab.caps, c, slab.position, shift, depth_gain)
                slab.body = _place(slab.body, c, slab.position, shift, depth_gain)
                if slab.edge is not None:
                    slab.edge = _place(slab.edge, c, slab.position, shift, depth_gain)
                slab.shift = shift
            scene.parallel = True
            if slice_labels:
                names = "xyz"[c.slice_axis]
                scene.labels = [(f"{names} = {s.position - label_origin:+.1f} mm", s.shift, height) for s in slabs]
        else:
            scene.caption = f"{axis.capitalize()} slices at " + ", ".join(f"{s.position - label_origin:+.1f}" for s in slabs) + " mm"
        for s in slabs:
            s.caps_normals = s.caps
            s.body = s.body.compute_normals(point_normals=True, cell_normals=False, split_vertices=False)
        pts = _bounds_corners([m for s in slabs for m in (s.caps, s.body)])
        if tiled and slice_labels:  # room for the labels below the tiles
            below = np.array([c.right * s.shift[0] + c.up * (s.shift[1] - height * 0.75) for s in slabs])
            pts = np.vstack([pts, below])
    else:
        pts = np.array([(x, y, z) for x in c.extent[0] for y in c.extent[1] for z in c.extent[2]])
        gray = _gray(c, c.under) * c.mask
        if view == "volume":
            grid = pv.ImageData(dimensions=c.under.shape, spacing=tuple(float(s) for s in c.spacing), origin=tuple(float(o) for o in c.origin))
            grid.point_data["anat"] = ndimage.gaussian_filter(gray, 0.6).ravel(order="F")
            volume_items.append(("anat", grid))
            frac_axis = np.linspace(0, 1, 100)  # (a table of exactly 256 entries is misread by pyvista)
            c.table_anat = volume_opacity * 0.08 * np.clip((frac_axis - 0.2) / 0.8, 0, 1) ** 1.5
            c.table_map = np.where(frac_axis > 0, _alpha_ramp(frac_axis, alpha_min, alpha_max), 0.0)
            grid.point_data["pos"] = np.where(c.over > threshold, c.over, 0).ravel(order="F").astype(np.float32)
            grid.point_data["neg"] = np.where(c.over < -threshold, -c.over, 0).ravel(order="F").astype(np.float32)
            c.volume_grid = grid
        else:
            closed = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(c.mask, 3), iterations=3))[3:-3, 3:-3, 3:-3]
            c.brain_surface = _iso_surface(pv, c, closed.astype(np.float32), 0.5, 1.0, 30)
            levels = threshold + (vmax - threshold) * np.arange(max(surface_levels, 1)) / max(surface_levels, 1)
            for sign, cm in ((1, c.cmap_pos), (-1, NEG_CMAP)):
                for k, level in enumerate(levels):
                    surf = _iso_surface(pv, c, np.maximum(sign * c.over, 0), level + 1e-6 * vmax, 0.7)
                    if surf is not None:
                        opacity = 0.5 + 0.5 * k / max(len(levels) - 1, 1)
                        surface_items.append((surf, tuple(cm((level - threshold) / max(vmax - threshold, 1e-9))[:3]), opacity))
        scene.sway = 0.0

    centre3 = (pts.min(axis=0) + pts.max(axis=0)) / 2
    radius = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) / 2)
    scene.focal, scene.radius = centre3, radius

    def fit_orthographic(angles, aspect):
        """Camera scale and centre so that the tiles fill the picture, for each azimuth in `angles`."""
        half_w = half_h = 0.0
        for az in angles:
            right_c, up_c = _basis(c, _direction(c, az, elevation))
            x, y = (pts - centre3) @ right_c, (pts - centre3) @ up_c
            half_w, half_h = max(half_w, (x.max() - x.min()) / 2), max(half_h, (y.max() - y.min()) / 2)
        return max(half_h, half_w / aspect) * 1.04, half_w, half_h

    def set_camera(pl, az, aspect, parallel_scale=None):
        d = _direction(c, az, elevation)
        if scene.parallel:
            pl.camera.parallel_projection = True
            distance = 4 * radius
            pl.camera.parallel_scale = parallel_scale
        elif view_angle and view_angle > 0:
            pl.camera.view_angle = float(view_angle)
            distance = 0.8 * radius / np.sin(np.radians(view_angle) / 2)  # the box diagonal is larger than the brain
        else:
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = radius * 1.05 / min(1.0, aspect)
            distance = 4 * radius
        pl.camera.focal_point = tuple(scene.focal)
        pl.camera.position = tuple(scene.focal + d * distance)
        pl.camera.up = tuple(c.up)
        pl.camera.clipping_range = (max(distance - 2 * radius, 1e-3), distance + 2 * radius)

    def build_plotter(off_screen, size, antialiasing, line_scale):
        pl = pv.Plotter(off_screen=off_screen, window_size=list(size), lighting="none")
        pl.set_background(bg)
        if view == "stack":
            for slab in slabs:
                pl.add_mesh(slab.caps, scalars="color", rgb=True, lighting=False, show_scalar_bar=False)
                pl.add_mesh(slab.body, scalars="color", rgb=True, smooth_shading=True, ambient=0.45, diffuse=0.75, specular=0.08, specular_power=15,
                            show_scalar_bar=False)
                if slab.edge is not None and slab.edge.n_points:
                    pl.add_mesh(slab.edge, color=tuple(EDGE_COLOR), line_width=max(1.0, line_scale), render_lines_as_tubes=True, lighting=False)
        elif view == "volume":
            pl.add_volume(c.volume_grid, scalars="anat", cmap="gray", opacity=c.table_anat, clim=(0, 1), show_scalar_bar=False, mapper="gpu")
            for key, cm in (("pos", c.cmap_pos), ("neg", NEG_CMAP)):
                if c.volume_grid.point_data[key].max() > 0:
                    pl.add_volume(c.volume_grid, scalars=key, cmap=cm, opacity=c.table_map, clim=(threshold, vmax), show_scalar_bar=False,
                                  mapper="gpu")
        else:
            pl.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
            if c.brain_surface is not None:
                pl.add_mesh(c.brain_surface, color=(0.5, 0.54, 0.62), opacity=0.22, smooth_shading=True, ambient=0.35, diffuse=0.7,
                            specular=0.15, specular_power=20, show_scalar_bar=False, culling="back")
            for surf, color, opacity in surface_items:
                pl.add_mesh(surf, color=color, opacity=opacity, smooth_shading=True, ambient=0.35, diffuse=0.75, specular=0.3, specular_power=25,
                            show_scalar_bar=False)
        pl.add_light(pv.Light(light_type="headlight", intensity=0.55))
        pl.add_light(pv.Light(position=(-1.0, 1.2, 1.0), focal_point=(0, 0, 0), light_type="camera light", intensity=0.45))
        pl.enable_anti_aliasing(antialiasing)
        return pl

    text_color = "black" if matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(bg))[2] > 0.5 else "white"

    def decorate(pl):
        _add_scalar_bar(pl, pv, bar_cmap, bar_limits, label, text_color, scene.focal)
        if title:
            pl.add_text(title, position="upper_edge", font_size=12, color=text_color)
        if scene.caption:
            pl.add_text(scene.caption, position="lower_edge", font_size=9, color=text_color)

    fig = None
    if out_file is not None or not (interactive or movie_file):
        window = int(np.clip(max(figsize) * dpi, 600, 3600 if view == "stack" else 2400))
        aspect = figsize[0] / figsize[1]
        if scene.parallel:
            _, half_w, half_h = fit_orthographic([azimuth], 1.0)
            right_c, up_c = _basis(c, direction)
            x, y = (pts - centre3) @ right_c, (pts - centre3) @ up_c
            scene.focal = centre3 + right_c * (x.max() + x.min()) / 2 + up_c * (y.max() + y.min()) / 2  # centre of what is seen
            aspect = max(half_w / max(half_h, 1e-9), 0.3)
            scale = half_h * 1.04
            size = (window, int(window / aspect))
            if size[1] > window:
                size = (int(window * aspect), window)
        else:
            size = (window, int(window / aspect))
        pl = build_plotter(True, size, "ssaa" if view != "volume" else "fxaa", window / 1500)
        set_camera(pl, azimuth, size[0] / size[1], scale if scene.parallel else None)
        image = pl.screenshot(return_img=True, transparent_background=background == "transparent")
        pl.close()
        scene.focal_static, scene.focal = scene.focal, centre3
        figure_labels = []
        if scene.labels:
            right_c, up_c = _basis(c, direction)
            pixels_per_mm = size[1] / (2 * scale)
            for text, shift, tile_height in scene.labels:
                anchor = c.right * shift[0] + c.up * (shift[1] - tile_height * 0.58) - scene.focal_static
                figure_labels.append((text, size[0] / 2 + (anchor @ right_c) * pixels_per_mm, size[1] / 2 - (anchor @ up_c) * pixels_per_mm))

        # ---- crop to the content and add colorbar, labels and title with matplotlib
        background_rgb = np.array(matplotlib.colors.to_rgb(bg))
        if image.shape[2] == 4:
            content = image[..., 3] > 8
        else:
            content = np.any(np.abs(image[..., :3] / 255.0 - background_rgb) > 0.02, axis=2)
        for _, px, py in figure_labels:  # labels belong to the content
            content[int(np.clip(py + 0.04 * size[1], 0, size[1] - 1)), int(np.clip(px, 0, size[0] - 1))] = True
        rows, cols_ = np.where(content)
        margin = int(0.04 * max(image.shape[:2]))
        y0, y1 = max(rows.min() - margin, 0), min(rows.max() + margin, image.shape[0])
        x0, x1 = max(cols_.min() - margin, 0), min(cols_.max() + margin, image.shape[1])
        cropped = image[y0:y1, x0:x1]

        width_in = figsize[0] * (0.84 if colorbar else 0.96)
        height_in = width_in * cropped.shape[0] / cropped.shape[1]
        top, bottom = (0.45 if title else 0.08), (0.4 if scene.caption else 0.08)
        fig_w, fig_h = figsize[0], height_in + top + bottom
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="none" if background == "transparent" else background)
        ax = fig.add_axes([0.02, bottom / fig_h, width_in / fig_w, height_in / fig_h])
        ax.imshow(cropped, interpolation="lanczos", extent=(0, cropped.shape[1], cropped.shape[0], 0))
        ax.axis("off")
        font_px = cropped.shape[1] / (width_in * 72)  # pixels per point
        for text, px, py in figure_labels:
            ax.text(px - x0, py - y0, text, ha="center", va="top", fontsize=7, color=text_color)
        if scene.caption:
            fig.text(0.5, 0.12 / fig_h, scene.caption, ha="center", va="bottom", fontsize=9)
        if title:
            fig.text(0.5, 1 - 0.1 / fig_h, title, ha="center", va="top", fontsize=11)
        if colorbar:
            mappable = ScalarMappable(Normalize(*bar_limits), bar_cmap)
            cax = fig.add_axes([0.90, bottom / fig_h + 0.25 * height_in / fig_h, 0.018, 0.5 * height_in / fig_h])
            cbar = fig.colorbar(mappable, cax=cax)
            cbar.set_label(label)
        if out_file:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_file, dpi=dpi, transparent=background == "transparent")
            what = f"{len(slabs)} slices" if view == "stack" else view
            print(f"[OK] Figure saved: {out_file}  ({view}, {what})")

    if movie_file:
        movie_size = (1280, 720)
        frames = max(int(movie_seconds * movie_fps), 2)
        if scene.sway:
            angles = azimuth + scene.sway * np.sin(2 * np.pi * np.arange(frames) / frames)
        else:
            angles = azimuth + 360.0 * np.arange(frames) / frames
        scale_movie = None
        if scene.parallel:
            scale_movie = fit_orthographic(np.linspace(azimuth - scene.sway, azimuth + scene.sway, 9), movie_size[0] / movie_size[1])[0]
        pl = build_plotter(True, movie_size, "fxaa" if view == "volume" else "ssaa", 1.0)
        decorate(pl)
        Path(movie_file).parent.mkdir(parents=True, exist_ok=True)
        pl.open_movie(str(movie_file), framerate=int(movie_fps), quality=8)
        for az in angles:
            set_camera(pl, float(az), movie_size[0] / movie_size[1], scale_movie)
            pl.write_frame()
        pl.close()
        print(f"[OK] Movie saved: {movie_file}  ({frames} frames)")

    if interactive:
        save_to = Path(view_file) if view_file else (Path(out_file).with_name(Path(out_file).stem + "_view.png") if out_file else Path("slice_stack_view.png"))
        size = (1400, 1000)
        pl = build_plotter(_screenshot_instead is not None, size, "msaa", 1.5)
        decorate(pl)
        scale_view = fit_orthographic([azimuth], size[0] / size[1])[0] if scene.parallel else None
        set_camera(pl, azimuth, size[0] / size[1], scale_view)
        pl.add_text("drag: rotate   scroll: zoom   shift+drag: move   s: save view as PNG   q: close", position="upper_left", font_size=7, color=text_color)

        def save_view():
            save_to.parent.mkdir(parents=True, exist_ok=True)
            pl.screenshot(str(save_to), transparent_background=background == "transparent")
            print(f"[OK] View saved: {save_to}")

        pl.add_key_event("s", save_view)
        if _screenshot_instead is not None:  # test mode: no window, one picture
            pl.screenshot(str(_screenshot_instead))
            pl.close()
        else:
            print("Interactive 3D viewer: drag to rotate, scroll to zoom, shift+drag to move, 's' saves the current view, 'q' closes it.")
            pl.show(title="Brain 3D")
    if show:
        plt.show()
    return fig


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


# (key, label, default, kind, choices): "file" fields get a Browse button, "choice" a drop-down, "check" a checkbox, "section" is a heading
GUI_FIELDS = [
    ("map", "Map in atlas space (.nii/.nii.gz)", "", "file", None),
    ("underlay", "Atlas template (same grid as the map)", "", "file", None),
    ("out", "Output image (.png/.pdf/.svg)", "", "save", None),
    ("view", "View", "stack", "choice", ["stack", "volume", "surface"]),
    ("_s1", "Slices (View = stack)", "", "section", None),
    ("layout", "Layout (tiled = flat tiles in rows, physical = slabs behind each other)", "tiled", "choice", ["tiled", "physical"]),
    ("axis", "Slice direction (also the viewing direction of volume / surface)", "coronal", "choice", list(AXES)),
    ("n_slices", "Number of slices (tiled: default 8; physical: max., empty = automatic)", "", "text", None),
    ("thickness", "Slice thickness (mm, empty = automatic)", "", "text", None),
    ("depth_gain", "Tiled: exaggerate the thickness (x)", "3", "text", None),
    ("tile_gap", "Tiled: space between tiles (fraction of width)", "0.15", "text", None),
    ("tile_rise", "Tiled: step down of each tile in a row (fraction of height)", "0.2", "text", None),
    ("columns", "Tiled: tiles per row (empty = automatic)", "", "text", None),
    ("gap", "Physical: gap between slabs (mm)", "2.0", "text", None),
    ("projection", "Map over slice thickness (max for small effects)", "mean", "choice", ["mean", "max"]),
    ("cover", "Slices cover", "data", "choice", ["data", "brain"]),
    ("range_start", "Range start (mm, empty = automatic)", "", "text", None),
    ("range_end", "Range end (mm, empty = automatic)", "", "text", None),
    ("reverse_order", "Reverse the order of the slices", False, "check", None),
    ("slice_labels", "Show slice position labels", True, "check", None),
    ("label_origin", "Slice labels: subtract (mm, e.g. bregma)", "0", "text", None),
    ("_s2", "Camera", "", "section", None),
    ("tile_view", "Tiled camera", "paper oblique", "choice", ["paper oblique", "shallow oblique", "face-on", "custom"]),
    ("azimuth", "Turn to the side (deg; physical, volume, surface, custom)", "35", "text", None),
    ("elevation", "Raise above the slices (deg)", "25", "text", None),
    ("view_angle", "Perspective angle (deg, 0 = none)", "25", "text", None),
    ("flip_view", "View from the other side", False, "check", None),
    ("_s3", "Map and colours", "", "section", None),
    ("threshold", "Threshold (values below are hidden)", "0", "text", None),
    ("vmax", "Colour scale maximum (empty = automatic)", "", "text", None),
    ("cmap", "Colormap (matplotlib name)", "autumn", "text", None),
    ("alpha_min", "Opacity of weak values (0-1)", "0.55", "text", None),
    ("surface_levels", "Surface: number of nested surfaces", "3", "text", None),
    ("volume_opacity", "Volume: brain opacity factor", "1", "text", None),
    ("brain_mask", "Brain mask (optional)", "", "file", None),
    ("label", "Colorbar label", "", "text", None),
    ("title", "Title", "", "text", None),
    ("colorbar", "Show colorbar", True, "check", None),
    ("transparent", "Transparent background", False, "check", None),
    ("dpi", "Resolution (dpi)", "300", "text", None),
    ("_s4", "Output", "", "section", None),
    ("interactive", "Also open an interactive 3D viewer (rotate with the mouse)", True, "check", None),
    ("movie", "Also write a video (.mp4, next to the image)", False, "check", None),
    ("movie_seconds", "Video length (s)", "8", "text", None),
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
    out = Path(v["out"])
    return dict(
        map_file=v["map"], underlay_file=v["underlay"], out_file=v["out"], view=v["view"], layout=v["layout"], axis=v["axis"],
        thickness=number("thickness", optional=True), gap=number("gap"), azimuth=number("azimuth"), elevation=number("elevation"),
        view_angle=number("view_angle"), tile_view=v["tile_view"], n_slices=number("n_slices", int, optional=True),
        depth_gain=number("depth_gain"), tile_gap=number("tile_gap"), tile_rise=number("tile_rise"), columns=number("columns", int, optional=True),
        reverse_order=bool(v["reverse_order"]), slice_range=None if start is None else (start, end), cover=v["cover"],
        threshold=number("threshold"), vmax=number("vmax", optional=True), cmap=v["cmap"].strip() or "autumn", alpha_min=number("alpha_min"),
        surface_levels=number("surface_levels", int), volume_opacity=number("volume_opacity"), projection=v["projection"],
        brain_mask=v["brain_mask"].strip() or None, flip_view=bool(v["flip_view"]), slice_labels=bool(v["slice_labels"]),
        label_origin=number("label_origin"), label=v["label"], title=v["title"], colorbar=bool(v["colorbar"]),
        background="transparent" if v["transparent"] else "white", dpi=number("dpi", int),
        movie_file=str(out.with_suffix(".mp4")) if v["movie"] else None, movie_seconds=number("movie_seconds"))


def run_gui():
    """Form for all settings; 'Plot' saves the figure (and video) and opens it, the form stays open to adjust and plot again.
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
    root.title("Brain 3D plot")
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    # scrollable form (it has many rows), the buttons below stay visible
    canvas = tk.Canvas(root, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame = ttk.Frame(canvas, padding=12)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta if sys.platform == "darwin" else e.delta / 120)), "units"))
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
        if kind == "section":
            ttk.Label(frame, text=label, font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 2))
            continue
        if kind == "check":
            variables[key] = tk.BooleanVar(value=bool(value))
            ttk.Checkbutton(frame, text=label, variable=variables[key]).grid(row=row, column=1, sticky="w", pady=1)
            continue
        variables[key] = tk.StringVar(value=str(value))
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=1)
        if kind == "choice":
            ttk.Combobox(frame, textvariable=variables[key], values=choices, state="readonly", width=14).grid(row=row, column=1, sticky="w")
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
            variables["out"].set(str(Path(map_file).parent / (Path(map_file).name.split(".")[0] + "_3d.png")))

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
        status.set("Plotting ..." + (" (the video takes a minute)" if kwargs["movie_file"] else ""))
        root.update_idletasks()
        # The 3D rendering runs in its own process: a 3D window and the Tk form do not share a process reliably (macOS)
        import tempfile
        job = Path(tempfile.mkdtemp()) / "job.json"
        job.write_text(json.dumps(kwargs))
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--json", str(job)], capture_output=True, text=True)
        if result.returncode != 0:
            status.set("")
            messagebox.showerror("Plotting failed", (result.stderr.strip().splitlines() or ["unknown error"])[-1])
            return
        status.set(f"Saved: {kwargs['out_file']}" + (f" and {kwargs['movie_file']}" if kwargs["movie_file"] else ""))
        _open_file(kwargs["out_file"])
        if values.get("interactive"):
            viewer_kwargs = dict(kwargs, out_file=None, movie_file=None, interactive=True,
                                 view_file=str(Path(kwargs["out_file"]).with_name(Path(kwargs["out_file"]).stem + "_view.png")))
            job_view = Path(tempfile.mkdtemp()) / "job_view.json"
            job_view.write_text(json.dumps(viewer_kwargs))
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--json", str(job_view)])
            status.set(status.get() + "; the 3D viewer is open (s = save the view)")

    buttons = ttk.Frame(root, padding=(12, 6))
    buttons.grid(row=1, column=0, columnspan=2, sticky="ew")
    ttk.Button(buttons, text="Plot", command=plot).pack(side="left")
    ttk.Button(buttons, text="Close", command=root.destroy).pack(side="left", padx=8)
    ttk.Label(buttons, textvariable=status).pack(side="left", padx=8)
    on_map_changed()
    root.update_idletasks()
    canvas.configure(width=frame.winfo_reqwidth(), height=min(frame.winfo_reqheight(), root.winfo_screenheight() - 160),
                     scrollregion=(0, 0, frame.winfo_reqwidth(), frame.winfo_reqheight()))
    root.mainloop()


def main():
    plt.switch_backend("Agg")  # run as a script, figures are only saved (the form opens them with the system viewer)
    if len(sys.argv) == 1:
        run_gui()
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--json":  # settings written by the form
        kwargs = json.loads(Path(sys.argv[2]).read_text())
        if kwargs.get("slice_range"):
            kwargs["slice_range"] = tuple(kwargs["slice_range"])
        plot_slice_stack(kwargs.pop("map_file"), kwargs.pop("underlay_file"), kwargs.pop("out_file", None), **kwargs)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map", help="Signal change / t-stat map in atlas space (NIfTI)")
    p.add_argument("underlay", help="Atlas template on the same grid (NIfTI)")
    p.add_argument("out", nargs="?", help="Output image (.png, .pdf, .svg); optional with --interactive or --movie")
    p.add_argument("--view", choices=VIEWS, default="stack")
    p.add_argument("--layout", choices=("tiled", "physical"), default="tiled")
    p.add_argument("--axis", choices=AXES, default="coronal")
    p.add_argument("--n-slices", type=int, help="number of slices (tiled: default 8)")
    p.add_argument("--thickness", type=float, help="slice thickness in mm")
    p.add_argument("--tile-view", choices=list(TILE_VIEWS) + ["custom"], default="paper oblique")
    p.add_argument("--depth-gain", type=float, default=3.0, help="tiled: exaggeration of the slice thickness")
    p.add_argument("--tile-gap", type=float, default=0.15)
    p.add_argument("--tile-rise", type=float, default=0.2)
    p.add_argument("--columns", type=int, help="tiled: tiles per row (default: automatic)")
    p.add_argument("--reverse-order", action="store_true")
    p.add_argument("--gap", type=float, default=2.0, help="physical: space between slabs in mm")
    p.add_argument("--azimuth", type=float, default=35.0)
    p.add_argument("--elevation", type=float, default=25.0)
    p.add_argument("--range", nargs=2, type=float, metavar=("START", "END"), dest="slice_range",
                   help="range along the slice axis in atlas mm (default: extent of the map)")
    p.add_argument("--cover", choices=("data", "brain"), default="data")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--vmax", type=float)
    p.add_argument("--cmap", default="autumn")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--alpha-min", type=float, default=0.55, help="opacity of weak values (0-1)")
    p.add_argument("--projection", choices=("mean", "max"), default="mean")
    p.add_argument("--view-angle", type=float, default=25.0, help="perspective opening angle in degrees, 0 = parallel projection ")
    p.add_argument("--surface-levels", type=int, default=3)
    p.add_argument("--volume-opacity", type=float, default=1.0)
    p.add_argument("--interactive", action="store_true", help="open an interactive 3D viewer (rotate with the mouse; s = save the view)")
    p.add_argument("--movie", metavar="FILE.mp4", help="also write a video")
    p.add_argument("--movie-seconds", type=float, default=8.0)
    p.add_argument("--outline", action="store_true", help="dark line around the faces of the slices")
    p.add_argument("--side-color", choices=("gray", "anatomy"), default="gray", help="sides of the slabs ")
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
    if not a.out and not a.interactive and not a.movie:
        p.error("an output file is required (or use --interactive / --movie)")
    plot_slice_stack(a.map, a.underlay, a.out, view=a.view, layout=a.layout, axis=a.axis, thickness=a.thickness, gap=a.gap, azimuth=a.azimuth,
                     elevation=a.elevation, tile_view=a.tile_view, n_slices=a.n_slices, depth_gain=a.depth_gain, tile_gap=a.tile_gap,
                     tile_rise=a.tile_rise, columns=a.columns, reverse_order=a.reverse_order, slice_range=a.slice_range, cover=a.cover,
                     threshold=a.threshold, vmax=a.vmax, cmap=a.cmap, alpha=a.alpha, alpha_min=a.alpha_min, projection=a.projection,
                     brain_mask=a.brain_mask, upsample=a.upsample, flip_view=a.flip_view, slice_labels=not a.no_labels,
                     label_origin=a.label_origin, label=a.label, title=a.title, colorbar=not a.no_colorbar, background=a.background,
                     dpi=a.dpi, view_angle=a.view_angle, outline=a.outline, side_color=a.side_color, surface_levels=a.surface_levels,
                     volume_opacity=a.volume_opacity, interactive=a.interactive, movie_file=a.movie, movie_seconds=a.movie_seconds)


if __name__ == "__main__":
    main()
