"""3D figures of signal change maps on the atlas: slice stack, volume rendering and surface rendering.

The rendering follows the 3D viewer of the deConfUSIon toolbox (fusiVolumeGUI, github.com/SonerCgu/deConfUSIon):

view="stack" (default)  slabs of the atlas (mean over `slab_slices` atlas slices) laid out as flat tiles next to each other, each tile a
                        little higher than its left neighbour, seen at a slight angle with a parallel (orthographic) camera. Every
                        tile has a front and a back face that show the anatomy with the map on top, and side walls along the exact
                        outline of the brain; the slab thickness is exaggerated by `depth_gain` for the display only.
view="volume"           volume rendering of one colour volume: the atlas anatomy is a faint blue-gray haze whose opacity is set for the
                        whole brain (`atlas_opacity`, along the longest path through the brain), the map is drawn in its colours with
                        an opacity that rises with the value, and the anatomy is dimmed around strong responses so they are not hidden.
view="surface"          translucent brain surface with the map as unsmoothed voxels (exact colour per voxel) or as a smooth contour.

The map is shown where the measurement covers the brain (`coverage`, default: brain_mask_atlasspace.nii.gz next to the map, otherwise
where the map is not 0). Output: a figure (png/pdf/svg, with colorbar), an interactive window that can be rotated with the mouse,
and a rotating video (mp4). Needs pyvista and, for the video, imageio-ffmpeg.

Mouse maps are in the space of the DevCCF P56 MRI template, which contains the Allen CCFv3 template; the Allen template
(allen_ccfv3_template_50um.nii.gz, made by tools/build_allen_atlas.py) is the default anatomy. Rat maps use the SIGMA template.

Run without arguments (`python plot_slice_stack.py`) to get a form that asks for the files and all settings.
Command line:
    python plot_slice_stack.py roi_atlas_mouse/atlas_space/stim_tstat_atlasspace.nii.gz stack.png --threshold 2 --label t
    python plot_slice_stack.py map.nii.gz vol.png --view volume --movie vol.mp4
(the anatomy is found automatically for maps made by the pipeline; otherwise give it with --underlay)
In a notebook:
    from plot_slice_stack import plot_slice_stack
    plot_slice_stack(map_file, "stack.png", threshold=2, label="t")
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from nibabel.processing import resample_from_to
from scipy import ndimage

# FSLeyes "blackbdy_iso" (the default map colours of deConfUSIon), 17 control points of the 256-entry table
BLACKBODY = LinearSegmentedColormap.from_list("blackbdy", [
    "#000000", "#2f0400", "#4b0000", "#640000", "#7e0000", "#980000", "#b40000", "#d10000", "#ec0f00",
    "#fc3c00", "#ff6600", "#ff8800", "#ffa300", "#ffbe00", "#ffd702", "#ffef45", "#fffcf2"])
NEG_CMAP = LinearSegmentedColormap.from_list("blackbdy_neg", ["#000000", "#001040", "#0030a0", "#0060ff", "#00b0ff", "#c0f4ff"])
VIEWS = ("stack", "volume", "surface")
# Slice direction: (slice axis, in-plane axis shown left -> right, in-plane axis shown bottom -> top) in RAS voxel axes
AXES = {"coronal": (1, 0, 2), "sagittal": (0, 1, 2), "axial": (2, 0, 1)}
# Volume / surface camera presets: direction from the brain to the camera in RAS (x right, y anterior, z dorsal)
CAMERA_PRESETS = {"oblique": (-0.6, -0.45, 0.7), "dorsal": (0.0, -0.05, 1.0), "coronal": (0.0, -1.0, 0.08), "sagittal": (-1.0, 0.0, 0.08),
                  "frontal oblique": (-0.55, 0.7, 0.45)}
STACK_VIEWS = {"paper oblique": (-6.0, 8.0), "shallow oblique": (-3.0, 4.0), "face-on": (0.0, 0.0)}
SETTINGS_FILE = Path(__file__).resolve().parent / "plot_slice_stack_last.json"


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as err:
        raise ImportError("This script needs pyvista: pip install pyvista imageio-ffmpeg") from err
    return pv


def _colormap(name):
    if not isinstance(name, str):
        return name
    if name.lower() in ("blackbdy", "blackbdy_iso", "blackbody"):
        return BLACKBODY
    return matplotlib.colormaps[name]


def _load(path, reference=None, order=1):
    img = nib.as_closest_canonical(nib.load(str(path)))
    if reference is not None and (img.shape[:3] != reference.shape[:3] or not np.allclose(img.affine, reference.affine, atol=1e-3)):
        img = resample_from_to(img, (reference.shape[:3], reference.affine), order=order, cval=0.0)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    return img, (data[..., 0] if data.ndim == 4 else data)


def _same_grid(a, b):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-3)


def find_structural(map_file):
    """(structural image, brain mask) if the map is in structural space, i.e. on the grid of a `cleaned_struct.nii.gz` of a sibling
    `<scan>anatomy` folder of the session (e.g. cleaned_signal_change_map_coregistered_structural_space.nii.gz). Several structural
    scans of one session with the same grid are told apart by the correlation of the aligned functional mean
    (`mean_func_struct_aligned.nii.gz`, next to the map) with them. ("", "") if the map is not in structural space."""
    map_path = Path(map_file).resolve()
    map_img = nib.load(str(map_path))
    candidates = []
    for folder in list(map_path.parents)[:4]:
        for struct in sorted(folder.glob("*anatomy*/cleaned_struct.nii.gz")):
            if _same_grid(nib.load(str(struct)), map_img):
                candidates.append(struct)
        if candidates:
            break
    if not candidates:
        return "", ""
    best = candidates[0]
    aligned = map_path.with_name("mean_func_struct_aligned.nii.gz")
    if len(candidates) > 1 and aligned.is_file():
        reference = np.asarray(nib.load(str(aligned)).get_fdata(dtype=np.float32)).ravel()
        scores = [np.corrcoef(reference, np.asarray(nib.load(str(c)).get_fdata(dtype=np.float32)).ravel())[0, 1] for c in candidates]
        best = candidates[int(np.nanargmax(scores))]
    mask = best.with_name("mask_struct.nii.gz")
    print(f"[INFO] Map in structural space: anatomy {best}")
    return str(best), str(mask) if mask.is_file() else ""


def find_anatomy(map_file):
    """(anatomy, brain mask) for a map made by the pipeline. A map in structural space gets the structural image of the animal
    (find_structural). A map in atlas space: the atlas folder of the analysis records the template it was registered to
    (atlas_used.json); the Allen template next to it is the anatomy. Falls back to template_registration_*.nii.gz."""
    structural = find_structural(map_file)
    if structural[0]:
        return structural
    for folder in list(Path(map_file).resolve().parents)[:4]:
        record = folder / "atlas_used.json"
        if record.is_file():
            template = Path(json.loads(record.read_text())["template"])
            allen = template.parent / "allen_ccfv3_template_50um.nii.gz"
            mask = template.parent / "devccf_brainmask_50um.nii.gz"
            if allen.is_file():
                return str(allen), str(mask) if mask.is_file() else ""
            return str(template), ""
    for folder in list(Path(map_file).resolve().parents)[:4]:
        found = sorted(folder.glob("template_registration_*mm.nii.gz"))
        if found:
            return str(found[0]), ""
    return "", ""


def _prepare(map_file, underlay_file, brain_mask, coverage, threshold, vmax, color_min, cmap, sign):
    """Loads the data on the grid of the anatomy (RAS) and computes the colour and opacity of the map (deConfUSIon transfer)."""
    under_img, under = _load(underlay_file)
    aff = under_img.affine
    if np.abs(aff[:3, :3] - np.diag(np.diag(aff[:3, :3]))).max() > 1e-4:
        raise ValueError("The anatomy must be in standard (non-oblique) orientation.")
    _, over = _load(map_file, reference=under_img)
    over = np.nan_to_num(over)
    map_spacing = np.abs(np.array(nib.as_closest_canonical(nib.load(str(map_file))).header.get_zooms()[:3], dtype=float))
    if brain_mask:
        brain = _load(brain_mask, reference=under_img, order=0)[1] > 0.5
    else:
        tissue = under[under > 0]
        brain = under > 0.05 * np.percentile(tissue, 99) if tissue.size else under > 0
        brain = ndimage.binary_fill_holes(ndimage.binary_opening(brain, iterations=2))
    if coverage is None:
        guess = Path(map_file).with_name("brain_mask_atlasspace.nii.gz")
        coverage = str(guess) if guess.is_file() else ""
    covered = (_load(coverage, reference=under_img, order=0)[1] > 0.5) if coverage else (over != 0)
    inside = brain & covered
    psc = np.where(inside, over, np.nan).astype(np.float32)

    tissue = under[brain]
    hi = np.percentile(tissue, 99.9) if tissue.size else 1.0
    anatomy = np.clip(under / max(hi, 1e-9), 0, 1).astype(np.float32)
    anatomy[~brain] = 0

    values = np.abs(psc[np.isfinite(psc)])
    shown = values[values > threshold]
    if vmax is None:
        vmax = float(np.percentile(shown, 99.5)) if shown.size else threshold + 1
    negative = sign != "positive" and bool((psc < -threshold).any())
    return SimpleNamespace(affine=aff, spacing=np.abs(np.diag(aff)[:3]), world=[aff[a, a] * np.arange(under.shape[a]) + aff[a, 3] for a in range(3)],
                           anatomy=anatomy, brain=brain, covered=inside, psc=psc, threshold=float(threshold), vmax=float(vmax),
                           color_min=float(0.0 if color_min is None else color_min), cmap=_colormap(cmap), sign=sign,
                           negative=negative, map_spacing=map_spacing)


def _regrid(c, factor_limit=2.0):
    """Interpolates (linearly) the strongly anisotropic direction of the data, i.e. the thick slices of an EPI acquisition (0.7 mm next
    to 0.047 mm in the plane), to about twice the finest voxel size, so that the map becomes a smooth 3D object instead of a pile of
    plates. The map is interpolated with 0 outside the measured region (values fade between slices) and cut off again where less
    than half of the interpolated voxel lies in the measured region."""
    axis = int(np.argmax(c.spacing))
    finest = float(c.spacing.min())
    factor = int(round(c.spacing[axis] / (factor_limit * finest)))
    if factor < 2:
        return c
    zoom = [float(factor) if a == axis else 1.0 for a in range(3)]

    def up(arr, order=1):
        return ndimage.zoom(arr.astype(np.float32), zoom, order=order, grid_mode=True, mode="nearest")

    covered = up(c.covered) > 0.5
    brain = up(c.brain) > 0.5
    psc = np.where(covered, up(np.nan_to_num(c.psc)), np.nan).astype(np.float32)
    anatomy = np.clip(up(c.anatomy), 0, 1)
    anatomy[~brain] = 0
    spacing = c.spacing.copy()
    spacing[axis] /= factor
    world = list(c.world)
    world[axis] = c.world[axis][0] + spacing[axis] * (np.arange(anatomy.shape[axis]) - (factor - 1) / 2)
    return SimpleNamespace(**{**vars(c), "anatomy": anatomy, "brain": brain, "covered": covered, "psc": psc, "spacing": spacing, "world": world})


def _transfer(c, values, alpha_max, strength, lift=True):
    """RGB and opacity of map values (fusiOverlayAppearance + the 3D settings of deConfUSIon): the opacity rises from the threshold
    to vmax, is lifted (square root) so that weak responses stay visible and multiplied by `strength`; the colour follows the
    colormap between color_min and vmax. NaN (outside the coverage) is transparent."""
    v = np.nan_to_num(values)
    if c.sign == "positive":
        show = v > 0
    elif c.sign == "negative":
        show = v < 0
    else:
        show = v != 0
    magnitude = np.abs(v)
    show &= (magnitude >= c.threshold) & np.isfinite(values)
    modulation = np.clip((magnitude - c.threshold) / max(c.vmax - c.threshold, 1e-9), 0, 1)
    if c.sign != "positive":
        modulation = 0.2 + 0.8 * modulation
    alpha = np.where(show, alpha_max * modulation, 0.0)
    if lift and alpha_max > 0:
        alpha = alpha_max * np.sqrt(alpha / alpha_max)
    alpha = np.minimum(alpha_max, alpha * strength)
    scale = np.clip((magnitude - c.color_min) / max(c.vmax - c.color_min, 1e-9), 0, 1)
    rgb = np.where((v >= 0)[..., None], c.cmap(scale)[..., :3], NEG_CMAP(scale)[..., :3])
    return rgb.astype(np.float32), alpha.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------- stack view
def _slabs(c, axis, n_tiles, slab_slices, slice_range, cover, reverse_order, projection="mean"):
    """Slab centres along the slice axis (atlas voxel indices) and the mean anatomy / map / brain outline of every slab."""
    sa = AXES[axis][0]
    support = c.brain if cover == "brain" else c.covered
    present = np.where(support.any(axis=tuple(a for a in range(3) if a != sa)))[0]
    if slice_range is not None:
        w = c.world[sa]
        present = np.where((w >= min(slice_range)) & (w <= max(slice_range)))[0]
    if present.size == 0:
        raise ValueError("Nothing to show: no brain voxel with data in the selected range.")
    half_lo, half_hi = (slab_slices - 1) // 2, slab_slices - 1 - (slab_slices - 1) // 2
    lo, hi = present.min() + half_lo, present.max() - half_hi
    if lo > hi:
        lo = hi = int(round((present.min() + present.max()) / 2))
    centres = np.unique(np.round(np.linspace(lo, hi, n_tiles)).astype(int))
    descending = sa != 0  # coronal: anterior first, axial: dorsal first, sagittal: left first
    if descending != bool(reverse_order):
        centres = centres[::-1]
    slabs = []
    for centre in centres:
        idx = np.arange(max(0, centre - half_lo), min(c.anatomy.shape[sa], centre + half_hi + 1))
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            block = c.psc.take(idx, axis=sa)
            if projection == "max":  # the value with the largest magnitude, so small or sparse responses are not averaged away
                pick = np.nanargmax(np.where(np.isfinite(block), np.abs(block), -1), axis=sa)
                psc = np.take_along_axis(block, np.expand_dims(pick, sa), axis=sa).squeeze(sa)
            else:
                psc = np.nanmean(block, axis=sa)
        slabs.append(SimpleNamespace(anatomy=c.anatomy.take(idx, axis=sa).mean(axis=sa), brain=c.brain.take(idx, axis=sa).any(axis=sa),
                                     psc=psc, position=float(c.world[sa][idx].mean()), thickness=len(idx) * c.spacing[sa]))
    return slabs


def _image_orientation(arr2d, axis):
    """2D slab (two remaining RAS axes in order) -> image with [row, col], row 0 at the top."""
    sa, ax_x, ax_y = AXES[axis]
    remaining = [a for a in range(3) if a != sa]
    img = arr2d if remaining.index(ax_x) == 0 else arr2d.T  # now [x, y]
    if axis == "sagittal":
        img = img[::-1]  # anterior on the left
    return img.T[::-1]  # [row = top -> bottom, col = left -> right]


def _pixel_faces(mask, x0, y0, sx, sy, z):
    """Quads of the pixels of `mask` (row 0 at the top) in the plane z. Returns points, faces (vtk layout), pixel indices."""
    rows, cols = np.nonzero(mask)
    xa, xb = x0 + cols * sx, x0 + (cols + 1) * sx
    ya, yb = y0 - rows * sy, y0 - (rows + 1) * sy
    zz = np.full(rows.size, z)
    pts = np.stack([np.stack([xa, ya, zz], 1), np.stack([xb, ya, zz], 1), np.stack([xb, yb, zz], 1), np.stack([xa, yb, zz], 1)], 1).reshape(-1, 3)
    faces = np.hstack([np.full((rows.size, 1), 4), np.arange(rows.size * 4).reshape(-1, 4)]).ravel()
    return pts, faces, (rows, cols)


def _wall_faces(mask, x0, y0, sx, sy, depth):
    """Side walls along the exact pixel edges of the outline (fusiVolumeSlabGeometry). Returns points, faces, pixel indices."""
    padded = np.pad(mask, 1)
    pts, idx_r, idx_c = [], [], []
    # (row offset, col offset of the neighbour) -> edge corners in (dx, dy) pixel units, dy measured downwards
    for (dr, dc), (a, b) in (((-1, 0), ((0, 0), (1, 0))), ((1, 0), ((0, 1), (1, 1))), ((0, -1), ((0, 0), (0, 1))), ((0, 1), ((1, 0), (1, 1)))):
        neighbour = padded[1 + dr:padded.shape[0] - 1 + dr, 1 + dc:padded.shape[1] - 1 + dc]
        rows, cols = np.nonzero(mask & ~neighbour)
        pa = np.stack([x0 + (cols + a[0]) * sx, y0 - (rows + a[1]) * sy], 1)
        pb = np.stack([x0 + (cols + b[0]) * sx, y0 - (rows + b[1]) * sy], 1)
        lo, hi = np.full((rows.size, 1), -depth / 2), np.full((rows.size, 1), depth / 2)
        pts.append(np.stack([np.hstack([pa, lo]), np.hstack([pb, lo]), np.hstack([pb, hi]), np.hstack([pa, hi])], 1).reshape(-1, 3))
        idx_r.append(rows)
        idx_c.append(cols)
    pts = np.vstack(pts)
    n = len(pts) // 4
    faces = np.hstack([np.full((n, 1), 4), np.arange(n * 4).reshape(-1, 4)]).ravel()
    return pts, faces, (np.concatenate(idx_r), np.concatenate(idx_c))


def _stack_layout(count, width, height, columns, gap, rise, aspect):
    """Tile origins (top-left corner, x right / y up) as in fusiVolumeStackLayout: a strip of tiles that rise to the right; more than
    six tiles are split into strips with the number of columns that makes the tiles largest."""
    if columns is None:
        columns = count
        if count > 6:
            best = 0
            for nc in range(1, min(10, count) + 1):
                nr = int(np.ceil(count / nc))
                span = (width * (1 + (nc - 1) * (1 + gap)), height * (nr * (1.2 + (nc - 1) * rise) - 0.2))
                score = min(aspect / span[0], 1 / span[1])
                if score > best:
                    best, columns = score, nc
    columns = max(1, min(int(columns), count))
    origins = []
    for k in range(count):
        col, row = k % columns, k // columns
        origins.append((col * width * (1 + gap), -(row * height * (1.2 + (columns - 1) * rise) - col * rise * height)))
    return origins, columns


def _build_stack(pv, c, axis, n_tiles, slab_slices, depth_gain, gap, rise, columns, slice_range, cover, reverse_order, alpha_max,
                 strength, upsample, aspect, projection):
    sa, ax_x, ax_y = AXES[axis]
    sx, sy = c.spacing[ax_x] / upsample, c.spacing[ax_y] / upsample
    slabs = _slabs(c, axis, n_tiles, slab_slices, slice_range, cover, reverse_order, projection)
    images = []
    for s in slabs:
        u, brain = _image_orientation(s.anatomy, axis), _image_orientation(s.brain.astype(np.float32), axis)
        p = _image_orientation(s.psc, axis)
        valid = np.isfinite(p)
        if upsample > 1:
            u = ndimage.zoom(u, upsample, order=1)
            brain = ndimage.zoom(brain, upsample, order=1)
            num, den = ndimage.zoom(np.where(valid, p, 0), upsample, order=1), ndimage.zoom(valid.astype(np.float32), upsample, order=1)
            p = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
        images.append((u, brain > 0.5, p))
    height = max(i[0].shape[0] for i in images) * sy
    width = max(i[0].shape[1] for i in images) * sx
    origins, ncols = _stack_layout(len(slabs), width, height, columns, gap, rise, aspect)
    meshes, labels = [], []
    for s, (u, brain, p), (ox, oy) in zip(slabs, images, origins):
        rgb, a = _transfer(c, p, alpha_max, strength)
        colour = np.repeat(u[..., None], 3, axis=-1) * (1 - a[..., None]) + rgb * a[..., None]
        depth = s.thickness * depth_gain
        for z in (-depth / 2, depth / 2):  # front and back face, both textured (fusiVolumeGUI draws both)
            pts, faces, (r, cc) = _pixel_faces(brain, ox, oy, sx, sy, z)
            mesh = pv.PolyData(pts, faces)
            mesh.cell_data["rgb"] = (np.clip(colour[r, cc], 0, 1) * 255).astype(np.uint8)
            meshes.append(mesh)
        pts, faces, (r, cc) = _wall_faces(brain, ox, oy, sx, sy, depth)
        walls = pv.PolyData(pts, faces)
        walls.cell_data["rgb"] = (np.repeat(np.clip(u[r, cc] * 0.7, 0, 1)[:, None], 3, axis=1) * 255).astype(np.uint8)
        meshes.append(walls)
        labels.append((s.position, s.thickness, (ox + width / 2, oy - height * 1.08, 0.0)))
    scale_anchor = (origins[ncols - 1][0] + width * 0.92, origins[ncols - 1][1] - height * 1.32)
    return SimpleNamespace(meshes=meshes, labels=labels, width=width, height=height, scale_anchor=scale_anchor, slabs=slabs)


# --------------------------------------------------------------------------------------------------------------- volume view
def _block_mean(arr, stride, nan=False):
    if stride == 1:
        return arr
    shape = [int(np.ceil(n / stride)) * stride for n in arr.shape]
    pad = [(0, s - n) for s, n in zip(shape, arr.shape)]
    a = np.pad(arr.astype(np.float32), pad, constant_values=np.nan if nan else 0)
    a = a.reshape(shape[0] // stride, stride, shape[1] // stride, stride, shape[2] // stride, stride)
    if nan:
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            return np.nanmean(a, axis=(1, 3, 5))
    return a.mean(axis=(1, 3, 5))


def _downsampled(c, max_dim, budget):
    """Block means with the smallest stride that keeps the grid under `budget` voxels (or at most `max_dim` voxels per axis if given)."""
    if max_dim:
        stride = max(1, int(np.ceil(max(c.anatomy.shape) / max_dim)))
    else:
        stride = max(1, int(np.ceil((c.anatomy.size / budget) ** (1 / 3))))
    anatomy = _block_mean(c.anatomy, stride)
    brain = _block_mean(c.brain, stride) > 0.5
    psc = _block_mean(c.psc, stride, nan=True)
    spacing = c.spacing * stride
    origin = np.array([c.world[a][0] for a in range(3)]) + (stride - 1) / 2 * c.spacing
    return anatomy, brain, psc, spacing, origin


def _build_volume(pv, c, atlas_opacity, alpha_max, strength, max_dim):
    """Volume rendering as in fusiVolumeRGBA ("overall" atlas opacity, clear PSC on), made of scalar volumes with opacity tables:
    the anatomy as a faint blue-gray haze (dimmed around strong responses) and the map, positive and negative values separately."""
    tissue, brain, psc, spacing, origin = _downsampled(c, max_dim, 8e6)
    weight = brain * np.maximum(0.15, tissue)
    # "typical" path through the brain (95th percentile of the paths along the three axes), as in fusiVolumeAutoAppearance: with the longest
    # path a thin slab or an image with a strong intensity fall-off would come out far too transparent
    paths = np.concatenate([weight.sum(axis=a).ravel() for a in range(3)])
    path = float(np.percentile(paths[paths > 0], 95)) if (paths > 0).any() else 1.0
    _, ap = _transfer(c, psc, alpha_max, strength)
    envelope = ndimage.gaussian_filter(ap, 1.0, mode="nearest")
    attenuation = 1 - 0.8 * np.minimum(1, envelope / max(alpha_max, 1e-9))
    anatomy = (np.where(brain, np.maximum(tissue, 1e-3), 0) * attenuation).astype(np.float32)

    def grid_of(values):
        grid = pv.ImageData(dimensions=values.shape, spacing=tuple(float(s) for s in spacing), origin=tuple(float(o) for o in origin))
        grid.point_data["v"] = np.ascontiguousarray(values, dtype=np.float32).ravel(order="F")
        return grid

    # anatomy: the opacity of a voxel follows its (undimmed) tissue value so that the total opacity along a typical path is atlas_opacity
    x = np.linspace(0, 1, 100)
    haze_opacity = np.where(x > 0, 1 - (1 - min(1.0, atlas_opacity)) ** (np.maximum(0.15, x) / max(path, 1e-9)), 0.0)
    tone = np.minimum(1, 1.5 * np.sqrt(x))
    haze_colors = np.stack([0.85 * tone, 0.90 * tone, 0.95 * tone], 1)
    volumes = [dict(grid=grid_of(anatomy), cmap=ListedColormap(haze_colors), clim=(0.0, 1.0), opacity=haze_opacity)]
    for sign, cmap in ((1, c.cmap), (-1, NEG_CMAP)):
        if sign == 1 and c.sign == "negative" or sign == -1 and c.sign == "positive":
            continue
        magnitude = np.nan_to_num(np.where(sign * psc > 0, np.abs(psc), 0))
        if not (magnitude >= c.threshold).any():
            continue
        values = np.linspace(c.color_min, c.vmax, 100)
        shown = values * sign if c.sign == "both" else values
        _, table = _transfer(c, np.array(shown), alpha_max, strength)
        table = np.where(values >= c.threshold, table, 0.0)
        volumes.append(dict(grid=grid_of(magnitude), cmap=cmap, clim=(c.color_min, c.vmax), opacity=table))
    return SimpleNamespace(volumes=volumes, spacing=spacing, bounds_brain=_brain_bounds(brain, spacing, origin))


def _brain_bounds(brain, spacing, origin, n_points=20000):
    """Points on the surface of the brain (for fitting the camera; the corners of the bounding box would fit too loosely)."""
    edge = brain & ~ndimage.binary_erosion(brain)
    idx = np.argwhere(edge if edge.any() else np.ones_like(brain))
    if len(idx) > n_points:
        idx = idx[np.random.default_rng(0).choice(len(idx), n_points, replace=False)]
    return idx * spacing + origin


def _add_volume(pl, vol):
    for item in vol.volumes:
        # opacity per voxel (as MATLAB's volshow): the unit distance of the opacity table is the voxel size. (A table of exactly 256
        # entries is misread by pyvista, hence 100.)
        pl.add_volume(item["grid"], scalars="v", cmap=item["cmap"], clim=item["clim"], opacity=list(item["opacity"]), show_scalar_bar=False,
                      mapper="gpu", opacity_unit_distance=float(vol.spacing.min()), shade=False)


# -------------------------------------------------------------------------------------------------------------- surface view
def _iso(pv, volume, level, spacing, origin, sigma, max_faces=60000):
    vol = np.pad(ndimage.gaussian_filter(volume.astype(np.float32), sigma) if sigma else volume.astype(np.float32), 1)
    grid = pv.ImageData(dimensions=vol.shape, spacing=tuple(float(s) for s in spacing), origin=tuple(float(o) for o in origin - spacing))
    grid.point_data["v"] = vol.ravel(order="F")
    surf = grid.contour([level], scalars="v", method="marching_cubes").extract_surface(algorithm=None).clean()
    if surf.n_cells == 0:
        return None
    if surf.n_cells > max_faces:
        surf = surf.decimate(1 - max_faces / surf.n_cells)
    return surf.smooth_taubin(n_iter=20, pass_band=0.1).compute_normals(split_vertices=False)


def _voxel_faces(pv, active, rgb, alpha, spacing, origin):
    """Exposed faces of the active voxels, coloured per voxel (fusiVolumeVoxelFaces)."""
    padded = np.pad(active, 1)
    pts, cols = [], []
    corner_sets = {0: [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], 1: [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
                   2: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]}
    for axis in range(3):
        for step in (-1, 1):
            shift = [slice(1, -1)] * 3
            shift[axis] = slice(1 + step, padded.shape[axis] - 1 + step)
            idx = np.argwhere(active & ~padded[tuple(shift)])
            if idx.size == 0:
                continue
            base = idx.astype(np.float32) - 0.5
            base[:, axis] += 1.0 if step > 0 else 0.0
            quad = np.stack([base + np.array(c_, dtype=np.float32) * (np.arange(3) != axis) for c_ in corner_sets[axis]], 1)
            pts.append((quad * spacing + origin).reshape(-1, 3))
            colour = np.concatenate([rgb[tuple(idx.T)], alpha[tuple(idx.T)][:, None]], 1)
            cols.append(colour)
    if not pts:
        return None
    pts = np.vstack(pts)
    n = len(pts) // 4
    mesh = pv.PolyData(pts, np.hstack([np.full((n, 1), 4), np.arange(n * 4).reshape(-1, 4)]).ravel())
    mesh.cell_data["rgba"] = (np.clip(np.vstack(cols), 0, 1) * 255).astype(np.uint8)
    return mesh


def _build_surface(pv, c, brain_opacity, alpha_max, strength, max_dim, surface_mode):
    tissue, brain, small_psc, small_spacing, small_origin = _downsampled(c, max_dim, 2.5e6)
    brain_surface = _iso(pv, brain, 0.5, small_spacing, small_origin, 1.0)
    # the map keeps the resolution of the data (not the coarser grid of the brain surface)
    psc, spacing, origin = c.psc, c.spacing, np.array([w[0] for w in c.world])
    rgb, ap = _transfer(c, psc, alpha_max, strength)
    items = []
    if surface_mode == "voxels":
        mesh = _voxel_faces(pv, ap > 0, rgb, ap, spacing, origin)
        if mesh is not None:
            items.append(("voxels", mesh))
    else:
        for sign in ((1,) if c.sign == "positive" else (-1,) if c.sign == "negative" else (1, -1)):
            v = np.nan_to_num(sign * psc)
            peak = v.max()
            if peak <= c.threshold:
                continue
            level = min(c.threshold + 0.35 * (c.vmax - c.threshold), (c.threshold + peak) / 2)
            surf = _iso(pv, v, level, spacing, origin, 0.6)
            if surf is not None:
                col, a = _transfer(c, np.array([sign * level]), alpha_max, strength)
                items.append(("contour", (surf, tuple(col[0]), float(a[0]))))
    return SimpleNamespace(brain=brain_surface, brain_opacity=brain_opacity, items=items,
                           bounds_brain=_brain_bounds(brain, small_spacing, small_origin))


# ------------------------------------------------------------------------------------------------------------------ figure
def plot_slice_stack(map_file, out_file=None, *, underlay_file=None, view="stack", axis="coronal", n_tiles=6, slab_slices=None,
                     depth_gain=3.0, tile_gap=0.15, tile_rise=0.2, columns=None, stack_view="paper oblique", yaw=None, tilt=None,
                     reverse_order=False, slice_range=None, cover="data", projection="mean", camera="oblique", azimuth=None, elevation=None,
                     threshold=0.0, vmax=None, color_min=None, cmap="blackbdy", sign="positive", alpha_max=1.0, strength=5.0,
                     atlas_opacity=0.8, brain_opacity=0.3, surface_mode="voxels", max_dim=None, upsample=None, brain_mask=None,
                     coverage=None, slice_labels=True, label_origin=0.0, scale_bar_um=500, label="", title="", colorbar=True,
                     background="black", dpi=300, figsize=(10, 6), interactive=False, view_file=None, movie_file=None,
                     movie_seconds=8.0, movie_fps=20, show=False, _screenshot_instead=None):
    """Draws the 3D figure and saves it to `out_file` (png/pdf/svg). Returns the matplotlib figure (None without `out_file`).

    map_file        map in atlas space (e.g. roi_atlas_mouse/atlas_space/stim_tstat_atlasspace.nii.gz)
    underlay_file   anatomy on the atlas grid; default: found from the analysis folder (mouse: Allen CCFv3 template)
    view            "stack", "volume" or "surface"
    axis            stack: slice direction "coronal" (default), "sagittal" or "axial"
    n_tiles         stack: number of slabs, evenly spread over the extent of the data (or `slice_range` in atlas mm)
    slab_slices     stack: slices averaged per slab (default: about 250 µm, i.e. 5 slices of the Allen template, 1 slice of a 0.7 mm
                    structural-space map)
    projection      stack: map over the slab thickness, "mean" (default, as deConfUSIon) or "max" (largest magnitude; better for
                    small or sparse responses, which the mean dilutes below the threshold)
    depth_gain      stack: the slab thickness is shown this many times thicker (display only)
    tile_gap, tile_rise  stack: space between the tiles (fraction of the width) and rise of each next tile (fraction of the height)
    columns         stack: tiles per strip (default: one strip up to 6 tiles, otherwise automatic)
    stack_view      stack camera: "paper oblique" (yaw -6°, tilt 8°), "shallow oblique", "face-on" or "custom" (`yaw`, `tilt`)
    camera          volume / surface camera: "oblique", "dorsal", "coronal", "sagittal", "frontal oblique" or "custom"
                    (`azimuth`, `elevation` in degrees: 0/0 = from the front, azimuth turns to the left, elevation from above)
    threshold       |values| below are not shown; vmax: end of the colour scale (default: 99.5th percentile)
    color_min       start of the colour scale (default 0; values below the threshold stay hidden)
    cmap            colormap of positive values (default "blackbdy", FSLeyes blackbdy_iso; any matplotlib name)
    sign            "positive" (default), "negative" or "both"
    alpha_max, strength  opacity of the map: rises from the threshold to vmax up to alpha_max, square-root lifted, times strength
    atlas_opacity   volume: opacity of the anatomy along a typical path through the brain
    brain_opacity   surface: opacity of the brain surface
    surface_mode    surface: "voxels" (unsmoothed voxels, exact colours) or "contour" (smooth surface at one level)
    max_dim         volume / surface: the anatomy is averaged down to at most this many voxels per axis (default: automatic, by the
                    number of voxels; the map itself is shown at its full resolution in the surface view)
    upsample        stack: in-plane enlargement of the tiles (default: 2, or 1 if the map has a voxel size of 0.06 mm or finer in the plane)
    coverage        image of the measured brain region in atlas space (default: brain_mask_atlasspace.nii.gz next to the map)
    label_origin    mm subtracted from the slab positions in the labels (e.g. the position of bregma in the atlas)
    interactive     open a window that can be rotated with the mouse (s = save the view as PNG, q = close)
    movie_file      write a video (.mp4): the stack sways to both sides, volume and surface turn around once
    """
    pv = _pyvista()
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    if underlay_file is None:
        underlay_file, guessed_mask = find_anatomy(map_file)
        if not underlay_file:
            raise ValueError("No anatomy found for this map; give it with underlay_file / --underlay.")
        brain_mask = brain_mask or guessed_mask or None
    c = _prepare(map_file, underlay_file, brain_mask, coverage, threshold, vmax, color_min, cmap, sign)
    bg = "black" if background == "transparent" else background
    text_color = "black" if matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(bg))[2] > 0.5 else "white"
    if c.negative:
        bar_cmap = ListedColormap(np.vstack([NEG_CMAP(np.linspace(1, 0, 256)), c.cmap(np.linspace(0, 1, 256))]))
        bar_limits = (-c.vmax, c.vmax)
    else:
        bar_cmap, bar_limits = (NEG_CMAP, (-c.vmax, -c.color_min)) if sign == "negative" else (c.cmap, (c.color_min, c.vmax))
    aspect = figsize[0] / figsize[1]

    if view == "stack":
        slice_step = float(c.spacing[AXES[axis][0]])
        slab_slices = max(1, int(round(0.25 / slice_step))) if slab_slices is None else int(slab_slices)
        if upsample is None:
            in_plane = [c.map_spacing[a] for a in range(3) if a != AXES[axis][0]]
            upsample = 1 if max(in_plane) <= 0.06 else 2
        scene = _build_stack(pv, c, axis, int(n_tiles), slab_slices, depth_gain, tile_gap, tile_rise, columns, slice_range, cover,
                             reverse_order, alpha_max, strength, int(upsample), aspect, projection)
        if stack_view != "custom":
            yaw, tilt = STACK_VIEWS[stack_view]
        yaw, tilt = float(yaw or 0.0), float(tilt or 0.0)
        pts = np.vstack([m.points for m in scene.meshes])
        up = np.array([0.0, 1.0, 0.0])
    else:
        cf = _regrid(c)
        built = _build_volume(pv, cf, atlas_opacity, alpha_max, strength, max_dim) if view == "volume" else \
            _build_surface(pv, cf, brain_opacity, alpha_max, strength, max_dim, surface_mode)
        pts = built.bounds_brain
        up = np.array([0.0, 0.0, 1.0])
    centre = (pts.min(0) + pts.max(0)) / 2
    radius = float(np.linalg.norm(pts.max(0) - pts.min(0)) / 2)

    def direction_for(turn):
        """Unit vector from the scene to the camera; `turn` (degrees) rotates it about the vertical axis (movie)."""
        if view == "stack":
            y, t = np.radians(yaw + turn), np.radians(tilt)
            return np.array([np.sin(y) * np.cos(t), np.sin(t), np.cos(y) * np.cos(t)])  # camera on the +z side: +x right, +y up
        if camera == "custom":
            az, el = np.radians(float(azimuth or 0.0) + turn), np.radians(float(elevation or 0.0))
            d = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
        else:
            d = np.array(CAMERA_PRESETS[camera], dtype=float)
            if turn:
                r = np.radians(turn)
                d = np.array([d[0] * np.cos(r) - d[1] * np.sin(r), d[0] * np.sin(r) + d[1] * np.cos(r), d[2]])
        return d / np.linalg.norm(d)

    def screen_axes(d):
        right = np.cross(-d, up)
        if np.linalg.norm(right) < 1e-6:  # looking along the up direction
            right = np.cross(-d, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        return right, np.cross(right, -d)

    def fit(turns, aspect_ratio):
        half_w = half_h = 0.0
        for t in turns:
            right, scr_up = screen_axes(direction_for(t))
            x, y = (pts - centre) @ right, (pts - centre) @ scr_up
            half_w = max(half_w, np.abs(x).max())
            half_h = max(half_h, np.abs(y).max())
        return max(half_h, half_w / aspect_ratio) * 1.06

    def set_camera(pl, turn, scale, bar_margin=0.0):
        """bar_margin: share of the window width kept free on the right for the colorbar of the 3D window."""
        d = direction_for(turn)
        right, scr_up = screen_axes(d)
        width = 2 * scale * pl.window_size[0] / pl.window_size[1]
        focal = centre + right * bar_margin / 2 * width
        pl.camera.parallel_projection = True
        pl.camera.focal_point = tuple(focal)
        pl.camera.position = tuple(focal + d * 4 * radius)
        pl.camera.up = tuple(scr_up)
        pl.camera.parallel_scale = scale
        pl.camera.clipping_range = (0.5 * radius, 8 * radius)

    def build_plotter(off_screen, size):
        pl = pv.Plotter(off_screen=off_screen, window_size=list(size), lighting="none")
        pl.set_background(bg)
        if view == "stack":
            for mesh in scene.meshes:
                pl.add_mesh(mesh, scalars="rgb", rgb=True, lighting=False, show_scalar_bar=False, preference="cell")
            if scale_bar_um:
                x1, y1 = scene.scale_anchor
                bar = pv.Line((x1 - scale_bar_um / 1000, y1, 0), (x1, y1, 0))
                pl.add_mesh(bar, color=text_color, line_width=3, lighting=False)
        elif view == "volume":
            _add_volume(pl, built)
        else:
            pl.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
            if built.brain is not None:
                pl.add_mesh(built.brain, color=(0.75, 0.78, 0.82), opacity=built.brain_opacity, smooth_shading=True, ambient=0.3,
                            diffuse=0.7, specular=0.1, show_scalar_bar=False)
            for kind, item in built.items:
                if kind == "voxels":
                    pl.add_mesh(item, scalars="rgba", rgba=True, lighting=False, show_scalar_bar=False, preference="cell")
                else:
                    surf, colour, opacity = item
                    pl.add_mesh(surf, color=colour, opacity=opacity, smooth_shading=True, ambient=0.35, diffuse=0.75, show_scalar_bar=False)
            pl.add_light(pv.Light(light_type="headlight", intensity=1.0))
        return pl

    def label_text(position, thickness):
        return f"{position - label_origin:+.2f} mm | {thickness * 1000:.0f} µm"

    fig = None
    if out_file is not None or not (interactive or movie_file):
        window = int(np.clip(max(figsize) * dpi, 800, 3000))
        size = (window, int(window / aspect)) if aspect >= 1 else (int(window * aspect), window)
        scale = fit([0.0], size[0] / size[1])
        pl = build_plotter(True, size)
        set_camera(pl, 0.0, scale)
        if view == "stack":
            pl.enable_anti_aliasing("ssaa")
        image = pl.screenshot(return_img=True, transparent_background=background == "transparent")
        pl.close()
        px_per_mm = size[1] / (2 * scale)
        right, scr_up = screen_axes(direction_for(0.0))

        def to_pixels(p):
            v = np.asarray(p) - centre
            return size[0] / 2 + (v @ right) * px_per_mm, size[1] / 2 - (v @ scr_up) * px_per_mm

        texts = []
        if view == "stack" and slice_labels:
            texts += [(label_text(pos, th), *to_pixels(anchor), "center") for pos, th, anchor in scene.labels]
        if view == "stack" and scale_bar_um:
            x1, y1 = scene.scale_anchor
            texts.append((f"{scale_bar_um:g} µm", *to_pixels((x1, y1 + scene.height * 0.1, 0)), "right"))
        # crop to the content
        background_rgb = np.array(matplotlib.colors.to_rgb(bg))
        content = image[..., 3] > 8 if image.shape[2] == 4 else np.any(np.abs(image[..., :3] / 255.0 - background_rgb) > 0.03, axis=2)
        for _, px, py, _ in texts:
            content[int(np.clip(py, 0, size[1] - 1)), int(np.clip(px, 0, size[0] - 1))] = True
        rows, cols = np.where(content)
        margin = int(0.03 * max(image.shape[:2]))
        y0, y1_ = max(rows.min() - margin, 0), min(rows.max() + margin, image.shape[0])
        x0, x1_ = max(cols.min() - margin, 0), min(cols.max() + margin, image.shape[1])
        cropped = image[y0:y1_, x0:x1_]

        width_in = figsize[0] * (0.86 if colorbar else 0.98)
        height_in = width_in * cropped.shape[0] / cropped.shape[1]
        top = 0.45 if title else 0.05
        fig_w, fig_h = figsize[0], height_in + top + 0.05
        face = "none" if background == "transparent" else bg
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=face)
        ax = fig.add_axes([0.01, 0.05 / fig_h, width_in / fig_w, height_in / fig_h])
        ax.imshow(cropped, interpolation="lanczos", extent=(0, cropped.shape[1], cropped.shape[0], 0))
        ax.axis("off")
        for text, px, py, ha in texts:
            ax.text(px - x0, py - y0, text, ha=ha, va="center", fontsize=7, color=text_color)
        if title:
            fig.text(0.5, 1 - 0.1 / fig_h, title, ha="center", va="top", fontsize=11, color=text_color)
        if colorbar:
            cax = fig.add_axes([0.905, 0.05 / fig_h + 0.25 * height_in / fig_h, 0.016, 0.5 * height_in / fig_h])
            cbar = fig.colorbar(ScalarMappable(Normalize(*bar_limits), bar_cmap), cax=cax)
            cbar.set_label(label, color=text_color)
            cbar.ax.tick_params(colors=text_color, labelsize=8)
            cbar.outline.set_edgecolor(text_color)
        if out_file:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_file, dpi=dpi, facecolor=face, transparent=background == "transparent")
            print(f"[OK] Figure saved: {out_file}  ({view})")

    if movie_file:
        movie_size = (1280, 720)
        frames = max(int(movie_seconds * movie_fps), 2)
        steps = np.arange(frames) / frames
        turns = 15.0 * np.sin(2 * np.pi * steps) if view == "stack" else 360.0 * steps
        scale = fit(np.linspace(turns.min(), turns.max(), 13), 0.84 * movie_size[0] / movie_size[1])
        pl = build_plotter(True, movie_size)
        if view == "stack":
            pl.enable_anti_aliasing("fxaa")
        _add_window_decorations(pl, pv, bar_cmap, bar_limits, label, title, text_color, centre)
        Path(movie_file).parent.mkdir(parents=True, exist_ok=True)
        pl.open_movie(str(movie_file), framerate=int(movie_fps), quality=8)
        for turn in turns:
            set_camera(pl, float(turn), scale, bar_margin=0.16)
            pl.write_frame()
        pl.close()
        print(f"[OK] Movie saved: {movie_file}  ({frames} frames)")

    if interactive:
        save_to = Path(view_file) if view_file else (Path(out_file).with_name(Path(out_file).stem + "_view.png") if out_file else Path("brain3d_view.png"))
        size = (1400, 900)
        pl = build_plotter(_screenshot_instead is not None, size)
        _add_window_decorations(pl, pv, bar_cmap, bar_limits, label, title, text_color, centre)
        if view == "stack" and slice_labels:
            pl.add_point_labels(np.array([a for _, _, a in scene.labels]), [label_text(p, t) for p, t, _ in scene.labels], font_size=11,
                                text_color=text_color, shape=None, show_points=False, always_visible=True)
        set_camera(pl, 0.0, fit([0.0], 0.84 * size[0] / size[1]), bar_margin=0.16)
        pl.add_text("drag: rotate   scroll: zoom   shift+drag: move   s: save view as PNG   q: close", position="upper_left", font_size=7,
                    color=text_color)

        def save_view():
            save_to.parent.mkdir(parents=True, exist_ok=True)
            pl.screenshot(str(save_to), transparent_background=background == "transparent")
            print(f"[OK] View saved: {save_to}")

        pl.add_key_event("s", save_view)
        if _screenshot_instead is not None:
            pl.screenshot(str(_screenshot_instead))
            pl.close()
        else:
            print("Interactive 3D viewer: drag to rotate, scroll to zoom, shift+drag to move, 's' saves the current view, 'q' closes it.")
            pl.show(title="Brain 3D")
    if show:
        plt.show()
    return fig


def _add_window_decorations(pl, pv, bar_cmap, bar_limits, label, title, text_color, anchor):
    """Colorbar and title inside the 3D window (an invisible point carries the colour scale)."""
    marker = pv.PolyData(np.array([anchor]))
    marker["scale"] = np.array([bar_limits[0]])
    pl.add_mesh(marker, scalars="scale", cmap=bar_cmap, clim=bar_limits, opacity=0.0, show_scalar_bar=True,
                scalar_bar_args={"title": label or " ", "vertical": True, "position_x": 0.9, "position_y": 0.25, "height": 0.5, "width": 0.04,
                                 "color": text_color, "n_labels": 5, "fmt": "%.2g"})
    if title:
        pl.add_text(title, position="upper_edge", font_size=12, color=text_color)


# --------------------------------------------------------------------------------------------------------------------- form
def _open_file(path):
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    elif sys.platform.startswith("win"):
        os.startfile(str(path))
    else:
        subprocess.run(["xdg-open", str(path)])


# (key, label, default, kind, choices): "file" fields get a Browse button, "choice" a drop-down, "check" a checkbox, "section" a heading
GUI_FIELDS = [
    ("map", "Map in atlas space (.nii/.nii.gz)", "", "file", None),
    ("underlay", "Anatomy (empty = automatic: Allen template for mouse)", "", "file", None),
    ("out", "Output image (.png/.pdf/.svg)", "", "save", None),
    ("view", "View", "stack", "choice", list(VIEWS)),
    ("_s1", "Slice stack", "", "section", None),
    ("axis", "Slice direction", "coronal", "choice", list(AXES)),
    ("n_tiles", "Number of slabs", "6", "text", None),
    ("slab_slices", "Slices per slab (empty = about 250 µm)", "", "text", None),
    ("depth_gain", "Show the slabs this many times thicker", "3", "text", None),
    ("tile_gap", "Gap between slabs (fraction of width)", "0.15", "text", None),
    ("tile_rise", "Rise of each next slab (fraction of height)", "0.2", "text", None),
    ("columns", "Slabs per strip (empty = automatic)", "", "text", None),
    ("stack_view", "Camera", "paper oblique", "choice", list(STACK_VIEWS) + ["custom"]),
    ("yaw", "Custom camera: yaw (deg)", "-6", "text", None),
    ("tilt", "Custom camera: tilt (deg)", "8", "text", None),
    ("projection", "Map over the slab (max for small or sparse effects)", "mean", "choice", ["mean", "max"]),
    ("cover", "Slabs cover", "data", "choice", ["data", "brain"]),
    ("range_start", "Range start (atlas mm, empty = automatic)", "", "text", None),
    ("range_end", "Range end (atlas mm, empty = automatic)", "", "text", None),
    ("reverse_order", "Reverse the order of the slabs", False, "check", None),
    ("slice_labels", "Show slab position labels", True, "check", None),
    ("label_origin", "Labels: subtract (mm, e.g. bregma)", "0", "text", None),
    ("scale_bar_um", "Scale bar (µm, 0 = none)", "500", "text", None),
    ("_s2", "Volume / surface", "", "section", None),
    ("camera", "Camera", "oblique", "choice", list(CAMERA_PRESETS) + ["custom"]),
    ("azimuth", "Custom camera: azimuth (deg)", "-35", "text", None),
    ("elevation", "Custom camera: elevation (deg)", "35", "text", None),
    ("atlas_opacity", "Volume: anatomy opacity (0-1)", "0.8", "text", None),
    ("brain_opacity", "Surface: brain opacity (0-1)", "0.3", "text", None),
    ("surface_mode", "Surface: map as", "voxels", "choice", ["voxels", "contour"]),
    ("_s3", "Map", "", "section", None),
    ("threshold", "Threshold (|values| below are hidden)", "0", "text", None),
    ("vmax", "Colour scale maximum (empty = automatic)", "", "text", None),
    ("color_min", "Colour scale minimum (empty = 0)", "", "text", None),
    ("cmap", "Colormap (blackbdy or a matplotlib name)", "blackbdy", "text", None),
    ("sign", "Values shown", "positive", "choice", ["positive", "negative", "both"]),
    ("strength", "Map opacity strength", "5", "text", None),
    ("alpha_max", "Map opacity maximum (0-1)", "1", "text", None),
    ("brain_mask", "Brain mask (empty = automatic)", "", "file", None),
    ("coverage", "Measured region (empty = brain_mask_atlasspace next to the map)", "", "file", None),
    ("label", "Colorbar label", "", "text", None),
    ("title", "Title", "", "text", None),
    ("colorbar", "Show colorbar", True, "check", None),
    ("background", "Background", "black", "choice", ["black", "white", "transparent"]),
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

    for key, name in (("map", "map"), ("out", "output file")):
        if not str(v[key]).strip():
            raise ValueError(f"Please select the {name}.")
    for key in ("map", "underlay", "brain_mask", "coverage"):
        if str(v[key]).strip() and not Path(v[key]).is_file():
            raise ValueError(f"File not found: {v[key]}")
    start, end = number("range_start", optional=True), number("range_end", optional=True)
    if (start is None) != (end is None):
        raise ValueError("Enter both range start and range end, or leave both empty.")
    out = Path(v["out"])
    return dict(
        map_file=v["map"], out_file=v["out"], underlay_file=v["underlay"].strip() or None, view=v["view"], axis=v["axis"],
        n_tiles=number("n_tiles", int), slab_slices=number("slab_slices", int, optional=True), depth_gain=number("depth_gain"),
        tile_gap=number("tile_gap"), tile_rise=number("tile_rise"), columns=number("columns", int, optional=True),
        stack_view=v["stack_view"], yaw=number("yaw"), tilt=number("tilt"), cover=v["cover"], projection=v["projection"],
        slice_range=None if start is None else (start, end), reverse_order=bool(v["reverse_order"]), slice_labels=bool(v["slice_labels"]),
        label_origin=number("label_origin"), scale_bar_um=number("scale_bar_um"), camera=v["camera"], azimuth=number("azimuth"),
        elevation=number("elevation"), atlas_opacity=number("atlas_opacity"), brain_opacity=number("brain_opacity"),
        surface_mode=v["surface_mode"], threshold=number("threshold"), vmax=number("vmax", optional=True),
        color_min=number("color_min", optional=True), cmap=v["cmap"].strip() or "blackbdy", sign=v["sign"], strength=number("strength"),
        alpha_max=number("alpha_max"), brain_mask=v["brain_mask"].strip() or None, coverage=v["coverage"].strip() or None,
        label=v["label"], title=v["title"], colorbar=bool(v["colorbar"]), background=v["background"], dpi=number("dpi", int),
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
        if kind == "choice" and value not in choices:
            value = default
        variables[key] = tk.StringVar(value=str(value))
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=1)
        if kind == "choice":
            ttk.Combobox(frame, textvariable=variables[key], values=choices, state="readonly", width=16).grid(row=row, column=1, sticky="w")
        else:
            ttk.Entry(frame, textvariable=variables[key], width=70 if kind in ("file", "save") else 16).grid(row=row, column=1, sticky="w")
            if kind in ("file", "save"):
                ttk.Button(frame, text="Browse...", command=lambda k=key, t=kind: browse(k, t)).grid(row=row, column=2, padx=6)

    def on_map_changed(*_):
        map_file = variables["map"].get().strip()
        if map_file and Path(map_file).is_file() and not variables["out"].get().strip():
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
        plot_slice_stack(kwargs.pop("map_file"), kwargs.pop("out_file", None), **kwargs)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map", help="map in atlas space (NIfTI)")
    p.add_argument("out", nargs="?", help="output image (.png, .pdf, .svg); optional with --interactive or --movie")
    p.add_argument("--underlay", help="anatomy on the atlas grid (default: found from the analysis folder)")
    p.add_argument("--view", choices=VIEWS, default="stack")
    p.add_argument("--axis", choices=list(AXES), default="coronal")
    p.add_argument("--n-tiles", type=int, default=6)
    p.add_argument("--slab-slices", type=int, help="slices averaged per slab (default: about 250 µm)")
    p.add_argument("--depth-gain", type=float, default=3.0)
    p.add_argument("--tile-gap", type=float, default=0.15)
    p.add_argument("--tile-rise", type=float, default=0.2)
    p.add_argument("--columns", type=int)
    p.add_argument("--stack-view", choices=list(STACK_VIEWS) + ["custom"], default="paper oblique")
    p.add_argument("--yaw", type=float)
    p.add_argument("--tilt", type=float)
    p.add_argument("--reverse-order", action="store_true")
    p.add_argument("--range", nargs=2, type=float, metavar=("START", "END"), dest="slice_range", help="slab range in atlas mm")
    p.add_argument("--cover", choices=("data", "brain"), default="data")
    p.add_argument("--projection", choices=("mean", "max"), default="mean", help="map over the slab thickness (max for sparse maps)")
    p.add_argument("--camera", choices=list(CAMERA_PRESETS) + ["custom"], default="oblique")
    p.add_argument("--azimuth", type=float)
    p.add_argument("--elevation", type=float)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--vmax", type=float)
    p.add_argument("--color-min", type=float)
    p.add_argument("--cmap", default="blackbdy")
    p.add_argument("--sign", choices=("positive", "negative", "both"), default="positive")
    p.add_argument("--strength", type=float, default=5.0)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--atlas-opacity", type=float, default=0.8)
    p.add_argument("--brain-opacity", type=float, default=0.3)
    p.add_argument("--surface-mode", choices=("voxels", "contour"), default="voxels")
    p.add_argument("--max-dim", type=int, help="volume / surface: voxels per axis of the anatomy (default: automatic)")
    p.add_argument("--upsample", type=int, help="stack: in-plane enlargement of the tiles (default: automatic)")
    p.add_argument("--brain-mask")
    p.add_argument("--coverage")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--label-origin", type=float, default=0.0)
    p.add_argument("--scale-bar-um", type=float, default=500)
    p.add_argument("--label", default="", help="colorbar label")
    p.add_argument("--title", default="")
    p.add_argument("--no-colorbar", action="store_true")
    p.add_argument("--background", default="black", help="colour or 'transparent'")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--interactive", action="store_true", help="open an interactive 3D viewer (s = save the view)")
    p.add_argument("--movie", metavar="FILE.mp4")
    p.add_argument("--movie-seconds", type=float, default=8.0)
    a = p.parse_args()
    if not a.out and not a.interactive and not a.movie:
        p.error("an output file is required (or use --interactive / --movie)")
    plot_slice_stack(a.map, a.out, underlay_file=a.underlay, view=a.view, axis=a.axis, n_tiles=a.n_tiles, slab_slices=a.slab_slices,
                     depth_gain=a.depth_gain, tile_gap=a.tile_gap, tile_rise=a.tile_rise, columns=a.columns, stack_view=a.stack_view,
                     yaw=a.yaw, tilt=a.tilt, reverse_order=a.reverse_order, slice_range=a.slice_range, cover=a.cover,
                     projection=a.projection, camera=a.camera,
                     azimuth=a.azimuth, elevation=a.elevation, threshold=a.threshold, vmax=a.vmax, color_min=a.color_min, cmap=a.cmap,
                     sign=a.sign, strength=a.strength, alpha_max=a.alpha_max, atlas_opacity=a.atlas_opacity, brain_opacity=a.brain_opacity,
                     surface_mode=a.surface_mode, max_dim=a.max_dim, upsample=a.upsample, brain_mask=a.brain_mask, coverage=a.coverage,
                     slice_labels=not a.no_labels, label_origin=a.label_origin, scale_bar_um=a.scale_bar_um, label=a.label, title=a.title,
                     colorbar=not a.no_colorbar, background=a.background, dpi=a.dpi, interactive=a.interactive, movie_file=a.movie,
                     movie_seconds=a.movie_seconds)


if __name__ == "__main__":
    main()
