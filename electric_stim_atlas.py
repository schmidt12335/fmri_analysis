"""Atlas registration helpers for the electrical stimulation analysis notebooks.

Follows roi_atlas_analysis.py: the cleaned structural scan is registered to the brain-masked functional mean (struct2func,
rigid, bounded like the coregistration of mol_fMRI_v5.6) and to the atlas template (struct2atlas). The two matrices are
composed into one functional -> atlas matrix, and the multi-label
atlas is resampled once onto the functional grid with nearest-neighbour interpolation, so the functional data itself is
never interpolated. Finished registrations in atlas_dir are reused.
"""
import re
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from electric_stim_helpers import bcolors, print_statement


def species_atlas_config(fmri_root):
    """Atlas files and registration settings per species. Rat: SIGMA in vivo. Mouse: DSURQE (40 µm, ex vivo T2w)."""
    atlas_root = f"{fmri_root}/atlas"
    sigma_dir = (f"{atlas_root}/SIGMA_rat_atlas/sigma_wistar_rat_brain_templatesandatlases_version1-2-1_2022-08-26/"
                 "SIGMA_Wistar_Rat_Brain_TemplatesAndAtlases_Version1.2.1")
    sigma_atlas_dir = f"{sigma_dir}/SIGMA_Rat_Brain_Atlases/SIGMA_Anatomical_Atlas/InVivo_Atlas"
    return {
        "rat": {
            "template": f"{sigma_dir}/SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_Brain_Template_Masked.nii",
            "labels": f"{sigma_atlas_dir}/SIGMA_InVivo_Anatomical_Brain_Atlas.nii",
            "mask": "", "names": f"{sigma_atlas_dir}/SIGMA_InVivo_Anatomical_Brain_Atlas_ListOfStructures.csv",
            "warp": "shift_rotate", "cost": "lpa+ZZ", "reg_voxel_mm": None,
        },
        "mouse": {
            "template": f"{atlas_root}/DSURQE_mouse_atlas/DSURQE_40micron_average.nii",
            "labels": f"{atlas_root}/DSURQE_mouse_atlas/DSURQE_40micron_labels.nii",
            "mask": f"{atlas_root}/DSURQE_mouse_atlas/DSURQE_40micron_mask.nii",
            "names": f"{atlas_root}/DSURQE_mouse_atlas/dsurqe_labels.csv",
            # Scaling absorbs the size difference between in vivo brains and the ex vivo template
            "warp": "shift_rotate_scale", "cost": "nmi", "reg_voxel_mm": 0.1,
        },
    }


def run_afni(cmd, capture=False):
    cmd = [str(c) for c in cmd]
    print("[COMMAND]", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def prepare_registration_template(atlas_template, atlas_mask, reg_voxel_mm, atlas_dir):
    """Brain-mask and/or downsample the template so registration is robust and fast."""
    template = Path(atlas_template)
    if atlas_mask:
        masked = Path(atlas_dir) / "template_masked.nii.gz"
        if not masked.exists():
            run_afni(["3dcalc", "-a", template, "-b", atlas_mask, "-expr", "a*step(b)", "-prefix", masked])
        template = masked
    if reg_voxel_mm:
        coarse = Path(atlas_dir) / f"template_registration_{reg_voxel_mm}mm.nii.gz"
        if not coarse.exists():
            run_afni(["3dresample", "-dxyz", reg_voxel_mm, reg_voxel_mm, reg_voxel_mm, "-rmode", "Cu",
                      "-input", template, "-prefix", coarse])
        template = coarse
    return template


def register(source, base, prefix, cost, cmass, warp, atlas_dir, extra_args=()):
    matrix = Path(atlas_dir) / f"{prefix}.aff12.1D"
    aligned = Path(atlas_dir) / f"{prefix}_aligned.nii.gz"
    if matrix.exists() and aligned.exists():
        print_statement(f"{prefix} registration already exists. Skipping.", bcolors.OKGREEN)
        return matrix
    cmd = ["3dAllineate", "-source", source, "-base", base, "-prefix", aligned, "-1Dmatrix_save", matrix,
           "-1Dparam_save", Path(atlas_dir) / f"{prefix}_params.1D", "-warp", warp, "-cost", cost,
           "-source_automask+2", "-twopass", "-twobest", "MAX", "-float", "-verb", "-overwrite", *extra_args]
    if cmass:
        cmd.append(f"-cmass{cmass}")
    run_afni(cmd)
    return matrix


def aff12_to_matvec(matrix, out):
    vals = [float(v) for line in Path(matrix).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#") for v in line.split()]
    if len(vals) != 12:
        raise ValueError(f"Expected 12 values in {matrix}, found {len(vals)}.")
    Path(out).write_text("\n".join(" ".join(f"{v:.12g}" for v in vals[i:i + 4]) for i in (0, 4, 8)) + "\n")
    return out


STRUCT2FUNC_MAX_ROT_DEG, STRUCT2FUNC_MAX_SHIFT_MM = 5.0, 2.0  # same bounds as the coregistration of mol_fMRI_v5.6
STRUCT2FUNC_LIMITS = ("-maxrot", str(STRUCT2FUNC_MAX_ROT_DEG), "-maxshf", str(STRUCT2FUNC_MAX_SHIFT_MM))


AFNI_MIN_BASE_DIM = 9  # 3dAllineate refuses a base image with fewer voxels along the first two axes


def pad_for_allineate(image_file, padded_file):
    """Copy of `image_file` with empty slices added along the first two axes that are too short for 3dAllineate (e.g. a
    slab of 8 slices). Voxel positions in mm do not change, so a matrix estimated on the padded image also applies to
    the original grid. Returns `image_file` itself if nothing has to be padded."""
    img = nib.load(str(image_file))
    shape = img.shape
    pad = [max(0, int(np.ceil((AFNI_MIN_BASE_DIM - shape[axis]) / 2))) if axis < 2 and shape[axis] < AFNI_MIN_BASE_DIM else 0
           for axis in range(3)]
    if not any(pad):
        return image_file
    data = np.pad(img.get_fdata(dtype=np.float32), [(p, p) for p in pad] + [(0, 0)] * (img.ndim - 3))
    affine = img.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ np.array(pad, dtype=float)
    nib.save(nib.Nifti1Image(data, affine), str(padded_file))
    print(f"[INFO] {Path(image_file).name} has only {shape[:3]} voxels; padded with empty slices to {data.shape[:3]} for the registration")
    return padded_file


def struct2func_matrix(struct_file, func_reference, brain_mask, atlas_dir):
    """Registers the cleaned structural scan to the brain-masked functional mean (rigid, bounded like mol_fMRI_v5.6).
    Cached under its own name, so results of the earlier unmasked registration (`struct2func*`) are not reused."""
    atlas_dir = Path(atlas_dir)
    masked = atlas_dir / "mean_mc_func_masked.nii.gz"
    if not masked.exists() or masked.stat().st_mtime < max(Path(func_reference).stat().st_mtime, Path(brain_mask).stat().st_mtime):
        run_afni(["3dcalc", "-a", func_reference, "-b", brain_mask, "-expr", "a*step(b)", "-prefix", masked, "-overwrite"])
    base = pad_for_allineate(masked, atlas_dir / "mean_mc_func_masked_padded.nii.gz")
    return register(struct_file, base, "struct2func_masked", "nmi", None, "shift_rotate", atlas_dir, STRUCT2FUNC_LIMITS)


def register_to_atlas(struct_file, func_reference, brain_mask, atlas_dir, species_cfg, atlas_template, atlas_labels, atlas_mask=""):
    """Registers structural -> functional and structural -> atlas, composes the functional -> atlas matrix and
    resamples the atlas labels onto the functional grid. Returns (composite matrix, labels in functional space)."""
    atlas_dir = Path(atlas_dir)
    struct2func = struct2func_matrix(struct_file, func_reference, brain_mask, atlas_dir)
    struct2atlas = register(struct_file, prepare_registration_template(atlas_template, atlas_mask, species_cfg["reg_voxel_mm"], atlas_dir),
                            "struct2atlas", species_cfg["cost"], "+xz", species_cfg["warp"], atlas_dir)  # +xz suits a coronal slab with limited A-P coverage

    # 3dAllineate matrices are base -> source; functional -> atlas = inverse(struct2atlas) after struct2func
    composite = atlas_dir / "func_to_atlas_for_roi_apply.aff12.1D"
    res = run_afni(["cat_matvec", "-ONELINE",
                    aff12_to_matvec(struct2func, atlas_dir / "struct2func.matvec"),
                    aff12_to_matvec(struct2atlas, atlas_dir / "struct2atlas.matvec"), "-I"], capture=True)
    composite.write_text(res.stdout.strip() + "\n")

    # Labels stay at full resolution; the matrix is in physical coordinates so the downsampled template doesn't matter here
    labels_funcspace = atlas_dir / "atlas_labels_funcspace.nii.gz"
    run_afni(["3dAllineate", "-source", atlas_labels, "-master", func_reference, "-1Dmatrix_apply", composite,
              "-final", "NN", "-prefix", labels_funcspace, "-overwrite"])
    print_statement(f"[OK] Atlas labels in functional space: {labels_funcspace}", bcolors.OKGREEN)
    return composite, labels_funcspace


# ---------------------------------------------------------------------------------------------------------------------
# Quality control of the registrations (like coregistration_qc of mol_fMRI_v5.6): it only inspects, nothing is changed.
# Each check writes an overlay image and a text report to <atlas_dir>/qc and prints a warning for anything that looks wrong.
# ---------------------------------------------------------------------------------------------------------------------

class _QcReport:
    def __init__(self):
        self.lines, self.warnings, self.metrics = [], [], {}

    def line(self, text=""):
        self.lines.append(text)

    def warn(self, text):
        self.warnings.append(text)

    def status(self):
        return "CHECK: " + "; ".join(self.warnings) if self.warnings else "no problems detected"

    def finish(self, title, out_txt):
        self.lines += ["", f"RESULT: {self.status()}"]
        Path(out_txt).write_text("\n".join(self.lines) + "\n")
        print_statement(f"--- {title} ---", bcolors.HEADER)
        print("\n".join(self.lines))
        for warning in self.warnings:
            print_statement(f"[WARNING] {warning}", bcolors.FAIL)
        if not self.warnings:
            print_statement(f"[OK] {title}: no problems detected", bcolors.OKGREEN)
        return {"metrics": self.metrics, "warnings": self.warnings}


def _qc_dir(atlas_dir, out_dir):
    out_dir = Path(out_dir) if out_dir else Path(atlas_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load3d(path):
    data = nib.load(str(path)).get_fdata()
    return data[..., 0] if data.ndim == 4 else data


def _warp(source, master, matrix, interpolation, tmp_dir, name):
    """Resample `source` onto the grid of `master` with a 3dAllineate matrix (base -> source, as saved by 3dAllineate)."""
    out = Path(tmp_dir) / name
    subprocess.run(["3dAllineate", "-source", str(source), "-master", str(master), "-1Dmatrix_apply", str(matrix),
                    "-final", interpolation, "-prefix", str(out), "-overwrite"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return _load3d(out)


def _identity_matrix(tmp_dir):
    path = Path(tmp_dir) / "identity.aff12.1D"
    path.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
    return path


def _helper_file(tmp_dir, name, reference, data):
    """Binary helper image on the grid of `reference` (structural brain mask, or an all-ones image for the coverage)."""
    img = nib.load(str(reference))
    path = Path(tmp_dir) / name
    nib.save(nib.Nifti1Image(data.astype(np.uint8), img.affine), str(path))
    return path


def _struct_mask_file(struct_file, struct_mask, tmp_dir):
    if struct_mask is not None and Path(struct_mask).exists():
        return struct_mask
    return _helper_file(tmp_dir, "struct_mask.nii.gz", struct_file, _load3d(struct_file) > 0)


def _read_params(params_file):
    """Shifts (mm), angles (deg) and, if the warp has them, scales from a 3dAllineate -1Dparam_save file."""
    if not Path(params_file).exists():
        return None
    values = [float(v) for line in open(params_file) if not line.startswith("#") for v in line.split()]
    names = ["x-shift (mm)", "y-shift (mm)", "z-shift (mm)", "z-angle (deg)", "x-angle (deg)", "y-angle (deg)"]
    params = dict(zip(names, values[:6]))
    scales = dict(zip(["x-scale", "y-scale", "z-scale"], values[6:9])) if len(values) >= 9 else {}
    return params, scales


def _nmi(a, b, bins=64):
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    p = hist / hist.sum()
    entropy = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    return (entropy(p.sum(1)) + entropy(p.sum(0))) / entropy(p)


def _dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def _edge_distance(mask_a, mask_b, zooms, axis):
    """Mean in-plane distance (mm) from the boundary of mask_a to the boundary of mask_b, averaged over the slices."""
    from scipy import ndimage as ndi
    sampling = [zooms[i] for i in range(3) if i != axis]
    values = []
    for a, b in zip(np.moveaxis(mask_a, axis, 0), np.moveaxis(mask_b, axis, 0)):
        if a.any() and b.any():
            edge_a, edge_b = a & ~ndi.binary_erosion(a), b & ~ndi.binary_erosion(b)
            values.append(ndi.distance_transform_edt(~edge_b, sampling=sampling)[edge_a].mean())
    return float(np.mean(values)) if values else float("nan")


def _slice_axis(path):
    """Axis of the anterior-posterior direction (coronal slices); axis 1 if the orientation is unknown."""
    codes = nib.aff2axcodes(nib.load(str(path)).affine)
    return next((i for i, code in enumerate(codes) if code in ("A", "P")), 1)


def _pick_slices(mask, axis, n_slices):
    present = np.where(np.moveaxis(mask, axis, 0).reshape(mask.shape[axis], -1).any(axis=1))[0]
    if len(present) == 0:
        return []
    # inner part of the range: the outermost slices only hold a thin sliver of the brain
    positions = np.linspace(0.1, 0.9, n_slices) * (len(present) - 1)
    return [int(present[i]) for i in np.unique(positions.round().astype(int))]


def _crop(mask, axis, pad=10):
    plane = mask.any(axis=axis)
    bounds = []
    for dim in range(2):
        idx = np.where(plane.any(axis=1 - dim))[0]
        bounds.append(slice(max(idx.min() - pad, 0), idx.max() + pad + 1) if len(idx) else slice(None))
    return tuple(bounds)


def _overlay_figure(panels, slices, axis, crop, title, out_png):
    """panels: list of (title, image, [(mask, colour), ...], labels or None); one row per slice."""
    import matplotlib.pyplot as plt
    if not slices:
        return
    fig, axes = plt.subplots(len(slices), len(panels), figsize=(5 * len(panels), 5 * len(slices)), squeeze=False)
    plane = lambda image, idx: np.take(image, idx, axis=axis)[crop]
    for row, idx in enumerate(slices):
        for col, (panel_title, image, outlines, labels) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(plane(image, idx).T, cmap="gray", origin="lower")
            if labels is not None:
                shown = plane(labels, idx)
                ax.imshow(np.ma.masked_where(shown == 0, shown % 20).T, cmap="tab20", alpha=0.5, origin="lower", interpolation="nearest")
            for mask, colour in outlines:
                mask_plane = plane(mask, idx).astype(float)
                if mask_plane.min() < 0.5 < mask_plane.max():
                    ax.contour(mask_plane.T, [0.5], colors=colour, linewidths=0.8)
            ax.set_title(f"{panel_title} (slice {idx})", fontsize=10)
            ax.axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_png, dpi=100)
    plt.show()
    plt.close(fig)


def qc_struct2func(struct_file, func_reference, brain_mask, atlas_dir, struct_mask=None, out_dir=None, n_slices=4,
                   min_dice=0.6):
    """QC of the structural -> functional registration (`struct2func_masked*` in atlas_dir): size of the transform,
    NMI and brain mask overlap against a header-only alignment, and an overlay image. Red outline: functional brain
    mask, cyan: structural brain mask moved onto the functional grid."""
    atlas_dir, out_dir = Path(atlas_dir), _qc_dir(atlas_dir, out_dir)
    matrix = atlas_dir / "struct2func_masked.aff12.1D"
    report = _QcReport()

    parsed = _read_params(atlas_dir / "struct2func_masked_params.1D")
    if parsed:
        report.line("Estimated transform (limits: "
                    f"{STRUCT2FUNC_MAX_SHIFT_MM:g} mm, {STRUCT2FUNC_MAX_ROT_DEG:g} deg):")
        for name, value in parsed[0].items():
            limit = STRUCT2FUNC_MAX_SHIFT_MM if "shift" in name else STRUCT2FUNC_MAX_ROT_DEG
            at_limit = abs(value) >= 0.95 * limit
            report.line(f"  {name:14s} {value:8.3f}{'  <-- at the limit' if at_limit else ''}")
            if at_limit:
                report.warn(f"{name} = {value:.2f} is at the limit of the allowed range ({limit:g}); the true offset may be larger")
        report.metrics["params"] = parsed[0]

    func, brain = _load3d(func_reference), _load3d(brain_mask) > 0
    if func.shape != brain.shape:
        raise ValueError(f"Functional brain mask {brain.shape} and functional image {func.shape} differ in shape.")
    axis = _slice_axis(func_reference)
    zooms = nib.load(str(func_reference)).header.get_zooms()[:3]

    with tempfile.TemporaryDirectory() as tmp:
        identity = _identity_matrix(tmp)
        smask = _struct_mask_file(struct_file, struct_mask, tmp)
        ones = _helper_file(tmp, "ones.nii.gz", struct_file, np.ones(nib.load(str(struct_file)).shape[:3]))
        reg = _warp(struct_file, func_reference, matrix, "linear", tmp, "reg.nii.gz")
        hdr = _warp(struct_file, func_reference, identity, "linear", tmp, "hdr.nii.gz")
        reg_mask = _warp(smask, func_reference, matrix, "NN", tmp, "reg_mask.nii.gz") > 0
        hdr_mask = _warp(smask, func_reference, identity, "NN", tmp, "hdr_mask.nii.gz") > 0
        reg_cover = _warp(ones, func_reference, matrix, "NN", tmp, "reg_cover.nii.gz") > 0
        hdr_cover = _warp(ones, func_reference, identity, "NN", tmp, "hdr_cover.nii.gz") > 0

    region = brain & ((reg > 0) | (hdr > 0))
    if region.sum() > 100:
        report.metrics["nmi_registered"] = float(_nmi(func[region], reg[region]))
        report.metrics["nmi_header_only"] = float(_nmi(func[region], hdr[region]))
        report.line("")
        report.line(f"NMI with the functional image (higher is better): registered {report.metrics['nmi_registered']:.4f}, "
                    f"header-only {report.metrics['nmi_header_only']:.4f}")
        if report.metrics["nmi_registered"] < 0.97 * report.metrics["nmi_header_only"]:
            report.warn("registration has a clearly LOWER mutual information (> 3%) than the header-only alignment")
    else:
        report.warn("the structural and functional images hardly overlap")

    report.metrics["dice_registered"] = _dice(reg_mask, brain & reg_cover)
    report.metrics["dice_header_only"] = _dice(hdr_mask, brain & hdr_cover)
    report.metrics["edge_mm_registered"] = _edge_distance(reg_mask, brain, zooms, axis)
    report.metrics["edge_mm_header_only"] = _edge_distance(hdr_mask, brain, zooms, axis)
    report.line(f"Brain mask Dice (in structural coverage): registered {report.metrics['dice_registered']:.3f}, "
                f"header-only {report.metrics['dice_header_only']:.3f}")
    report.line(f"Mean in-plane edge distance (mm):         registered {report.metrics['edge_mm_registered']:.3f}, "
                f"header-only {report.metrics['edge_mm_header_only']:.3f}")
    if report.metrics["dice_registered"] < min_dice:
        report.warn(f"brain mask Dice {report.metrics['dice_registered']:.2f} is below {min_dice}")
    if (report.metrics["dice_registered"] < report.metrics["dice_header_only"] - 0.02
            or report.metrics["edge_mm_registered"] > report.metrics["edge_mm_header_only"] + 0.1):
        report.warn("brain mask overlap is clearly WORSE than the header-only alignment")

    slices = _pick_slices(brain, axis, n_slices)
    out_png = out_dir / "qc_struct2func.png"
    _overlay_figure([("header-only alignment", hdr, [(brain, "red"), (hdr_mask, "cyan")], None),
                     ("registered", reg, [(brain, "red"), (reg_mask, "cyan")], None),
                     ("functional image", func, [(brain, "red"), (reg_mask, "cyan")], None)],
                    slices, axis, _crop(brain, axis),
                    f"struct2func QC - red: functional brain mask, cyan: structural brain mask\n{report.status()}", out_png)
    report.line(f"Overlay image: {out_png}")
    return report.finish("struct2func QC", out_dir / "qc_struct2func.txt")


def qc_struct2atlas(struct_file, atlas_dir, template, atlas_labels, struct_mask=None, out_dir=None, n_slices=4,
                    min_dice=0.6):
    """QC of the structural -> atlas registration (`struct2atlas*` in atlas_dir): the transform, and how well the
    structural brain mask matches the atlas brain (labels > 0) inside the part of the atlas covered by the structural
    scan. Red outline: atlas brain, cyan: structural brain mask on the template grid."""
    atlas_dir, out_dir = Path(atlas_dir), _qc_dir(atlas_dir, out_dir)
    report = _QcReport()

    parsed = _read_params(atlas_dir / "struct2atlas_params.1D")
    if parsed:
        report.line("Estimated transform:")
        for name, value in {**parsed[0], **parsed[1]}.items():
            report.line(f"  {name:14s} {value:8.3f}")
        report.metrics["params"] = {**parsed[0], **parsed[1]}
        for name, value in parsed[1].items():
            if abs(value - 1) > 0.25:
                report.warn(f"{name} = {value:.2f}: the registration needed a large size change")

    template_img = _load3d(template)
    axis = _slice_axis(template)
    zooms = nib.load(str(template)).header.get_zooms()[:3]

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["3dresample", "-master", str(template), "-rmode", "NN", "-input", str(atlas_labels),
                        "-prefix", str(Path(tmp) / "labels.nii.gz")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atlas_brain = _load3d(Path(tmp) / "labels.nii.gz") > 0
        matrix = atlas_dir / "struct2atlas.aff12.1D"
        smask = _struct_mask_file(struct_file, struct_mask, tmp)
        ones = _helper_file(tmp, "ones.nii.gz", struct_file, np.ones(nib.load(str(struct_file)).shape[:3]))
        reg = _warp(struct_file, template, matrix, "linear", tmp, "reg.nii.gz")
        reg_mask = _warp(smask, template, matrix, "NN", tmp, "reg_mask.nii.gz") > 0
        cover = _warp(ones, template, matrix, "NN", tmp, "cover.nii.gz") > 0

    atlas_in_cover = atlas_brain & cover
    if not reg_mask.any() or not atlas_in_cover.any():
        report.warn("the structural scan does not overlap the atlas brain at all")
    else:
        report.metrics["dice"] = _dice(reg_mask, atlas_in_cover)
        offset = (np.array(np.nonzero(reg_mask)).mean(axis=1) - np.array(np.nonzero(atlas_in_cover)).mean(axis=1)) * np.array(zooms)
        report.metrics["centroid_offset_mm"] = float(np.linalg.norm(offset))
        report.metrics["volume_ratio"] = float(reg_mask.sum() / atlas_in_cover.sum())
        report.line("")
        report.line(f"Brain mask Dice (in structural coverage): {report.metrics['dice']:.3f}")
        report.line(f"Distance between the centres of the structural and atlas brain: {report.metrics['centroid_offset_mm']:.2f} mm")
        report.line(f"Brain volume structural / atlas (in structural coverage): {report.metrics['volume_ratio']:.2f}")
        if report.metrics["dice"] < min_dice:
            report.warn(f"brain mask Dice {report.metrics['dice']:.2f} is below {min_dice}")

    slices = _pick_slices(reg_mask & cover, axis, n_slices)
    out_png = out_dir / "qc_struct2atlas.png"
    outlines = [(atlas_brain, "red"), (reg_mask, "cyan")]
    _overlay_figure([("atlas template", template_img, outlines, None), ("registered structural", reg, outlines, None)],
                    slices, axis, _crop(atlas_brain, axis),
                    f"struct2atlas QC - red: atlas brain, cyan: structural brain mask\n{report.status()}", out_png)
    report.line(f"Overlay image: {out_png}")
    return report.finish("struct2atlas QC", out_dir / "qc_struct2atlas.txt")


def qc_labels_funcspace(labels_file, func_reference, brain_mask, out_dir, min_voxels=5, n_slices=4,
                        min_coverage=0.9, min_dice=0.7):
    """QC of the end result: the atlas labels on the functional grid against the functional brain mask. Shows how much
    of the brain has an atlas label and how many regions have enough voxels inside the mask. Red outline: functional
    brain mask, cyan: atlas brain (labels > 0)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _QcReport()
    labels = np.rint(_load3d(labels_file)).astype(int)
    brain = _load3d(brain_mask) > 0
    func = _load3d(func_reference)
    if brain.shape != labels.shape:
        raise ValueError(f"Brain mask {brain.shape} and atlas labels {labels.shape} differ in shape.")

    atlas_brain = labels > 0
    coverage = float(atlas_brain[brain].mean())
    dice = _dice(atlas_brain, brain)
    ids, counts = np.unique(labels[brain & atlas_brain], return_counts=True)
    small = int((counts < min_voxels).sum())
    report.metrics.update({"label_coverage": coverage, "dice": dice, "regions": int(len(ids)), "regions_below_min_voxels": small})
    report.line(f"Share of the brain mask voxels that have an atlas label: {100 * coverage:.1f}%")
    report.line(f"Dice between atlas brain (labels > 0) and functional brain mask: {dice:.3f}")
    report.line(f"Regions inside the brain mask: {len(ids)}, of which {small} have fewer than {min_voxels} voxels")
    if coverage < min_coverage:
        report.warn(f"only {100 * coverage:.0f}% of the brain mask has an atlas label (< {100 * min_coverage:.0f}%)")
    if dice < min_dice:
        report.warn(f"atlas brain and brain mask overlap poorly (Dice {dice:.2f} < {min_dice})")

    axis = _slice_axis(func_reference)
    out_png = out_dir / "qc_labels_funcspace.png"
    outlines = [(brain, "red"), (atlas_brain, "cyan")]
    _overlay_figure([("brain mask and atlas brain", func, outlines, None), ("atlas labels", func, [(brain, "red")], labels)],
                    _pick_slices(brain, axis, n_slices), axis, _crop(brain | atlas_brain, axis),
                    f"Atlas labels in functional space QC - red: functional brain mask, cyan: atlas brain\n{report.status()}", out_png)
    report.line(f"Overlay image: {out_png}")
    return report.finish("Atlas labels QC", out_dir / "qc_labels_funcspace.txt")


def registration_qc(struct_file, func_reference, brain_mask, atlas_dir, species_cfg, atlas_template, atlas_labels,
                    labels_funcspace, struct_mask=None, atlas_mask="", min_voxels=5):
    """All three checks after `register_to_atlas`. Returns {"struct2func": ..., "struct2atlas": ..., "labels": ...},
    each with "metrics" and "warnings"."""
    atlas_dir = Path(atlas_dir)
    template = prepare_registration_template(atlas_template, atlas_mask, species_cfg["reg_voxel_mm"], atlas_dir)
    return {
        "struct2func": qc_struct2func(struct_file, func_reference, brain_mask, atlas_dir, struct_mask),
        "struct2atlas": qc_struct2atlas(struct_file, atlas_dir, template, atlas_labels, struct_mask),
        "labels": qc_labels_funcspace(labels_funcspace, func_reference, brain_mask, atlas_dir / "qc", min_voxels),
    }


def load_label_names(csv_path):
    """Accepts an id,name CSV or the DSURQE/SIGMA layouts with separate left and right label columns."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "id" in cols and "name" in cols:
        return {int(i): str(n) for i, n in zip(df[cols["id"]], df[cols["name"]])}
    left = next(c for k, c in cols.items() if "left" in k and "label" in k)
    right = next(c for k, c in cols.items() if "right" in k and "label" in k)
    name_col = next(c for k, c in cols.items() if k in ("structure", "region of interest"))
    names = {}
    for _, row in df.iterrows():
        for col, side in ((left, "L"), (right, "R")):
            if pd.notna(row[col]):
                names[int(row[col])] = f"{row[name_col]}_{side}"
    return names


def safe_name(text):
    return re.sub(r"[^\w.\-]+", "_", str(text)).strip("_")
