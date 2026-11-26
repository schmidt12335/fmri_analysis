"""Automatic brain masks (structural and functional) and their quality check.

Copied from the injection pipeline notebook mol_fMRI_v5.8.ipynb (cell 1, unchanged apart from the imports) for the
electrical stimulation preprocessing notebook (mol_fMRI_electric_stim_preprocessing_v2.4.ipynb). Still in testing:
see CHANGELOG.md (v5.8 and stimulation notebooks v2.4) for the methods and the test results.

Masks:
- structural: `automatic_structural_mask` (3dSkullStrip -rat on the head crop + 3dAutomask within 0.4 mm), made or found by
  `prepare_structural_mask` (an existing mask, hand-drawn or automatic, is never replaced)
- functional: `automatic_functional_mask` (structural mask moved to the functional grid through the headers; EPI: trimmed with
  an intensity threshold; SORDINO: partial-volume transfer, no trimming, slab of structural slices via `limit_mask_to_slab`)
- `mask_quality_check`: overlay figure and warnings for gross failures; the figure has to be looked at
- `record_mask_source` / `mask_source`: automatic masks are recorded in <mask>.source.json; a mask saved over it counts as hand-drawn
"""
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from electric_stim_helpers import bcolors, bruker_to_nifti, print_statement
from electric_stim_session import structural_mask_file as find_structural_mask

__all__ = ["record_mask_source", "mask_source", "find_structural_mask", "limit_mask_to_slab", "automatic_structural_mask",
           "automatic_functional_mask", "prepare_structural_mask", "MASK_QC_LIMITS", "mask_quality_check", "mean_of_middle_volumes"]


def record_mask_source(mask_file, method, **info):
    """Remember that a mask was made automatically (`<mask>.source.json`: method, file size and time). If a hand-drawn mask is saved
    over it later, size and time no longer match and the mask counts as hand-drawn again (see `mask_source`)."""
    mask_file = Path(mask_file)
    stat = mask_file.stat()
    Path(f"{mask_file}.source.json").write_text(json.dumps({"method": method, "size": stat.st_size, "mtime": stat.st_mtime, **info}, indent=1))


def mask_source(mask_file):
    """Method that created `mask_file` ('hand-drawn' unless it was created by an automatic method and not changed since)."""
    record = Path(f"{mask_file}.source.json")
    if not record.exists() or not Path(mask_file).exists():
        return "hand-drawn"
    info = json.loads(record.read_text())
    stat = Path(mask_file).stat()
    if stat.st_size == info["size"] and abs(stat.st_mtime - info["mtime"]) < 2:
        return info["method"]
    return "hand-drawn"


def _inplane_disk(slice_axis, radius):
    """3D structuring element: a disk of `radius` voxels in the image plane, one voxel thick along `slice_axis`, so that a
    morphological operation with it works slice by slice."""
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return np.expand_dims(yy ** 2 + xx ** 2 <= r * r, slice_axis)


def _clean_brain_mask(mask, slice_axis, open_radius):
    """In-plane opening (removes thin bridges to tissue outside the brain), largest connected piece, holes filled slice by slice
    and in 3D."""
    from scipy import ndimage as ndi

    opened = ndi.binary_opening(mask, structure=_inplane_disk(slice_axis, open_radius))
    if opened.any():
        mask = opened
    labels, n = ndi.label(mask)
    if n > 1:
        mask = labels == 1 + int(np.argmax(ndi.sum(mask, labels, range(1, n + 1))))
    planes = np.moveaxis(mask, slice_axis, 0).copy()
    for k in range(planes.shape[0]):
        planes[k] = ndi.binary_fill_holes(planes[k])
    return ndi.binary_fill_holes(np.moveaxis(planes, 0, slice_axis))


def limit_mask_to_slab(mask, affine, n_slices, shift_slices=0):
    """Keep `n_slices` consecutive slices of a brain mask: along the slice axis (the axis with the largest voxel size), centred on the
    widest slice of the brain (mask area per slice, smoothed over 3 slices) and moved by `shift_slices` slices towards anterior (+A of
    the image coordinates).

    Used for SORDINO: the 3D sordino volume contains the whole head, while the analysis covers the same slab as the hand-drawn masks
    (20 slices of 0.7 mm centred on the striatum, which lies about 3 slices anterior of the widest brain slice; in 3 test sessions this
    rule gave the hand-drawn slab exactly or one slice off).

    Returns (mask, (first slice, last slice), slice axis). A mask that spans `n_slices` slices or fewer is returned unchanged."""
    from scipy import ndimage as ndi

    affine = np.asarray(affine, dtype=float)
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    ax = int(np.argmax(zooms))
    area = mask.sum(axis=tuple(i for i in range(3) if i != ax)).astype(float)
    present = np.where(area > 0)[0]
    if present.size == 0 or present.max() - present.min() + 1 <= n_slices:
        return mask, ((int(present.min()), int(present.max())) if present.size else (0, -1)), ax
    if abs(affine[1, ax]) < 0.5 * zooms[ax]:
        print_statement(f"The slices are not coronal: the slab of {n_slices} slices is centred on the widest brain slice without the shift.", bcolors.NOTIFICATION)
        shift_slices = 0
    step = 1 if affine[1, ax] > 0 else -1
    centre = int(np.argmax(ndi.uniform_filter1d(area, 3))) + step * int(shift_slices)
    first = int(np.clip(centre - n_slices // 2, 0, max(0, mask.shape[ax] - n_slices)))
    last = min(first + n_slices, mask.shape[ax]) - 1
    keep = np.zeros(mask.shape[ax], bool)
    keep[first:last + 1] = True
    shape = [1, 1, 1]
    shape[ax] = -1
    return mask & keep.reshape(shape), (first, last), ax


def automatic_structural_mask(struct_file, out_mask, n_slices=None, shift_slices=0):
    """Brain mask of the structural scan: AFNI `3dSkullStrip -rat` (on the image resampled to isotropic voxels, as the rat mode expects),
    extended by `3dAutomask` close to it, then cleaned (in-plane opening of 0.3 mm, largest connected piece, holes filled).

    3dSkullStrip alone follows the brain edge well, but in an image with a strong intensity fall-off (surface coil) it can cut off the
    darker, ventral part of the brain along a straight line; 3dAutomask reaches the ventral edge. 3dAutomask can also take in bright
    tissue outside the brain (in one SORDINO test session both temporalis muscles), so only its voxels within 0.4 mm (in-plane) of the
    3dSkullStrip mask are added. 3dSkullStrip runs on the image cropped to the head (3dAutomask + 2 mm), and its mask is only used if it
    has 0.5-2x the size of the 3dAutomask mask (it can fail silently, e.g. lock onto a blob in the noise of a large field of view).

    Tests against hand-drawn masks (inside the slices covered by the hand-drawn mask): the full union of both (used in v5.8 before
    2026-11-25) gave Dice 0.97 (lowest 0.96) in 14 rat EPI sessions, but 0.83 in the SORDINO session with the muscles; the restricted
    version gave 0.94-0.96 in the 3 SORDINO sessions. It has not been re-tested on the 14 EPI sessions. If 3dSkullStrip fails,
    3dAutomask alone is used (Dice 0.93 in the EPI tests; it can include muscle).

    n_slices, shift_slices : optional (SORDINO)
        Keep only `n_slices` slices of the mask, see `limit_mask_to_slab`.

    Returns the name of the method that made the mask."""
    from scipy import ndimage as ndi
    import tempfile

    img = nib.load(str(struct_file))
    zooms = img.header.get_zooms()[:3]
    slice_axis = int(np.argmax(zooms))
    in_plane = min(float(z) for i, z in enumerate(zooms) if i != slice_axis)
    quiet = {"check": True, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["3dAutomask", "-overwrite", "-clfrac", "0.6", "-peels", "4", "-prefix", str(tmp / "automask.nii.gz"), str(struct_file)], **quiet)
        automask = np.asarray(nib.load(str(tmp / "automask.nii.gz")).dataobj).squeeze() > 0
        labels, n = ndi.label(automask)
        head = labels == 1 + int(np.argmax(ndi.sum(automask, labels, range(1, n + 1)))) if n else automask
        # 3dSkullStrip runs on the head only (3dAutomask's largest piece + 2 mm in-plane, all slices): in a large field of view with much
        # noise around a small head (38 mm, 0.055 mm voxels) it locked onto a blob in the noise outside the head
        strip_input = struct_file
        if head.any():
            box = []
            for i in range(3):
                found = np.where(head.any(axis=tuple(j for j in range(3) if j != i)))[0]
                margin = int(np.ceil(2.0 / float(zooms[i])))
                box.append(slice(None) if i == slice_axis else slice(max(0, found.min() - margin), min(head.shape[i], found.max() + margin + 1)))
            strip_input = tmp / "head.nii.gz"
            nib.save(img.slicer[tuple(box)], str(strip_input))
        try:
            voxel = min(zooms)
            subprocess.run(["3dresample", "-overwrite", "-dxyz", str(voxel), str(voxel), str(voxel), "-rmode", "Cu", "-prefix", str(tmp / "iso.nii.gz"),
                            "-input", str(strip_input)], **quiet)
            subprocess.run(["3dSkullStrip", "-overwrite", "-input", str(tmp / "iso.nii.gz"), "-prefix", str(tmp / "ss.nii.gz"), "-rat", "-push_to_edge"], **quiet)
            subprocess.run(["3dresample", "-overwrite", "-master", str(struct_file), "-rmode", "NN", "-prefix", str(tmp / "back.nii.gz"),
                            "-input", str(tmp / "ss.nii.gz")], **quiet)
            skullstrip = np.asarray(nib.load(str(tmp / "back.nii.gz")).dataobj).squeeze() > 0
            if not skullstrip.any():
                raise RuntimeError("empty mask")
            # a failed 3dSkullStrip does not report an error: its mask must have about the size of the head found by 3dAutomask
            ratio = skullstrip.sum() / max(1, head.sum())
            if head.any() and not 0.5 <= ratio <= 2.0:
                raise RuntimeError(f"implausible mask, {ratio:.2f} x the size of the 3dAutomask mask")
        except (subprocess.CalledProcessError, RuntimeError) as error:
            print_statement(f"3dSkullStrip failed ({error}): 3dAutomask alone is used (less accurate: it can cut off the darker ventral brain "
                            "and include muscle; check the figure)", bcolors.FAIL)
            skullstrip = None

    if skullstrip is None:
        mask, method = automask, "3dAutomask"
    else:
        near = ndi.binary_dilation(skullstrip, structure=_inplane_disk(slice_axis, round(0.4 / in_plane)))
        mask, method = skullstrip | (automask & near), "3dSkullStrip -rat + 3dAutomask within 0.4 mm"
    mask = _clean_brain_mask(mask, slice_axis, round(0.3 / in_plane))
    info = {}
    if n_slices:
        mask, (first, last), _ = limit_mask_to_slab(mask, img.affine, int(n_slices), int(shift_slices))
        info = {"n_slices": int(n_slices), "shift_slices": int(shift_slices), "slab": [first, last]}
        method += f", limited to {int(n_slices)} slices"

    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_data_shape(mask.shape)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, header), str(out_mask))
    record_mask_source(out_mask, method, **info)
    slab_text = f", slices {info['slab'][0]}-{info['slab'][1]}" if info else ""
    print_statement(f"[OK] Automatic structural mask ({method}) -> {out_mask} ({int(mask.sum())} voxels{slab_text})", bcolors.OKGREEN)
    return method


def automatic_functional_mask(func_image, struct_mask, out_mask, prior_out=None, trim_rel=0.5, partial_volume=False, pv_threshold=0.3,
                              n_slices=None, shift_slices=0):
    """Brain mask of the functional image made from the structural mask (hand-drawn or automatic).

    Both scans of a session share the scanner coordinates, so the structural mask is moved to the functional grid through the image
    headers only (no registration). Two variants:

    * EPI (default arguments): moved with nearest neighbour, then trimmed with an intensity threshold (voxels of a smoothed functional
      image below `trim_rel` x the median inside the mask are removed). Dice against hand-drawn masks 0.89 (lowest 0.72, 13 rat sessions).
    * SORDINO (`partial_volume=True, trim_rel=None, n_slices=20, shift_slices=3`): sordino has almost no intensity edge at the brain
      border (inside/outside ratio 1.0-1.3), so the trimming removes nothing and is not used. The structural mask is limited to
      `n_slices` slices (`limit_mask_to_slab`; a mask that spans no more slices, e.g. a hand-drawn mask of 20 slices, stays as it is)
      and moved as a partial-volume fraction: a functional voxel is brain if at least `pv_threshold` of it is covered by the structural
      mask (nearest neighbour puts the edge about half a functional voxel inside). In 3 rat sessions, fully automatic: Dice 0.87-0.93
      against the hand-drawn masks inside the 20-slice slab (before: 0.86-0.92, with the hand-drawn structural mask).

    Holes are filled in both variants. CHECK the result: the mask is wrong if the animal moved between the structural and the functional
    scan (in one of the 3 sordino test sessions it was about one voxel off, probably for this reason).

    Parameters
    ----------
    func_image : str
        3D functional image (the notebook uses the mean of the volumes around the middle of the series).
    struct_mask : str
        Brain mask of the structural scan.
    out_mask : str
        Output mask on the functional grid.
    prior_out : str, optional
        If given, the structural mask on the functional grid before the trimming / hole filling is saved here (shown in the quality check).
    trim_rel : float or None
        Intensity trimming (None = no trimming).
    partial_volume : bool
        Move the mask as a partial-volume fraction (threshold `pv_threshold`) instead of nearest neighbour.
    n_slices, shift_slices : optional
        Limit the structural mask to a slab of slices first (`limit_mask_to_slab`).

    Returns
    -------
    dict with `method`, `prior_voxels`, `voxels` and `struct_slab` (first and last structural slice, if limited)."""
    from scipy import ndimage as ndi
    import tempfile

    img = nib.load(str(func_image))
    data = np.asarray(img.dataobj, dtype=np.float32).squeeze()
    simg = nib.load(str(struct_mask))
    smask = np.asarray(simg.dataobj).squeeze() > 0
    slab = None
    if n_slices:
        smask, slab, _ = limit_mask_to_slab(smask, simg.affine, int(n_slices), int(shift_slices))
    source_header = simg.header.copy()
    if partial_volume:
        # fraction of each functional voxel covered by the mask: box average over the size of a functional voxel (measured along each
        # structural axis; odd box widths so the box is centred), then linear interpolation at the functional voxel centres
        to_func_index = np.linalg.inv(img.affine[:3, :3])
        sizes = []
        for i in range(3):
            step_mm = np.linalg.norm(simg.affine[:3, i])
            extent_mm = 1.0 / max(np.abs(to_func_index @ (simg.affine[:3, i] / step_mm)).max(), 1e-9)
            n = max(1, int(round(extent_mm / step_mm)))
            sizes.append(n + 1 - n % 2)
        values, final = ndi.uniform_filter(smask.astype(np.float32), size=sizes, mode="constant"), "linear"
        source_header.set_data_dtype(np.float32)
    else:
        values, final = smask.astype(np.uint8), "NN"
        source_header.set_data_dtype(np.uint8)
    source_header.set_data_shape(values.shape)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        identity = tmp / "identity.aff12.1D"
        identity.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
        source = tmp / "struct_mask_source.nii.gz"
        nib.save(nib.Nifti1Image(values, simg.affine, source_header), str(source))
        moved = tmp / "struct_mask_in_func.nii.gz"
        subprocess.run(["3dAllineate", "-overwrite", "-source", str(source), "-master", str(func_image), "-1Dmatrix_apply", str(identity),
                        "-final", final, "-prefix", str(moved)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        moved_values = np.asarray(nib.load(str(moved)).dataobj, dtype=np.float32).squeeze()
    prior = moved_values >= pv_threshold if partial_volume else moved_values > 0
    if not prior.any():
        raise ValueError("The structural mask lies outside the functional image (check that both scans share the same position).")

    if trim_rel is None:
        mask = prior.copy()
    else:
        smooth = ndi.gaussian_filter(data, 1.0)
        mask = prior & (smooth > trim_rel * np.median(smooth[prior]))
    for y in range(mask.shape[1]):
        mask[:, y, :] = ndi.binary_fill_holes(mask[:, y, :])
    mask = ndi.binary_fill_holes(mask)

    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_data_shape(mask.shape)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, header), str(out_mask))
    if prior_out is not None:
        nib.save(nib.Nifti1Image(prior.astype(np.uint8), img.affine, header), str(prior_out))
    method = "structural mask" + (f" limited to {int(n_slices)} slices" if slab is not None and n_slices else "")
    method += f" moved to the functional grid as partial-volume fraction >= {pv_threshold}" if partial_volume else " moved to the functional grid"
    method += " + intensity trimming" if trim_rel is not None else ""
    record_mask_source(out_mask, method, **({"struct_slab": list(slab)} if slab is not None else {}))
    slab_text = f"; structural slices {slab[0]}-{slab[1]}" if slab is not None else ""
    print_statement(f"[OK] Automatic functional mask -> {out_mask} ({int(mask.sum())} voxels; {int(prior.sum())} before the "
                    f"{'intensity trimming and ' if trim_rel is not None else ''}hole filling{slab_text})", bcolors.OKGREEN)
    return {"method": method, "prior_voxels": int(prior.sum()), "voxels": int(mask.sum()), "struct_slab": slab}


def prepare_structural_mask(in_path, struct_scan_number, analysed_struct_dir, source, functional_reference=None, force_qc=True,
                            n_slices=None, shift_slices=0):
    """Make sure the structural scan is converted and its brain mask exists; returns the path of the mask.

    An existing mask (hand-drawn, or automatic from an earlier run) is never replaced. A missing mask is made automatically
    (source="auto", `automatic_structural_mask`) or drawn by hand in fsleyes (source="hand"). The conversion needs the working
    directory to be the structural folder, which is restored afterwards.

    Parameters
    ----------
    in_path, struct_scan_number : raw data folder and the Bruker scan number of the structural scan
    analysed_struct_dir : folder of the analysed structural data
    source : "hand" or "auto"
    functional_reference : str, optional
        Image shown next to the structural scan in fsleyes when the mask is drawn by hand.
    n_slices, shift_slices : optional (SORDINO)
        A new automatic mask covers only this many slices (`limit_mask_to_slab`). An existing mask is not changed."""
    analysed_struct_dir = Path(analysed_struct_dir)
    analysed_struct_dir.mkdir(parents=True, exist_ok=True)
    previous_dir = os.getcwd()
    try:
        os.chdir(analysed_struct_dir)
        if not (analysed_struct_dir / "struct.nii.gz").exists():
            print_statement("Converting the structural scan (needed for the structural mask)", bcolors.NOTIFICATION)
            bruker_to_nifti(in_path, str(struct_scan_number), "struct.nii.gz")
        mask_path = find_structural_mask(analysed_struct_dir)
        if mask_path.exists():
            print_statement(f"Structural mask exists ({mask_path.name}, {mask_source(mask_path)}).", bcolors.OKGREEN)
        elif source == "auto":
            automatic_structural_mask(analysed_struct_dir / "struct.nii.gz", mask_path, n_slices=n_slices, shift_slices=shift_slices)
        else:
            print_statement("Please create a mask for the structural image and save it as struct_mask.nii.gz", bcolors.NOTIFICATION)
            subprocess.run(["fsleyes", "struct.nii.gz"] + ([str(functional_reference)] if functional_reference else []))
            if not mask_path.exists():
                raise FileNotFoundError(f"{mask_path} not found: save the hand-drawn structural brain mask under this name.")
    finally:
        os.chdir(previous_dir)
    return mask_path


# Limits of the mask quality check, set from hand-drawn and automatic masks of earlier rat sessions. Edge contrast: hand-drawn and automatic
# structural masks were 2.1-3.2, functional masks 0.9-2.2. The checks only catch gross failures: for functional masks none of the numbers
# predicted which automatic masks overlapped less with a hand-drawn mask, so the figure has to be looked at.
MASK_QC_LIMITS = {"structural_edge_contrast": 1.5, "functional_edge_contrast": 1.05, "kept_fraction": 0.6, "centre_shift_mm": 1.0}


def mask_quality_check(image_file, mask_file, kind, out_png, prior_file=None, title=None, show=True, edge_warning=True):
    """Quality check of a brain mask: numbers, warnings and an overlay figure of the slices (red outline = mask; cyan dashed = the structural
    mask before the trimming, for an automatic functional mask). The checks cannot prove that a mask is right, they only catch typical
    failures; always look at the figure.

    Checks
    ------
    * connected pieces (a brain is one piece); empty slices between the first and the last slice of the mask;
    * share of the mask at the border of the image (the mask runs out of the image);
    * edge contrast: mean intensity in a thin band just inside the mask edge divided by the band just outside, in-plane. The brain is
      brighter than its surroundings in these images, so a ratio close to or below 1 means that the edge does not follow the brain;
    * for a functional mask made from the structural mask: share of the structural mask that was kept by the intensity trimming and the
      shift of the mask centre by it (a large change means the two scans do not agree, e.g. because the animal moved).

    Parameters
    ----------
    image_file, mask_file : str
        Image (3D, or 4D: the first volume is used) and mask on the same grid.
    kind : "structural" or "functional"
    out_png : str
        Overlay figure.
    prior_file : str, optional
        Structural mask on the functional grid before the trimming.
    edge_warning : bool
        Warn about a low edge contrast (False for SORDINO, which has almost no intensity edge at the brain border; the number is
        still reported).

    Returns
    -------
    dict with the numbers and `warnings` (a list of texts, empty if all checks are passed)."""
    from scipy import ndimage as ndi

    img = nib.load(str(image_file))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = data.squeeze()
    mask = np.asarray(nib.load(str(mask_file)).dataobj).squeeze() > 0
    zooms = img.header.get_zooms()[:3]
    prior = np.asarray(nib.load(str(prior_file)).dataobj).squeeze() > 0 if prior_file else None

    result = {"voxels": int(mask.sum()), "volume_mm3": float(mask.sum() * np.prod(zooms)), "warnings": []}
    warnings = result["warnings"]
    if not mask.any():
        warnings.append("the mask is empty")
        return result

    labels, n_pieces = ndi.label(mask, structure=np.ones((3, 3, 3)))
    result["pieces"] = int(n_pieces)
    if n_pieces > 1:
        sizes = np.bincount(labels.ravel())[1:]
        small = int(sizes.sum() - sizes.max())
        result["voxels_outside_largest_piece"] = small
        if small > 0.01 * mask.sum():
            warnings.append(f"the mask has {n_pieces} separate pieces ({small} voxels outside the largest)")

    slices = np.where(mask.any(axis=(0, 2)))[0]
    gaps = [int(y) for y in range(slices.min(), slices.max() + 1) if y not in set(slices.tolist())]
    result["slices"] = (int(slices.min()), int(slices.max()))
    if gaps:
        warnings.append(f"empty slice(s) inside the mask: {gaps}")

    rim = np.zeros(mask.shape, bool)
    rim[:2, :, :] = rim[-2:, :, :] = rim[:, :, :2] = rim[:, :, -2:] = True
    result["border_fraction"] = float((mask & rim).sum() / mask.sum())
    if result["border_fraction"] > 0.02:
        warnings.append(f"{100 * result['border_fraction']:.0f}% of the mask touches the border of the image")

    width = int(min(6, max(1, round(0.3 / zooms[0]))))
    inner, outer = [], []
    for y in slices:
        plane = mask[:, y, :]
        inner.append(data[:, y, :][plane & ~ndi.binary_erosion(plane, iterations=width)])
        outer.append(data[:, y, :][ndi.binary_dilation(plane, iterations=width) & ~plane])
    inner, outer = np.concatenate(inner), np.concatenate(outer)
    result["edge_contrast"] = float(inner.mean() / max(outer.mean(), 1e-6)) if len(inner) and len(outer) else float("nan")
    min_contrast = {"structural": MASK_QC_LIMITS["structural_edge_contrast"], "functional": MASK_QC_LIMITS["functional_edge_contrast"]}[kind]
    if edge_warning and result["edge_contrast"] < min_contrast:
        warnings.append(f"low edge contrast {result['edge_contrast']:.2f} (< {min_contrast}): the edge of the mask does not follow a brain/surrounding border")

    if prior is not None and prior.any():
        result["kept_fraction"] = float(mask.sum() / prior.sum())
        shift = np.linalg.norm((np.array(ndi.center_of_mass(mask)) - np.array(ndi.center_of_mass(prior))) * np.array(zooms))
        result["centre_shift_mm"] = float(shift)
        if result["kept_fraction"] < MASK_QC_LIMITS["kept_fraction"]:
            warnings.append(f"only {100 * result['kept_fraction']:.0f}% of the structural mask is kept in the functional image: the scans may not agree (animal moved?)")
        if shift > MASK_QC_LIMITS["centre_shift_mm"]:
            warnings.append(f"the intensity trimming moved the mask centre by {shift:.2f} mm (> {MASK_QC_LIMITS['centre_shift_mm']} mm)")

    # overlay figure
    shown = np.unique(np.linspace(slices.min(), slices.max(), 12).round().astype(int))
    n_cols = 4
    n_rows = int(np.ceil(len(shown) / n_cols))
    xs, zs = np.where(mask.any(axis=1))
    pad = max(3, int(0.15 * max(np.ptp(xs), np.ptp(zs))))
    x0, x1, z0, z1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[0] - 1), max(zs.min() - pad, 0), min(zs.max() + pad, mask.shape[2] - 1)
    vmax = np.percentile(data[mask], 99.5)
    aspect = (z1 - z0 + 1) / (x1 - x0 + 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, (4 * aspect + 0.5) * n_rows + 1.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, y in zip(axes.ravel(), shown):
        ax.imshow(data[x0:x1 + 1, y, z0:z1 + 1].T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        if prior is not None and prior[:, y, :].any():
            ax.contour(prior[x0:x1 + 1, y, z0:z1 + 1].T.astype(float), [0.5], colors="cyan", linewidths=1, linestyles="dashed")
        ax.contour(mask[x0:x1 + 1, y, z0:z1 + 1].T.astype(float), [0.5], colors="red", linewidths=1.2)
        ax.set_title(f"slice {y}", fontsize=9)
    name = title or f"{kind} brain mask"
    verdict = "CHECK: " + "; ".join(warnings) if warnings else "no problems found by the automatic checks (still look at the outline)"
    legend = "red = mask" + (", cyan dashed = structural mask before the trimming" if prior is not None else "")
    fig.suptitle(f"{name} ({legend})\n{verdict}", fontsize=10, color="red" if warnings else "black")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=90)
    if show:
        plt.show()
    plt.close(fig)

    summary = [f"{name}: {result['voxels']} voxels ({result['volume_mm3'] / 1000:.2f} cm3), slices {result['slices'][0]}-{result['slices'][1]}, "
               f"{result.get('pieces', 1)} piece(s), edge contrast {result['edge_contrast']:.2f}, {100 * result['border_fraction']:.1f}% at the image border"]
    if "kept_fraction" in result:
        summary.append(f"  kept {100 * result['kept_fraction']:.0f}% of the structural mask after the intensity trimming, centre moved {result['centre_shift_mm']:.2f} mm")
    for line in summary:
        print(line)
    for w in warnings:
        print_statement(f"[WARNING] {w}", bcolors.FAIL)
    print_statement(f"Overlay figure: {out_png}", bcolors.NOTIFICATION)
    return result


def mean_of_middle_volumes(func_file, out_file, n_volumes=30):
    """Mean of up to `n_volumes` volumes around the middle of a 4D series (less noisy than a single volume; the head hardly moves
    within this stretch). Used as the image for the automatic functional mask and its quality check."""
    img = nib.load(str(func_file))
    n = img.shape[-1]
    start = max(0, n // 2 - n_volumes // 2)
    mean = np.asarray(img.dataobj[..., start:min(n, start + n_volumes)], dtype=np.float32).mean(-1)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(mean.shape)
    nib.save(nib.Nifti1Image(mean, img.affine, header), str(out_file))
    return out_file
