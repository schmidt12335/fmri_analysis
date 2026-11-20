"""Builds the mouse atlas files of the pipeline: Allen CCFv3 regions on the DevCCF P56 MRI template.

The animals are registered to the DevCCF P56 T2-weighted MRI template (Kronman et al. 2024, Nat Commun 15:9072,
https://doi.org/10.6084/m9.figshare.25714965). DevCCF ships the Allen CCFv3 template and the Allen CCFv3 annotation warped
into the space of this MRI template, so after an MRI-to-MRI registration everything is in Allen regions. This script makes,
in the atlas folder (default: <fmri_root>/atlas/Allen_DevCCF_P56):

    devccf_t2_50um.nii.gz              DevCCF P56 T2w MRI template (registration target)
    devccf_brainmask_50um.nii.gz       its brain mask
    allen_summary_labels_50um.nii.gz   Allen "summary structures" (316 grey matter regions, structure set 167587189, cortical
                                       layers merged), split at the midline: right = 1..316, left = 1001..1316; fibre tracts and
                                       ventricles are 0. On the 50 µm grid of the MRI template.
    allen_summary_labels.csv           id, name (with _L / _R), acronym, hemisphere, allen_id, color
    allen_ccfv3_template_50um.nii.gz   Allen CCFv3 average template (two-photon autofluorescence), 50 µm, same grid; used as
                                       anatomy in the 3D figures (plot_slice_stack.py)

All four share one grid. The DevCCF files have their coordinate origin in a corner of the image; here the origin is moved so
that the centre of the brain is at 0 mm (as in DSURQE and roughly in the animal scans), because the registration only corrects
large shifts left-right and up-down by the centre of mass. The image data is not changed.

The downloaded DevCCF files (P56_MRI-T2_50um, P56_MRI-MaskBrain_50um, P56_CCFv3_20um, P56_CCFv3_Annotation_20um,
CCFv3_OntologyStructure_u16.xlsx) must be in the same folder; `--download` fetches them (about 200 MB, needs `remotezip`).
Run once:  python tools/build_allen_atlas.py [--atlas-dir DIR] [--download]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

DEVCCF_ZIP = "https://ndownloader.figshare.com/files/45984852"
DEVCCF_FILES = ["CCFv3_OntologyStructure_u16.xlsx", "READ_ME.txt", "P56/P56_MRI-T2_50um.nii.gz", "P56/P56_MRI-MaskBrain_50um.nii.gz",
                "P56/P56_CCFv3_20um.nii.gz", "P56/P56_CCFv3_Annotation_20um.nii.gz"]
SUMMARY_STRUCTURES_URL = ("http://api.brain-map.org/api/v2/data/query.json?criteria=model::Structure,rma::criteria,"
                          "%5Bgraph_id$eq1%5D,structure_sets%5Bid$eq167587189%5D,"
                          "rma::options%5Bnum_rows$eqall%5D%5Border$eq'structures.graph_order'%5D")
LEFT_OFFSET = 1000


def download(atlas_dir):
    try:
        from remotezip import RemoteZip
    except ImportError:
        sys.exit("--download needs remotezip: pip install remotezip (or download DevCCFv1.zip from figshare and unpack the P56 files)")
    with RemoteZip(DEVCCF_ZIP) as z:
        for name in DEVCCF_FILES:
            target = atlas_dir / Path(name).name
            if not target.exists():
                print(f"Downloading {name} ...")
                target.write_bytes(z.read(f"DevCCFv1/{name}"))


def summary_structures(atlas_dir):
    cache = atlas_dir / "allen_summary_structures.json"
    if not cache.exists():
        with urllib.request.urlopen(SUMMARY_STRUCTURES_URL) as response:
            cache.write_bytes(response.read())
    rows = json.loads(cache.read_text())["msg"]
    if len(rows) != 316:
        raise RuntimeError(f"Expected 316 summary structures, got {len(rows)}")
    return rows


def midline_x(t2, mask):
    """x voxel position of the midline: the mirror axis at which the template matches its own mirror image best."""
    profile = np.where(mask, t2, 0).astype(np.float32)
    lo, hi = np.where(mask.any(axis=(1, 2)))[0][[0, -1]]
    best, best_score = (lo + hi) / 2, -np.inf
    idx = np.arange(profile.shape[0])
    for centre2 in np.arange(lo + hi - 20, lo + hi + 21):  # mirror axis at centre2 / 2, within 10 voxels of the middle of the mask
        mirrored = centre2 - idx
        ok = (mirrored >= 0) & (mirrored < profile.shape[0])
        score = np.corrcoef(profile[idx[ok]].ravel(), profile[mirrored[ok]].ravel())[0, 1]
        if score > best_score:
            best, best_score = centre2 / 2, score
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas-dir", help="default: <fmri_root>/atlas/Allen_DevCCF_P56")
    p.add_argument("--download", action="store_true", help="download the DevCCF P56 files first")
    a = p.parse_args()
    if a.atlas_dir:
        atlas_dir = Path(a.atlas_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from fmri_paths import get_fmri_root
        atlas_dir = Path(get_fmri_root()) / "atlas" / "Allen_DevCCF_P56"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    if a.download:
        download(atlas_dir)

    t2_img = nib.load(atlas_dir / "P56_MRI-T2_50um.nii.gz")
    t2 = np.asarray(t2_img.dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(atlas_dir / "P56_MRI-MaskBrain_50um.nii.gz").dataobj) > 0.5
    ann_img = nib.load(atlas_dir / "P56_CCFv3_Annotation_20um.nii.gz")
    if nib.aff2axcodes(t2_img.affine) != nib.aff2axcodes(ann_img.affine):
        raise RuntimeError("The MRI template and the annotation have different orientations.")

    # Allen ontology: 16-bit annotation values -> Allen ids -> summary structure (first summary ancestor)
    onto = pd.read_excel(atlas_dir / "CCFv3_OntologyStructure_u16.xlsx")
    summary = summary_structures(atlas_dir)
    summary_index = {int(s["id"]): k + 1 for k, s in enumerate(summary)}
    lut = np.zeros(int(onto["id_16bit"].max()) + 1, dtype=np.int16)
    for _, row in onto.iterrows():
        path = [int(x) for x in str(row["structure_id_path"]).strip("/").split("/") if x]
        hit = next((summary_index[i] for i in reversed(path) if i in summary_index), 0)
        lut[int(row["id_16bit"])] = hit

    # Annotation (20 µm) -> summary index on the 50 µm MRI grid, nearest neighbour at the voxel centres
    ann = np.asarray(ann_img.dataobj, dtype=np.float32)
    to_ann = np.linalg.inv(ann_img.affine) @ t2_img.affine
    coords = np.indices(t2.shape, dtype=np.float32).reshape(3, -1)
    coords = (to_ann[:3, :3] @ coords + to_ann[:3, 3:4]).reshape((3, *t2.shape))
    values = ndimage.map_coordinates(ann, coords, order=0, mode="constant", cval=0).astype(np.int64)
    del ann
    labels = lut[np.clip(values, 0, len(lut) - 1)]
    labels[values >= len(lut)] = 0

    # Hemispheres: +x is right in this template (axis codes R, I, A)
    if nib.aff2axcodes(t2_img.affine)[0] != "R":
        raise RuntimeError("Expected the first axis of the template to point to the right.")
    mid = midline_x(t2, mask)
    x = np.arange(t2.shape[0])[:, None, None]
    labels = np.where((labels > 0) & (x < mid), labels + LEFT_OFFSET, labels).astype(np.int16)
    print(f"Midline at x voxel {mid:.1f} of {t2.shape[0]}")

    # Centred grid: the brain centre at 0 mm
    affine = t2_img.affine.copy()
    affine[:3, 3] -= (t2_img.affine @ np.r_[np.argwhere(mask).mean(axis=0), 1])[:3]

    def save(data, dtype, name):
        header = t2_img.header.copy()
        header.set_data_dtype(dtype)
        img = nib.Nifti1Image(data.astype(dtype), affine, header)
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
        nib.save(img, atlas_dir / name)

    save(t2, np.float32, "devccf_t2_50um.nii.gz")
    save(mask.astype(np.uint8), np.uint8, "devccf_brainmask_50um.nii.gz")
    save(labels, np.int16, "allen_summary_labels_50um.nii.gz")
    rows = []
    for k, s in enumerate(summary, start=1):
        for side, offset in (("R", 0), ("L", LEFT_OFFSET)):
            rows.append({"id": k + offset, "name": f"{s['name']}_{side}", "acronym": s["acronym"], "hemisphere": side,
                         "allen_id": s["id"], "color": "#" + s["color_hex_triplet"],
                         "n_voxels_50um": int((labels == k + offset).sum())})
    pd.DataFrame(rows).to_csv(atlas_dir / "allen_summary_labels.csv", index=False)
    missing = [r["name"] for r in rows if r["n_voxels_50um"] == 0]
    print(f"[OK] {len(rows)} regions, {len(missing)} without voxels{': ' + ', '.join(missing[:6]) if missing else ''}")

    # Allen template (20 µm) -> 50 µm on the MRI grid, averaged over the larger voxel
    ccf_img = nib.load(atlas_dir / "P56_CCFv3_20um.nii.gz")
    ccf = np.asarray(ccf_img.dataobj, dtype=np.float32)
    ccf = ndimage.gaussian_filter(ccf, 2.5 / np.sqrt(12))  # same width as a 50 µm box in 20 µm voxels (anti-aliasing)
    to_ccf = np.linalg.inv(ccf_img.affine) @ t2_img.affine
    coords = np.indices(t2.shape, dtype=np.float32).reshape(3, -1)
    coords = (to_ccf[:3, :3] @ coords + to_ccf[:3, 3:4]).reshape((3, *t2.shape))
    template = ndimage.map_coordinates(ccf, coords, order=1, mode="constant", cval=0).astype(np.float32)
    save(template, np.float32, "allen_ccfv3_template_50um.nii.gz")
    print(f"[OK] Files written to {atlas_dir}")


if __name__ == "__main__":
    main()
