"""Test of automatic brain masks against the hand-drawn masks of earlier sessions (brain mask of the functional image and
of the structural image). Nothing here changes any data: all results go to a scratch folder (default /tmp/automask_test).

    python scripts/automask_test.py inventory                       # find image / hand-drawn mask pairs in AnalysedData
    python scripts/automask_test.py evaluate --kind struct --methods threshold,atlas_affine --n 6
    python scripts/automask_test.py report                          # table of Dice / edge distance per method

Every method gets one 3D image and returns a boolean mask on the same grid. The score is the Dice overlap and the mean
surface distance (mm) to the hand-drawn mask; both are written to results.jsonl and summarised by `report`.
Needs FSL (bet) and AFNI (3dAllineate, 3dSkullStrip, 3dAutomask) on the PATH, plus nibabel, numpy, scipy.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fmri_paths import get_fmri_root  # noqa: E402

FMRI_ROOT = Path(get_fmri_root())
SCRATCH = Path(os.environ.get("AUTOMASK_SCRATCH", "/tmp/automask_test"))
HAND_MASKS = {"func": ("mask_mean_mc_func.nii.gz",), "struct": ("mask_struct.nii.gz", "struct_mask.nii.gz")}


# ----------------------------------------------------------------------------------------------------------------
# data set
# ----------------------------------------------------------------------------------------------------------------
def species_of(path):
    text = str(path).lower()
    return "mouse" if any(k in text for k in ("bl6", "mouse", "c57")) else "rat"


def find_pairs():
    """(image, hand-drawn mask) pairs: functional = mean_mc_func / middle_vol next to mask_mean_mc_func; structural = struct.nii.gz."""
    pairs = []
    analysed = FMRI_ROOT / "AnalysedData"
    for kind, names in HAND_MASKS.items():
        for name in names:
            for mask in analysed.glob(f"*/**/{name}"):
                if mask.name.startswith("._") or any(p.startswith("._") for p in mask.parts):
                    continue
                folder = mask.parent
                if kind == "struct":
                    candidates = [folder / "struct.nii.gz"]
                    modality = "struct"
                else:
                    sub = sorted(d for d in folder.iterdir() if d.is_dir() and not d.name.startswith("."))
                    candidates = [d / "mean_mc_func.nii.gz" for d in sub] + [folder / "middle_vol.nii.gz"]
                    modality = "sordino" if "sordino" in str(folder).lower() else "epi"
                image = next((c for c in candidates if c.exists()), None)
                if image is None:
                    continue
                try:
                    shape_i, shape_m = nib.load(image).shape[:3], nib.load(mask).shape[:3]
                except Exception:
                    continue
                if shape_i != shape_m:
                    continue
                pairs.append({"kind": kind, "modality": modality, "species": species_of(mask), "image": str(image), "mask": str(mask),
                              "id": "_".join(mask.relative_to(analysed).parts[-4:-1]).replace(" ", "")})
    return pairs


def load_pairs():
    cache = SCRATCH / "pairs.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pairs = find_pairs()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pairs, indent=1))
    return pairs


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def sh(cmd, **kwargs):
    return subprocess.run([str(c) for c in cmd], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)


def load3d(path):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data.mean(-1)
    return data.squeeze(), img


def save_like(array, ref_img, path):
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32 if array.dtype != bool else np.uint8)
    header.set_data_shape(array.shape)
    nib.save(nib.Nifti1Image(array.astype(np.float32 if array.dtype != bool else np.uint8), ref_img.affine, header), str(path))
    return path


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    return lab == (1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1)))))


def fill_slicewise(mask):
    out = mask.copy()
    for y in range(mask.shape[1]):
        out[:, y, :] = ndi.binary_fill_holes(mask[:, y, :])
    return ndi.binary_fill_holes(out)


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def surface_distance_mm(a, b, zooms):
    """Mean symmetric distance (mm) between the surfaces of two masks."""
    if not a.any() or not b.any():
        return float("nan")
    ea, eb = a & ~ndi.binary_erosion(a), b & ~ndi.binary_erosion(b)
    d_ab = ndi.distance_transform_edt(~eb, sampling=zooms)[ea].mean()
    d_ba = ndi.distance_transform_edt(~ea, sampling=zooms)[eb].mean()
    return float((d_ab + d_ba) / 2)


def score(auto, hand, zooms):
    """Overlap with the hand-drawn mask, over the whole volume and inside the slab of slices (axis 1) that the hand-drawn mask covers.
    The hand-drawn masks only cover the slices of the functional coverage, so the slab score is the one that matters."""
    ys = np.where(hand.any(axis=(0, 2)))[0]
    slab = np.zeros(hand.shape, bool)
    slab[:, ys.min():ys.max() + 1, :] = True
    a, h = auto & slab, hand & slab
    return {"dice_full": dice(auto, hand), "dice": dice(a, h), "surface_mm": surface_distance_mm(a, h, zooms),
            "auto_voxels": int(a.sum()), "volume_ratio": float(a.sum() / max(h.sum(), 1)), "slab_slices": int(len(ys))}


# ----------------------------------------------------------------------------------------------------------------
# atlas files
# ----------------------------------------------------------------------------------------------------------------
def atlas_files(species):
    """Template and brain mask of the species atlas (the same atlases as electric_stim_atlas.py)."""
    import electric_stim_atlas as atlas
    cfg = atlas.species_atlas_config(str(FMRI_ROOT))[species]
    if species == "rat":
        folder = Path(cfg["template"]).parent
        return {"template": folder / "SIGMA_InVivo_Brain_Template.nii", "template_masked": Path(cfg["template"]), "mask": folder / "SIGMA_InVivo_Brain_Mask.nii"}
    return {"template": Path(cfg["template"]), "template_masked": Path(cfg["template"]), "mask": Path(cfg["mask"])}


# ----------------------------------------------------------------------------------------------------------------
# methods: f(image3d_path, workdir, ctx) -> boolean mask on the grid of the image
# ----------------------------------------------------------------------------------------------------------------
def m_threshold(image, work, ctx):
    """Intensity: smoothed image above a fraction of the brain level, largest component, holes filled (slice-wise for struct)."""
    data, img = load3d(image)
    zooms = img.header.get_zooms()[:3]
    if ctx["kind"] == "struct":
        # the slices are thick: smooth and open in-plane only
        sm = ndi.gaussian_filter(data, (0.15 / zooms[0], 0, 0.15 / zooms[2]))
        ref = np.percentile(sm, 99.5)
        out = np.zeros(data.shape, bool)
        for y in range(data.shape[1]):
            m = sm[:, y, :] > 0.35 * ref
            m = ndi.binary_opening(m, iterations=max(1, int(round(0.3 / zooms[0]))))
            lab, n = ndi.label(m)
            if n:
                sizes = ndi.sum(m, lab, range(1, n + 1))
                out[:, y, :] = ndi.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
        return out
    sm = ndi.gaussian_filter(data, 1.0)
    ref = np.median(sm[sm > 0.5 * np.percentile(sm, 99.5)])
    m = largest_component(ndi.binary_opening(sm > 0.7 * ref, iterations=2))
    return ndi.binary_erosion(fill_slicewise(m), iterations=3)


def m_afni_automask(image, work, ctx):
    """AFNI 3dAutomask with the settings of the earlier attempt (mol_fMRI_v6_linux_automasks: -clfrac 0.6 -peels 4)."""
    out = work / "automask.nii.gz"
    sh(["3dAutomask", "-overwrite", "-clfrac", "0.6", "-peels", "4", "-prefix", out, image])
    return np.asarray(nib.load(str(out)).dataobj).squeeze() > 0


def m_skullstrip_rat(image, work, ctx):
    """AFNI 3dSkullStrip -rat on the image resampled to isotropic voxels (the rat mode expects ~isotropic data)."""
    data, img = load3d(image)
    zooms = img.header.get_zooms()[:3]
    iso = work / "iso.nii.gz"
    sh(["3dresample", "-overwrite", "-dxyz", min(zooms), min(zooms), min(zooms), "-rmode", "Cu", "-prefix", iso, "-input", image])
    out = work / "ss.nii.gz"
    sh(["3dSkullStrip", "-overwrite", "-input", iso, "-prefix", out, "-rat", "-push_to_edge"])
    sh(["3dresample", "-overwrite", "-master", image, "-rmode", "NN", "-prefix", work / "ss_back.nii.gz", "-input", out])
    return np.asarray(nib.load(str(work / "ss_back.nii.gz")).dataobj).squeeze() > 0


def m_bet(image, work, ctx):
    """FSL bet; the voxel sizes are scaled by 10 so that a rodent brain has the size bet expects."""
    data, img = load3d(image)
    scaled = work / "bet_in.nii.gz"
    aff = img.affine.copy()
    aff[:3, :3] *= 10
    nib.save(nib.Nifti1Image(data, aff), str(scaled))
    out = work / "bet_out"
    sh(["bet", scaled, out, "-m", "-R", "-f", "0.4"])
    return np.asarray(nib.load(str(out) + "_mask.nii.gz").dataobj).squeeze() > 0


def _register_template(image, work, ctx, cost="nmi", warp="affine_general", tag="atlas"):
    """Register the species template (source, restricted to its brain) to the image and move the template brain mask along."""
    files = atlas_files(ctx["species"])
    mat = work / f"{tag}.aff12.1D"
    for f in (mat, work / f"{tag}_reg.nii.gz"):
        if f.exists():
            f.unlink()
    sh(["3dAllineate", "-overwrite", "-base", image, "-source", files["template"], "-source_mask", files["mask"], "-prefix", work / f"{tag}_reg.nii.gz",
        "-1Dmatrix_save", mat, "-warp", warp, "-cost", cost, "-twopass", "-cmass", "-final", "linear"])
    sh(["3dAllineate", "-overwrite", "-source", files["mask"], "-master", image, "-1Dmatrix_apply", mat, "-final", "NN", "-prefix", work / f"{tag}_mask.nii.gz"])
    return np.asarray(nib.load(str(work / f"{tag}_mask.nii.gz")).dataobj).squeeze() > 0


def m_atlas_affine_nmi(image, work, ctx):
    return _register_template(image, work, ctx, cost="nmi", warp="affine_general", tag="atlas_nmi")


def m_atlas_affine_lpa(image, work, ctx):
    return _register_template(image, work, ctx, cost="lpa+ZZ", warp="affine_general", tag="atlas_lpa")


def m_atlas_refined(image, work, ctx):
    """Atlas mask (affine, nmi) used as a prior: the dilated prior is cut with an intensity threshold computed inside the prior."""
    prior = _register_template(image, work, ctx, cost="nmi", warp="affine_general", tag="atlas_nmi")
    data, img = load3d(image)
    zooms = img.header.get_zooms()[:3]
    if not prior.any():
        return prior
    sm = ndi.gaussian_filter(data, 0.15 / min(zooms) if ctx["kind"] == "struct" else 1.0)
    inside = sm[prior]
    thr = 0.35 * np.percentile(inside, 90)
    grown = ndi.binary_dilation(prior, iterations=max(1, int(round(0.5 / min(zooms)))))
    keep = grown & (sm > thr)
    return fill_slicewise(largest_component(ndi.binary_opening(keep, iterations=1)))


def struct_for(pair, pairs):
    """Structural pair (image + hand-drawn mask) of the same subject folder as a functional pair, or None."""
    subject = str(Path(pair["mask"]).parent.parent)
    return next((s for s in pairs if s["kind"] == "struct" and str(Path(s["mask"]).parent.parent) == subject), None)


def m_func_from_struct(image, work, ctx, source):
    """Functional brain mask from the structural mask of the same session. Both scans share the scanner coordinates, so the structural
    mask is moved to the functional grid through the headers only (no registration), then trimmed with an intensity threshold inside it
    (relative to the median inside the mask). `source` = "hand" (the hand-drawn structural mask, an upper bound) or the name of an
    automatic structural method (its masks are cached in the scratch folder)."""
    struct = ctx["struct_pair"]
    if struct is None:
        raise ValueError("no structural scan with a hand-drawn mask in the same subject folder")
    if source == "hand":
        struct_mask = struct["mask"]
    else:
        cache = SCRATCH / "struct_auto" / source / (struct["id"] + ".nii.gz")
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            data, img = load3d(struct["image"])
            sdir = SCRATCH / "work" / struct["id"] / source
            sdir.mkdir(parents=True, exist_ok=True)
            image3d = save_like(data, img, sdir / "image3d.nii.gz")
            save_like(METHODS[source](image3d, sdir, {"kind": "struct", "species": ctx["species"], "modality": "struct"}).astype(np.uint8), img, cache)
        struct_mask = cache
    identity = SCRATCH / "identity.aff12.1D"
    identity.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
    moved = work / "struct_mask_in_func.nii.gz"
    sh(["3dAllineate", "-overwrite", "-source", struct_mask, "-master", image, "-1Dmatrix_apply", identity, "-final", "NN", "-prefix", moved])
    prior = np.asarray(nib.load(str(moved)).dataobj).squeeze() > 0
    if not prior.any():
        return prior
    data, _ = load3d(image)
    sm = ndi.gaussian_filter(data, 1.0)
    return fill_slicewise(prior & (sm > 0.5 * np.median(sm[prior])))


METHODS = {"threshold": m_threshold, "afni_automask": m_afni_automask, "skullstrip_rat": m_skullstrip_rat, "bet": m_bet,
           "atlas_affine_nmi": m_atlas_affine_nmi, "atlas_affine_lpa": m_atlas_affine_lpa, "atlas_refined": m_atlas_refined}


# ----------------------------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------------------------
def evaluate(args):
    all_pairs = load_pairs()
    pairs = [p for p in all_pairs if p["kind"] == args.kind and (args.species is None or p["species"] == args.species)
             and (args.modality is None or p["modality"] == args.modality)]
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]
    results = SCRATCH / "results.jsonl"
    print(f"{len(pairs)} sessions, methods {args.methods}")
    for pair in pairs:
        hand = np.asarray(nib.load(pair["mask"]).dataobj).squeeze() > 0
        zooms = nib.load(pair["image"]).header.get_zooms()[:3]
        for method in args.methods:
            work = SCRATCH / "work" / pair["id"] / method
            work.mkdir(parents=True, exist_ok=True)
            ctx = {"kind": pair["kind"], "species": pair["species"], "modality": pair["modality"]}
            if pair["kind"] == "func":
                ctx["struct_pair"] = struct_for(pair, all_pairs)
            start = time.time()
            row = {k: pair[k] for k in ("id", "kind", "modality", "species")} | {"method": method, "hand_voxels": int(hand.sum())}
            try:
                data, img = load3d(pair["image"])
                image3d = work / "image3d.nii.gz"
                save_like(data, img, image3d)
                if method.startswith("func_from_struct:"):
                    auto = m_func_from_struct(image3d, work, ctx, method.split(":", 1)[1])
                else:
                    auto = METHODS[method](image3d, work, ctx)
                save_like(auto.astype(np.uint8), img, work / "auto_mask.nii.gz")
                row |= score(auto, hand, zooms)
            except Exception as e:
                row |= {"error": str(e)[-300:]}
            row["seconds"] = round(time.time() - start, 1)
            with open(results, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {pair['id'][:45]:45s} {method:18s} " + (f"dice {row['dice']:.2f}  dist {row['surface_mm']:.2f} mm  vol x{row['volume_ratio']:.2f}  {row['seconds']}s"
                                                          if "dice" in row else f"ERROR {row['error'][-120:]!r}"), flush=True)


def rescore(args):
    pairs = {p["id"]: p for p in load_pairs()}
    rows = [json.loads(l) for l in (SCRATCH / "results.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        auto_file = SCRATCH / "work" / r["id"] / r["method"] / "auto_mask.nii.gz"
        if r["id"] in pairs and auto_file.exists():
            p = pairs[r["id"]]
            hand = np.asarray(nib.load(p["mask"]).dataobj).squeeze() > 0
            auto = np.asarray(nib.load(str(auto_file)).dataobj).squeeze() > 0
            r = {k: v for k, v in r.items() if k not in ("dice", "surface_mm", "auto_voxels", "volume_ratio", "error")} | score(auto, hand, nib.load(p["image"]).header.get_zooms()[:3])
        out.append(r)
    (SCRATCH / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(out)} results rescored")


def report(args):
    rows = [json.loads(l) for l in (SCRATCH / "results.jsonl").read_text().splitlines()]
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"] if r["kind"] == "struct" else r["modality"], r["species"], r["method"]), []).append(r)
    print(f"{'image':8s} {'species':7s} {'method':18s} {'n':>3s} {'failed':>6s} {'Dice slab':>9s} {'min':>5s} {'dist mm':>8s} {'vol ratio':>9s}")
    for (kind, species, method), rs in sorted(groups.items()):
        ok = [r for r in rs if "dice" in r]
        if ok:
            print(f"{kind:8s} {species:7s} {method:18s} {len(rs):3d} {len(rs) - len(ok):6d} {np.mean([r['dice'] for r in ok]):9.2f} {min(r['dice'] for r in ok):5.2f} "
                  f"{np.nanmean([r['surface_mm'] for r in ok]):8.2f} {np.mean([r['volume_ratio'] for r in ok]):9.2f}")
        else:
            print(f"{kind:8s} {species:7s} {method:18s} {len(rs):3d} {len(rs):6d}  (all failed)")


def inventory(args):
    if args.refresh:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        pairs = find_pairs()
        (SCRATCH / "pairs.json").write_text(json.dumps(pairs, indent=1))
    else:
        pairs = load_pairs()
    counts = {}
    for p in pairs:
        counts[(p["kind"], p["modality"], p["species"])] = counts.get((p["kind"], p["modality"], p["species"]), 0) + 1
    for k, v in sorted(counts.items()):
        print(v, k)
    print(f"{len(pairs)} pairs, saved in {SCRATCH / 'pairs.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory"); p.add_argument("--refresh", action="store_true"); p.set_defaults(func=inventory)
    p = sub.add_parser("evaluate")
    p.add_argument("--kind", choices=["func", "struct"], required=True)
    p.add_argument("--methods", type=lambda s: s.split(","), default=list(METHODS))
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--species", choices=["rat", "mouse"])
    p.add_argument("--modality", choices=["epi", "sordino", "struct"])
    p.set_defaults(func=evaluate)
    p = sub.add_parser("rescore"); p.set_defaults(func=rescore)
    p = sub.add_parser("report"); p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
