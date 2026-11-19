"""Does an automatic brain mask change the functional -> structural coregistration?

For sessions that have hand-drawn masks of both the functional and the structural image, the registration of the pipeline
(3dAllineate, rigid body, nmi, bounded like coregistration_afni in mol_fMRI_v5.7) is run twice: with the hand-drawn masks and with
automatic masks (methods of automask_test.py). The transforms are compared (difference in shift and rotation) and the functional
brain mask is moved to structural space with both transforms and compared with the hand-drawn structural mask (Dice).
All files go to the scratch folder of automask_test.py; no data is changed.

    python scripts/automask_coreg_test.py --n 8 --func-method func_from_struct:skullstrip_rat --struct-method skullstrip_rat
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import automask_test as am  # noqa: E402


def session_pairs():
    """Functional and structural pair (image + hand-drawn mask) from the same subject folder."""
    pairs = am.load_pairs()
    func = [p for p in pairs if p["kind"] == "func" and p["species"] == "rat"]
    struct = [p for p in pairs if p["kind"] == "struct" and p["species"] == "rat"]
    by_subject = {}
    for s in struct:
        by_subject.setdefault(str(Path(s["mask"]).parent.parent), []).append(s)
    out = []
    for f in func:
        candidates = by_subject.get(str(Path(f["mask"]).parent.parent), [])
        if candidates:
            out.append((f, candidates[0]))
    return out


def masked(image, mask, out):
    data, img = am.load3d(image)
    m = np.asarray(nib.load(str(mask)).dataobj).squeeze() > 0
    am.save_like((data * m).astype(np.float32), img, out)
    return out


def register(func_masked, struct_masked, work, tag):
    """Rigid-body registration as in coregistration_afni (nmi, two pass, rotation <= 5 deg, shift <= 2 mm)."""
    mat, par = work / f"{tag}.aff12.1D", work / f"{tag}_params.1D"
    for f in (mat, par):
        if f.exists():
            f.unlink()
    am.sh(["3dAllineate", "-overwrite", "-base", struct_masked, "-source", func_masked, "-prefix", work / f"{tag}_aligned.nii.gz", "-1Dmatrix_save", mat,
           "-1Dparam_save", par, "-warp", "shift_rotate", "-cost", "nmi", "-twopass", "-maxrot", "5", "-maxshf", "2", "-final", "linear"])
    values = [float(v) for line in par.read_text().splitlines() if not line.startswith("#") for v in line.split()]
    return mat, np.array(values[:6])


def to_struct_space(mask_file, struct_ref, mat, out):
    am.sh(["3dAllineate", "-overwrite", "-source", mask_file, "-master", struct_ref, "-1Dmatrix_apply", mat, "-final", "NN", "-prefix", out])
    return np.asarray(nib.load(str(out)).dataobj).squeeze() > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--func-method", default="func_from_struct:skullstrip_rat")
    parser.add_argument("--struct-method", default="skullstrip_rat")
    args = parser.parse_args()

    pairs = session_pairs()
    random.Random(args.seed).shuffle(pairs)
    out_file = am.SCRATCH / "coreg_results.jsonl"
    print(f"{len(pairs)} sessions with both masks; testing {args.n} (functional mask: {args.func_method}, structural mask: {args.struct_method})")
    for func, struct in pairs[: args.n]:
        work = am.SCRATCH / "coreg" / func["id"]
        work.mkdir(parents=True, exist_ok=True)
        row = {"id": func["id"], "modality": func["modality"], "func_method": args.func_method, "struct_method": args.struct_method}
        start = time.time()
        try:
            f_img, s_img = am.load3d(func["image"])[1], am.load3d(struct["image"])[1]
            # automatic masks
            ctx_f = {"kind": "func", "species": "rat", "modality": func["modality"]}
            ctx_s = {"kind": "struct", "species": "rat", "modality": "struct"}
            for name, pair, method, ctx, img in (("func", func, args.func_method, ctx_f, f_img), ("struct", struct, args.struct_method, ctx_s, s_img)):
                d = work / name
                d.mkdir(exist_ok=True)
                data, _ = am.load3d(pair["image"])
                image3d = am.save_like(data, img, d / "image3d.nii.gz")
                if method.startswith("func_from_struct:"):
                    ctx["struct_pair"] = struct
                    auto = am.m_func_from_struct(image3d, d, ctx, method.split(":", 1)[1])
                else:
                    auto = am.METHODS[method](image3d, d, ctx)
                am.save_like(auto.astype(np.uint8), img, work / f"{name}_auto_mask.nii.gz")
            hand_struct = np.asarray(nib.load(struct["mask"]).dataobj).squeeze() > 0
            ys = np.where(hand_struct.any(axis=(0, 2)))[0]
            results = {}
            for cond, fm, sm in (("hand", func["mask"], struct["mask"]), ("auto", work / "func_auto_mask.nii.gz", work / "struct_auto_mask.nii.gz")):
                fmasked = masked(func["image"], fm, work / f"{cond}_func_masked.nii.gz")
                smasked = masked(struct["image"], sm, work / f"{cond}_struct_masked.nii.gz")
                mat, params = register(fmasked, smasked, work, cond)
                moved = to_struct_space(func["mask"], struct["image"], mat, work / f"{cond}_hand_func_mask_in_struct.nii.gz")
                slab = np.zeros(moved.shape, bool)
                slab[:, ys.min():ys.max() + 1, :] = True
                results[cond] = {"params": params, "dice_struct": am.dice(moved & slab, hand_struct & slab)}
            diff = results["auto"]["params"] - results["hand"]["params"]
            row |= {"shift_diff_mm": float(np.linalg.norm(diff[:3])), "rot_diff_deg": float(np.linalg.norm(diff[3:6])),
                    "dice_struct_hand_masks": results["hand"]["dice_struct"], "dice_struct_auto_masks": results["auto"]["dice_struct"],
                    "hand_params": results["hand"]["params"].round(3).tolist(), "auto_params": results["auto"]["params"].round(3).tolist()}
        except Exception as e:
            row["error"] = str(e)[-300:]
        row["seconds"] = round(time.time() - start, 1)
        with open(out_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {func['id'][:50]:50s} " + (f"transform differs by {row['shift_diff_mm']:.2f} mm / {row['rot_diff_deg']:.2f} deg; mask overlap in struct space: "
              f"hand masks {row['dice_struct_hand_masks']:.2f}, auto masks {row['dice_struct_auto_masks']:.2f}" if "shift_diff_mm" in row else f"ERROR {row['error'][-150:]!r}"), flush=True)


if __name__ == "__main__":
    main()
