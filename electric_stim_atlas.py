"""Atlas registration helpers for the electrical stimulation analysis notebooks.

Follows roi_atlas_analysis.py: the structural scan is registered to the functional mean (struct2func, rigid) and to the
atlas template (struct2atlas). The two matrices are composed into one functional -> atlas matrix, and the multi-label
atlas is resampled once onto the functional grid with nearest-neighbour interpolation, so the functional data itself is
never interpolated. Finished registrations in atlas_dir are reused.
"""
import re
import subprocess
from pathlib import Path

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


def register(source, base, prefix, cost, cmass, warp, atlas_dir):
    matrix = Path(atlas_dir) / f"{prefix}.aff12.1D"
    aligned = Path(atlas_dir) / f"{prefix}_aligned.nii.gz"
    if matrix.exists() and aligned.exists():
        print_statement(f"{prefix} registration already exists. Skipping.", bcolors.OKGREEN)
        return matrix
    cmd = ["3dAllineate", "-source", source, "-base", base, "-prefix", aligned, "-1Dmatrix_save", matrix,
           "-1Dparam_save", Path(atlas_dir) / f"{prefix}_params.1D", "-warp", warp, "-cost", cost,
           "-source_automask+2", "-twopass", "-twobest", "MAX", "-float", "-verb", "-overwrite"]
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


def register_to_atlas(struct_file, func_reference, atlas_dir, species_cfg, atlas_template, atlas_labels, atlas_mask=""):
    """Registers structural -> functional and structural -> atlas, composes the functional -> atlas matrix and
    resamples the atlas labels onto the functional grid. Returns (composite matrix, labels in functional space)."""
    atlas_dir = Path(atlas_dir)
    struct2func = register(struct_file, func_reference, "struct2func", "nmi", None, "shift_rotate", atlas_dir)
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
