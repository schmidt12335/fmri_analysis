"""
Stitching multiple electrical stimulation functional scans (same session,
same subject, back-to-back, unchanged acquisition parameters) into a single
continuous 4D timeseries.

Rationale / pipeline position
------------------------------
Stitch at the RAW level (before motion correction), then run motion
correction ONCE on the concatenated series. Reasoning:

  * AFNI 3dvolreg registers every volume independently to a single base/
    reference volume - it does not chain estimates between neighbouring
    timepoints. So there is no "continuity" problem with concatenating raw
    runs before running volreg: every volume in the whole stitched series,
    regardless of which original scan it came from, gets registered to the
    exact same reference.
  * The alternative - motion-correcting each scan separately (3 independent
    volreg runs, each against its own scan's middle volume) - actually
    introduces a subtler problem: each scan's own reference volume is an
    arbitrary snapshot of the animal at that moment (breathing phase, slow
    physiological drift), so the "zero position" after correction can differ
    slightly between scans even with no repositioning between them. Simply
    concatenating 3 independently-corrected scans can leave a small seam at
    each boundary.
  * Concatenating raw and correcting once removes that source of error
    entirely, since there is only one reference for the whole session.

This approach assumes the 3 scans share identical acquisition parameters,
matrix size, and orientation, and were acquired without repositioning the
animal (true back-to-back scans) - which is the case this script is written
for. If instead there was a gap or repositioning between scans, use
--already_motion_corrected (see below) and consider an explicit inter-scan
registration step, since a single motion-correction reference would no
longer be valid across a physical repositioning.

Downstream steps (masking, temporal/spatial smoothing, signal-change-map,
baseline/signal window selection) must not blend across scan boundaries -
this script writes scan_boundaries.csv recording the volume range of each
original scan within the stitched series so those steps can respect it.

Which input files to choose
----------------------------
The scans being stitched together are the repeated electrical-stimulation
functional runs collected back-to-back within ONE session for ONE subject
(e.g. 3 separate func scans acquired one after another without moving the
animal). Do not mix scans from different sessions/subjects, and do not
include structural/anatomical scans.

Default path (no --already_motion_corrected flag) - pass the RAW, PRE-motion
correction functional data for each scan, in acquisition order. Two kinds of
input are accepted for each scan, and they can be mixed freely in the same
--scans list / same popup selection:

  1) An already-converted NIfTI file, e.g. 'G1_cp_resampled.nii.gz' (see
     data_analysis.py / bruker_to_nifti()) or 'func.nii.gz', found in each
     scan's own analysed-data folder
     (e.g. .../<scan_number><SequenceName>/G1_cp_resampled.nii.gz).
  2) A RAW Bruker ParaVision scan folder straight off the scanner - i.e.
     before running any conversion notebook/script. This is a numbered
     subfolder of the study/subject raw-data directory, e.g.:
         <study_root>/10/            <- pass THIS folder as the scan input
             method                  <- required marker file, must exist
             acqp
             pdata/
                 1/
                     2dseq
                     ...
     Point at the '10' folder itself (not the study_root, and not the
     pdata/1 subfolder). The script auto-detects this case (a directory
     containing a 'method' file) and converts it to NIfTI for you using the
     same steps as bruker_to_nifti() in mol_fMRI_v5.3L_newcoreg.ipynb: run
     `brkraw tonii` on the study root for that scan number, merge echoes
     with fslmerge if the scan is multi-echo (PVM_NEchoImages > 1 in the
     method file), then resample to LPI orientation with AFNI. The
     converted file is written to <out_dir>/converted_raw/scan_<N>/ and
     that path is used for stitching in place of the raw folder.

  * Do NOT pass an already motion-corrected file (e.g. mc_func.nii.gz) here -
    motion correction is meant to run exactly once, on the concatenated raw
    series, so that all scans share a single reference volume.
  * Do NOT pass a structural/anatomical scan folder, the study_root itself,
    or a pdata subfolder - only a single numbered scan folder (or its
    already-converted NIfTI) per entry.
  * All scans must have identical acquisition parameters (matrix size, voxel
    size, orientation, number of slices, TR) since they are concatenated
    directly with fslmerge before any resampling.

Fallback path (--already_motion_corrected flag set) - pass each scan's
ALREADY motion-corrected functional file instead (e.g. mc_func.nii.gz, the
output of running motion_correction()/3dvolreg on that scan alone). Use this
path only when the scans were NOT acquired back-to-back (there was a gap or
the animal was repositioned between scans), since in that case a single
motion-correction reference volume across all scans would not be valid. Add
--align if the scans additionally need to be spatially registered to each
other (e.g. after repositioning) before concatenation.

Usage
-----
python electric_stim_stitching.py \
    --scans /path/scan1/func.nii.gz /path/scan2/func.nii.gz /path/scan3/func.nii.gz \
    --out_dir /path/stitched_output

If --scans and/or --out_dir are omitted, popup dialogs open instead: one to
add/remove/reorder the scan files to stitch, and one to pick the output
directory.

Outputs (written into --out_dir)
---------------------------------
  stitched_raw_func.nii.gz   - concatenated raw (pre-motion-correction) 4D timeseries
  middle_vol.nii.gz          - reference volume used for motion correction
  mc_stitched_func.nii.gz    - motion-corrected, stitched 4D timeseries (main output)
  motion.1D                  - motion correction parameters for the whole stitched series
  scan_boundaries.csv        - start/end volume index (0-based, inclusive) of each
                               original scan within the stitched series

Fallback: --already_motion_corrected
-------------------------------------
If the 3 scans were NOT acquired back-to-back (gap / repositioning between
them) and you only have each scan's independently motion-corrected
mc_func.nii.gz, pass --already_motion_corrected to skip raw concatenation +
single-pass motion correction and instead just concatenate the already
motion-corrected files directly. Add --align on top of that to additionally
register each scan's mean image onto scan 1's mean image (AFNI 3dAllineate,
same-modality least-squares cost) before merging - always visually check
aligned_scan{N} against mean_scan1 in fsleyes before trusting this.
"""

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

from nipype.interfaces import afni, fsl

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:
    tk = None
    filedialog = None
    messagebox = None

# Force compressed NIfTI output regardless of the shell's FSLOUTPUTTYPE setting.
# If left unset (or set to e.g. "NIFTI"), fslmerge/fslroi/fslmaths would silently
# write uncompressed .nii files, breaking the .nii.gz paths this script assumes.
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    NOTIFICATION = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message, color):
    line = "*" * 134
    print()
    print(f"{color}{line}{bcolors.ENDC}")
    print(f"{color}{message.center(len(line))}{bcolors.ENDC}")
    print(f"{color}{line}{bcolors.ENDC}")
    print()


def print_statement(message, color):
    print(f"{color}{message}{bcolors.ENDC}")


def choose_scans_via_dialog(already_motion_corrected=False):
    """Popup letting the user pick the scan files to stitch and set/adjust
    their acquisition order (the order scans are concatenated in matters).
    Returns a list of absolute file path strings in acquisition order, or
    None if the user cancelled."""
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass --scans manually.")

    selection = {"scans": None}

    dialog = tk.Tk()
    dialog.title("Select scans to stitch")
    dialog.geometry("760x480")

    if already_motion_corrected:
        help_text = (
            "Select the ALREADY motion-corrected functional scans (e.g. mc_func.nii.gz) for one \n"
            "subject/session, in acquisition order. Use this mode only if the scans were NOT \n"
            "acquired back-to-back (gap/repositioning between them)."
        )
    else:
        help_text = (
            "Select the RAW, pre-motion-correction functional scans for repeated electrical-stim \n"
            "runs acquired back-to-back within ONE session for ONE subject, in acquisition order. \n"
            "Each scan can be EITHER an already-converted NIfTI (e.g. func.nii.gz) OR a \n"
            "raw Bruker scan folder straight off the scanner (a numbered folder such as '10' \n"
            "containing a 'method' file and a pdata/ subfolder) - raw folders are converted to \n"
            "NIfTI automatically. Do not mix sessions/subjects, and do not include \n"
            "structural/anatomical scans or already motion-corrected files (mc_func.nii.gz) here - \n"
            "motion correction runs once on the whole concatenated series. All scans must share \n"
            "identical matrix size, voxel size, orientation and TR."
        )

    tk.Label(dialog, text=help_text, justify="left", pady=8).pack(anchor="w", padx=10)
    tk.Label(
        dialog,
        text="Add scans below (use Move Up/Down to fix acquisition order).",
        justify="left",
        pady=4,
    ).pack(anchor="w", padx=10)

    list_frame = tk.Frame(dialog)
    list_frame.pack(fill="both", expand=True, padx=10)

    scrollbar = tk.Scrollbar(list_frame, orient="vertical")
    listbox = tk.Listbox(list_frame, selectmode="extended", yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    def add_files():
        new_files = filedialog.askopenfilenames(
            title="Select scan NIfTI files",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*")],
        )
        for f in new_files:
            if f not in listbox.get(0, "end"):
                listbox.insert("end", f)

    def add_bruker_scans():
        selected_dir = filedialog.askdirectory(
            title="Select a raw Bruker scan folder, or its parent study folder"
        )
        if not selected_dir:
            return

        if is_bruker_raw_scan_dir(selected_dir):
            # User selected the scan folder itself (contains 'method' directly).
            if selected_dir not in listbox.get(0, "end"):
                listbox.insert("end", selected_dir)
            return

        # Otherwise treat it as a study root containing numbered scan subfolders.
        scan_numbers = choose_bruker_scan_numbers_via_dialog(selected_dir, dialog)
        for num in scan_numbers:
            scan_path = str(Path(selected_dir) / num)
            if scan_path not in listbox.get(0, "end"):
                listbox.insert("end", scan_path)

    def remove_selected():
        for idx in reversed(listbox.curselection()):
            listbox.delete(idx)

    def move_selected(offset):
        indices = list(listbox.curselection())
        if not indices:
            return
        if offset < 0:
            indices = sorted(indices)
        else:
            indices = sorted(indices, reverse=True)
        for idx in indices:
            new_idx = idx + offset
            if new_idx < 0 or new_idx >= listbox.size():
                continue
            value = listbox.get(idx)
            listbox.delete(idx)
            listbox.insert(new_idx, value)
            listbox.selection_set(new_idx)

    def clear_all():
        listbox.delete(0, "end")

    def on_ok():
        scans = list(listbox.get(0, "end"))
        if len(scans) < 2:
            messagebox.showerror("Not enough scans", "Select at least 2 scans to stitch.")
            return
        selection["scans"] = scans
        dialog.destroy()

    def on_cancel():
        selection["scans"] = None
        dialog.destroy()

    button_frame = tk.Frame(dialog, pady=8)
    button_frame.pack(fill="x", padx=10)
    tk.Button(button_frame, text="Add Files...", command=add_files).pack(side="left", padx=4)
    tk.Button(button_frame, text="Add Bruker Raw Scan(s)...", command=add_bruker_scans).pack(side="left", padx=4)
    tk.Button(button_frame, text="Remove Selected", command=remove_selected).pack(side="left", padx=4)
    tk.Button(button_frame, text="Move Up", command=lambda: move_selected(-1)).pack(side="left", padx=4)
    tk.Button(button_frame, text="Move Down", command=lambda: move_selected(1)).pack(side="left", padx=4)
    tk.Button(button_frame, text="Clear All", command=clear_all).pack(side="left", padx=4)

    ok_cancel_frame = tk.Frame(dialog, pady=8)
    ok_cancel_frame.pack()
    tk.Button(ok_cancel_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=10)
    tk.Button(ok_cancel_frame, text="Cancel", width=10, command=on_cancel).pack(side="right", padx=10)

    dialog.mainloop()

    return selection["scans"]


def choose_bruker_scan_numbers_via_dialog(study_root, parent):
    """Secondary popup used by choose_scans_via_dialog(): given a Bruker
    study/raw-data root folder, list its numbered scan subfolders (any
    directory directly containing a 'method' file) and let the user
    multi-select which ones to add. Returns a list of scan-number folder
    names (strings), in the order they appear in the list (not selection
    order) - use Move Up/Down in the main dialog afterwards to fix ordering."""
    candidates = sorted(
        (d.name for d in Path(study_root).iterdir() if is_bruker_raw_scan_dir(d)),
        key=lambda name: (len(name), name),
    )
    if not candidates:
        messagebox.showerror(
            "No scans found",
            f"No raw Bruker scan folders (numbered folders containing a 'method' file) "
            f"found directly inside {study_root}.",
        )
        return []

    picked = {"scans": []}
    sub = tk.Toplevel(parent)
    sub.title("Select scan numbers")
    sub.geometry("340x420")
    sub.transient(parent)

    tk.Label(
        sub,
        text=f"Raw Bruker scans found in:\n{study_root}\n\nSelect the scan(s) to add:",
        justify="left",
        pady=8,
    ).pack(anchor="w", padx=10)

    sub_list = tk.Listbox(sub, selectmode="extended")
    for name in candidates:
        sub_list.insert("end", name)
    sub_list.pack(fill="both", expand=True, padx=10, pady=6)

    def on_add():
        picked["scans"] = [sub_list.get(i) for i in sub_list.curselection()]
        sub.destroy()

    def on_cancel_sub():
        picked["scans"] = []
        sub.destroy()

    btn_frame = tk.Frame(sub, pady=8)
    btn_frame.pack()
    tk.Button(btn_frame, text="Add Selected", command=on_add).pack(side="left", padx=10)
    tk.Button(btn_frame, text="Cancel", command=on_cancel_sub).pack(side="right", padx=10)

    sub.grab_set()
    parent.wait_window(sub)
    return picked["scans"]


def choose_out_dir_via_dialog():
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass --out_dir manually.")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select output directory for stitched data")
    root.destroy()

    if not folder:
        return None
    return folder


def n_volumes(nifti_file):
    out = subprocess.run(
        ["fslnvols", str(nifti_file)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )
    return int(out.stdout.strip())


def write_boundaries_csv(scan_files, per_scan_volumes, out_csv):
    boundaries = []
    start = 0
    for idx, (scan_file, n_vols) in enumerate(zip(scan_files, per_scan_volumes), start=1):
        end = start + n_vols - 1
        boundaries.append({"scan": idx, "source_file": scan_file,
                            "start_volume": start, "end_volume": end, "n_volumes": n_vols})
        start = end + 1

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scan", "source_file", "start_volume", "end_volume", "n_volumes"])
        writer.writeheader()
        writer.writerows(boundaries)

    print_header("Scan boundaries in stitched series (0-based, inclusive)", bcolors.HEADER)
    for b in boundaries:
        print_statement(
            f"Scan {b['scan']}: volumes {b['start_volume']}-{b['end_volume']} "
            f"({b['n_volumes']} vols) <- {b['source_file']}",
            bcolors.OKBLUE,
        )
    return boundaries


def extract_middle_volume(in_file, reference_vol, out_file, size):
    extract_vol = fsl.ExtractROI()
    extract_vol.inputs.in_file = in_file
    extract_vol.inputs.t_min = reference_vol
    extract_vol.inputs.t_size = size
    extract_vol.inputs.roi_file = out_file
    extract_vol.run()
    print_statement("[OK] Reference volume extracted.", bcolors.OKGREEN)
    return out_file


def motion_correction(reference_vol, input_vol, output_prefix):
    volreg = afni.Volreg()
    volreg.inputs.in_file = input_vol
    volreg.inputs.basefile = reference_vol
    volreg.inputs.out_file = f"{output_prefix}.nii.gz"
    volreg.inputs.oned_file = "motion.1D"
    volreg.inputs.args = '-linear'
    volreg.inputs.oned_matrix_save = "rmsabs.1D"
    volreg.inputs.verbose = True
    volreg.run()
    print_statement(f"[OK] Motion correction complete -> {output_prefix}.nii.gz", bcolors.OKGREEN)
    return f"{output_prefix}.nii.gz"


def mean_image(in_file, out_file):
    mean = fsl.maths.MeanImage()
    mean.inputs.in_file = in_file
    mean.inputs.out_file = out_file
    mean.run()
    return out_file


def is_bruker_raw_scan_dir(path):
    """True if `path` is a numbered raw Bruker ParaVision scan folder, i.e. a
    directory directly containing a 'method' file (as opposed to a NIfTI
    file, a study root, or a pdata subfolder)."""
    p = Path(path)
    return p.is_dir() and (p / "method").is_file()


def bruker_to_nifti(in_path, scan_number, work_dir, out_name):
    """Convert one raw Bruker ParaVision scan to NIfTI. Mirrors bruker_to_nifti()
    in mol_fMRI_v5.3L_newcoreg.ipynb: run `brkraw tonii` on the study root for
    the given scan number, merge echoes with fslmerge if the scan is
    multi-echo, then resample to LPI orientation with AFNI. Runs isolated in
    `work_dir` (created if needed) so multiple scans can be converted without
    their intermediate files colliding. Returns the path to the converted,
    LPI-oriented NIfTI file (work_dir / out_name)."""
    in_path = Path(in_path)
    scan_number = str(scan_number)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    scan_dir = in_path / scan_number
    method_file = scan_dir / "method"
    if not is_bruker_raw_scan_dir(scan_dir):
        raise ValueError(
            f"{scan_dir} does not look like a raw Bruker scan folder (no 'method' file found). "
            "Expected structure: <study_root>/<scan_number>/{method,acqp,pdata,...}"
        )

    print_statement(f"Converting raw Bruker scan {scan_number} ({scan_dir}) -> NIfTI...", bcolors.NOTIFICATION)

    subprocess.run(["brkraw", "tonii", f"{in_path}/", "-s", scan_number], check=True, cwd=str(work_dir))

    n_echo_images = None
    with open(method_file) as f:
        for line in f:
            if "PVM_NEchoImages=" in line:
                echo_str = line.split("=")[1].strip()
                try:
                    n_echo_images = int(echo_str)
                except ValueError:
                    n_echo_images = 1
                break

    src_files = sorted(f for f in work_dir.glob(f"*{scan_number}*.nii*") if f.is_file())
    if not src_files:
        raise RuntimeError(f"No NIfTI files found after brkraw conversion for scan {scan_number} in {work_dir}")

    combined_file = work_dir / f"{scan_number}_combined.nii.gz"
    if n_echo_images is None or n_echo_images == 1:
        shutil.copy(src_files[0], combined_file)
    else:
        subprocess.run(["fslmerge", "-t", str(combined_file)] + [str(f) for f in src_files], check=True)
        print_statement(f"[OK] Merged {n_echo_images} echoes -> {combined_file}", bcolors.OKGREEN)

    out_file = work_dir / out_name
    resample = afni.Resample()
    resample.inputs.in_file = str(combined_file)
    resample.inputs.out_file = str(out_file)
    resample.inputs.orientation = "LPI"
    resample.run()

    with open(work_dir / "NIFTI_file_header_info.txt", "w") as f:
        subprocess.run(["fslhd", str(out_file)], stdout=f, check=True)

    print_statement(f"[OK] Raw Bruker scan {scan_number} converted -> {out_file}", bcolors.OKGREEN)
    return out_file


def resolve_scan_input(entry, conversion_root):
    """Accepts either a NIfTI file path or a raw Bruker scan folder for one
    scan entry. NIfTI paths are returned unchanged; raw Bruker folders are
    converted to NIfTI on the fly (see bruker_to_nifti()), isolated in their
    own subfolder of `conversion_root` to avoid collisions between scans."""
    p = Path(entry)
    if p.is_dir():
        if not is_bruker_raw_scan_dir(p):
            raise ValueError(
                f"{p} is a directory but does not look like a raw Bruker scan folder "
                "(no 'method' file found). Expected structure: <study_root>/<scan_number>/{method,pdata,...}"
            )
        in_path = p.parent
        scan_number = p.name
        work_dir = Path(conversion_root) / f"scan_{scan_number}"
        return bruker_to_nifti(in_path, scan_number, work_dir, f"{scan_number}_func_raw.nii.gz")
    return p


def coregister_scan_to_reference(mean_in_file, mean_reference_file, affine_out_file):
    """Rigid-ish affine registration of one scan's mean image to the reference
    scan's mean image. Same-modality (func-to-func), so a least-squares cost
    is used instead of the cross-modal correlation-ratio cost used for
    func-to-structural coregistration elsewhere in this codebase.
    Only used in the --already_motion_corrected --align fallback path."""
    allineate = afni.Allineate()
    allineate.inputs.in_file = mean_in_file
    allineate.inputs.reference = mean_reference_file
    allineate.inputs.out_matrix = affine_out_file
    allineate.inputs.cost = "ls"
    allineate.inputs.two_pass = True
    allineate.inputs.verbose = True
    allineate.inputs.out_file = "tmp_mean_aligned.nii.gz"
    allineate.run()
    os.remove("tmp_mean_aligned.nii.gz")
    return affine_out_file


def apply_affine_to_scan(in_file, reference_file, affine_file, out_file):
    allineate = afni.Allineate()
    allineate.inputs.in_file = in_file
    allineate.inputs.reference = reference_file
    allineate.inputs.in_matrix = affine_file
    allineate.inputs.master = reference_file
    allineate.inputs.final_interpolation = "linear"
    allineate.inputs.out_file = out_file
    allineate.run()
    return out_file


def stitch_raw_and_motion_correct(scan_files, out_dir, ref_volume_index=None):
    """Recommended path: concatenate raw scans, then motion-correct the whole
    stitched series once against a single common reference volume."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conversion_dir = out_dir / "converted_raw"
    scan_files = [str(resolve_scan_input(f, conversion_dir)) for f in scan_files]
    scan_files = [str(Path(f).resolve()) for f in scan_files]

    os.chdir(out_dir)

    print_header(f"Stitching {len(scan_files)} raw scans and applying motion correction", bcolors.HEADER)

    per_scan_volumes = [n_volumes(f) for f in scan_files]

    stitched_raw = "stitched_raw_func.nii.gz"
    subprocess.run(["fslmerge", "-t", stitched_raw] + scan_files, check=True)
    print_statement(f"[OK] Raw scans concatenated -> {stitched_raw}", bcolors.OKGREEN)

    write_boundaries_csv(scan_files, per_scan_volumes, "scan_boundaries.csv")

    total_vols = sum(per_scan_volumes)
    if ref_volume_index is None:
        ref_volume_index = total_vols // 2
    if not (0 <= ref_volume_index < total_vols):
        raise ValueError(f"ref_volume_index {ref_volume_index} out of range [0, {total_vols})")

    extract_middle_volume(stitched_raw, ref_volume_index, "middle_vol.nii.gz", 1)
    print_statement(f"[OK] Using volume {ref_volume_index} of stitched series as motion-correction reference.",
                    bcolors.NOTIFICATION)

    mc_stitched = motion_correction("middle_vol.nii.gz", stitched_raw, "mc_stitched_func")

    print_statement(f"[OK] Stitched, motion-corrected functional data -> {out_dir / mc_stitched}", bcolors.OKGREEN)
    return out_dir / mc_stitched, out_dir / "scan_boundaries.csv"


def stitch_already_motion_corrected(scan_files, out_dir, align=False):
    """Fallback path: scans were NOT back-to-back (gap/repositioning), and you only
    have each scan's independently motion-corrected mc_func.nii.gz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conversion_dir = out_dir / "converted_raw"
    scan_files = [str(resolve_scan_input(f, conversion_dir)) for f in scan_files]
    scan_files = [str(Path(f).resolve()) for f in scan_files]

    os.chdir(out_dir)

    reference_scan = scan_files[0]

    mode = "with inter-scan alignment" if align else "direct concatenation, no inter-scan alignment"
    print_header(f"Stitching {len(scan_files)} already motion-corrected scans ({mode})", bcolors.HEADER)

    if not align:
        aligned_files = scan_files
    else:
        mean_ref = mean_image(reference_scan, "mean_scan1.nii.gz")
        aligned_files = ["aligned_scan1.nii.gz"]
        subprocess.run(["fslmaths", reference_scan, aligned_files[0]], check=True)
        print_statement(f"[OK] Scan 1 (reference) prepared -> {aligned_files[0]}", bcolors.OKGREEN)

        for idx, scan_file in enumerate(scan_files[1:], start=2):
            mean_scan = mean_image(scan_file, f"mean_scan{idx}.nii.gz")
            affine_file = f"scan{idx}_to_scan1.aff12.1D"

            print_statement(f"Estimating alignment: scan{idx} -> scan1", bcolors.NOTIFICATION)
            coregister_scan_to_reference(mean_scan, mean_ref, affine_file)

            aligned_file = f"aligned_scan{idx}.nii.gz"
            apply_affine_to_scan(scan_file, mean_ref, affine_file, aligned_file)
            aligned_files.append(aligned_file)
            print_statement(f"[OK] Scan {idx} aligned to scan 1 space -> {aligned_file}", bcolors.OKGREEN)

    per_scan_volumes = [n_volumes(f) for f in aligned_files]
    write_boundaries_csv(scan_files, per_scan_volumes, "scan_boundaries.csv")

    stitched_out = "mc_stitched_func.nii.gz"
    subprocess.run(["fslmerge", "-t", stitched_out] + aligned_files, check=True)
    print_statement(f"[OK] Stitched functional data saved -> {out_dir / stitched_out}", bcolors.OKGREEN)

    return out_dir / stitched_out, out_dir / "scan_boundaries.csv"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stitch multiple electrical stim scans (same session) into one continuous 4D timeseries."
    )
    ap.add_argument("--scans", type=str, nargs="+", required=False,
                     help="Paths to scan inputs, in acquisition order. Each entry can be either an "
                          "already-converted NIfTI file (e.g. G1_cp_resampled.nii.gz / func.nii.gz) or a "
                          "raw Bruker scan folder (a numbered folder containing a 'method' file), which "
                          "will be auto-converted to NIfTI. If omitted, a popup will let you pick and "
                          "order the scans.")
    ap.add_argument("--out_dir", type=str, required=False,
                     help="Directory to write the stitched output and intermediate files. If omitted, a "
                          "folder picker dialog will open.")
    ap.add_argument("--ref_volume_index", type=int, default=None,
                     help="Volume index (0-based, within the stitched series) to use as the motion-correction "
                          "reference. Defaults to the middle volume of the whole stitched series.")
    ap.add_argument("--already_motion_corrected", action="store_true",
                     help="Scans were NOT acquired back-to-back (gap/repositioning) and are already "
                          "independently motion-corrected (mc_func.nii.gz). Skips raw concatenation + "
                          "single-pass motion correction; just concatenates the given files directly.")
    ap.add_argument("--align", action="store_true",
                     help="Only used with --already_motion_corrected: register each scan's mean image onto "
                          "scan 1's mean image before concatenating.")
    args = ap.parse_args()

    if args.scans is None:
        args.scans = choose_scans_via_dialog(already_motion_corrected=args.already_motion_corrected)
        if not args.scans:
            raise ValueError("No scans selected. Please run the script again and select at least 2 scans.")
        print_statement(f"Selected scans: {args.scans}", bcolors.OKBLUE)

    if args.out_dir is None:
        args.out_dir = choose_out_dir_via_dialog()
        if not args.out_dir:
            raise ValueError("No output directory selected. Please run the script again and choose one.")
        print_statement(f"Selected output directory: {args.out_dir}", bcolors.OKBLUE)

    if len(args.scans) < 2:
        raise ValueError("Provide at least 2 scans to stitch.")

    if args.already_motion_corrected:
        stitch_already_motion_corrected(args.scans, args.out_dir, align=args.align)
    else:
        stitch_raw_and_motion_correct(args.scans, args.out_dir, ref_volume_index=args.ref_volume_index)
