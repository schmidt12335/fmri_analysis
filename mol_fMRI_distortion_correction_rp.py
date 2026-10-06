"""
Susceptibility-distortion correction of an EPI functional scan using a short
reverse phase-encoded (blip-up/blip-down) EPI, via FSL's `topup` + `applytopup`.

Rationale
---------
EPI images are geometrically warped along the phase-encoding (PE) axis by
B0 field inhomogeneities (susceptibility distortion) - the warp is roughly
equal and opposite for two otherwise-identical EPI acquisitions whose PE
polarity is reversed (e.g. anterior->posterior vs posterior->anterior).
Given such a pair, FSL's `topup` estimates the underlying off-resonance
field (and the resulting voxel displacement field) from the pair, and
`applytopup` uses that field to unwarp the full functional EPI series.

Which input files to choose
----------------------------
  * --epi:         one OR MORE MAIN functional EPI series to be corrected -
                    each the full, multi-volume 4D timeseries (e.g. multiple
                    runs that will later be stitched together with
                    electric_stim_stitching.py). Pass several paths separated
                    by spaces to correct them all against the SAME
                    --rev_pe_epi in one run. Each entry can be EITHER an
                    already-converted NIfTI file OR a raw Bruker scan folder
                    straight off the scanner (a numbered folder such as '10'
                    containing a 'method' file and a pdata/ subfolder) - raw
                    folders are converted to NIfTI automatically (mirrors
                    bruker_to_nifti() in electric_stim_stitching.py /
                    data_analysis.py).
  * --rev_pe_epi:  a SINGLE SHORT EPI acquired with the SAME acquisition
                    parameters (matrix size, voxel size, slices, bandwidth/
                    readout) as --epi, except with the phasey-encoding
                    polarity reversed. This is typically only a handful of
                    volumes (sometimes just 1) - it is only used to estimate
                    the field, never written out as part of the corrected
                    data. Can also be EITHER a NIfTI file OR a raw Bruker
                    scan folder. It is resolved/converted only ONCE and
                    reused as the shared reference for every --epi entry.
  * Every --epi scan must share the same matrix size/orientation as
    --rev_pe_epi (same FOV, same number of slices) - only the PE polarity
    should differ.

You must also tell the script the phase-encoding axis of the MAIN EPI via
--pe_axis {x,y,z} and its polarity via --main_blip_sign {1,-1}. The
reverse-PE scan is assumed to have exactly the opposite polarity on the same
axis - this is what "reverse phase encoding" means and is what makes
topup's field estimate well-posed.

Automatic detection (raw Bruker --epi input only): if the FIRST raw-Bruker
--epi entry is a raw Bruker scan folder, the PE axis is auto-detected from
the scanner's BIDS metadata (PhaseEncodingDirection, extracted via
`brkraw tonii -b`) and carried through the LPI resampling done in
bruker_to_nifti(). The blip POLARITY (sign) is NOT reliably encoded in
standard Bruker headers, so the detected sign is only a best-effort default
(1, unless overridden). Either way - detected or manually specified via
--pe_axis/--main_blip_sign - the script always PAUSES and prints the
parameters it is about to use (applied identically to every --epi entry),
requiring you to type 'y' to confirm before topup/applytopup actually run.
If you are not sure of the
polarity, either value works as long as it is applied consistently; getting
it backwards will
still run but will make the distortion worse rather than better, so always
visually compare the corrected output against the original in fsleyes.

--readout_time is the TOTAL readout time (in seconds) of the EPI's phase-
encode train (i.e. effective echo spacing * (number of phase-encode
steps - 1)). This scales the estimated field's magnitude. If you do not
know the exact value for your sequence, using the same nominal value (e.g.
1.0) for both acquisitions still produces a geometrically correct unwarping
since only their (in this case, equal) magnitude enters the field-strength
scaling identically for both blips - but use the real value if you have it
for a physically accurate field map.

Pipeline (repeated for each --epi entry, against the shared --rev_pe_epi)
--------------------------------------------------------------------------
  1) Extract one representative volume from --epi and from --rev_pe_epi.
  2) Merge the two volumes into a single 4D file (blip_pair.nii.gz).
  3) Write acqparams.txt: one row per volume in blip_pair.nii.gz, encoding
     the PE direction (as a unit vector along --pe_axis) and total readout
     time for each.
  4) Run FSL `topup` on blip_pair.nii.gz + acqparams.txt to estimate the
     susceptibility off-resonance field (topup_results_fieldcoef.nii.gz /
     topup_results_movpar.txt).
  5) Run FSL `applytopup` on the FULL --epi series (using row 1 of
     acqparams.txt, i.e. the main EPI's own PE direction) to produce the
     distortion-corrected functional series.

Usage
-----
python rp_distortion_correction.py \
    --epi /path/func1.nii.gz /path/func2.nii.gz /path/func3.nii.gz \
    --rev_pe_epi /path/func_reverse_pe.nii.gz \
    --out_dir /path/distortion_corrected \
    --pe_axis y --readout_time 0.05

A single --epi works exactly the same way (just pass one path). If --epi,
--rev_pe_epi and/or --out_dir are omitted, popup dialogs open instead to
pick the file(s)/folder - the --epi dialog lets you add one or more NIfTI
files and/or raw Bruker scans before continuing.

Outputs (written into --out_dir)
---------------------------------
Each --epi entry gets its own subfolder under --out_dir (named after its
raw Bruker scan number, or its NIfTI filename, e.g. out_dir/scan_10/ or
out_dir/func1/), containing:
  blip_main.nii.gz              - extracted reference volume from this --epi
  blip_reverse.nii.gz           - extracted reference volume from --rev_pe_epi
  blip_pair.nii.gz              - the two volumes above, merged for topup
  acqparams.txt                 - phase-encoding/readout-time table used by topup
  topup_results_fieldcoef.nii.gz - estimated field spline coefficients (topup)
  topup_results_movpar.txt      - estimated between-volume movement parameters
  epi_distortion_corrected.nii.gz - the corrected, full functional series (main output)
A shared out_dir/converted_raw/ folder holds any raw-Bruker-to-NIfTI
conversion intermediates (including for --rev_pe_epi, converted only once).
"""

import argparse
import json
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
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

_AXIS_VECTORS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def _swap_yz_voxel_axes(nifti_path, out_path):
    """Swap voxel axes 1 ("y") and 2 ("z") of a NIfTI image, updating the
    affine to match, while leaving any additional (e.g. time) axes alone.
    FSL topup hard-codes array axis 2 as the through-plane/slice axis and
    refuses to run if the phase-encode vector's 3rd element is non-zero
    ("third element of pevec must be zero") - this can happen for scans
    whose acquisition plane isn't axial (e.g. coronal EPI), where the
    phase-encode direction legitimately ends up on the image's 3rd voxel
    axis. Applying this function moves that axis to slot 1 instead so topup
    accepts it; applying it a second time restores the original layout."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    perm = [0, 2, 1] + list(range(3, data.ndim))
    new_data = np.transpose(data, perm)

    new_affine = img.affine.copy()
    new_affine[:, [1, 2]] = img.affine[:, [2, 1]]

    nib.save(nib.Nifti1Image(new_data, new_affine), str(out_path))
    return out_path


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


def is_bruker_raw_scan_dir(path):
    """True if `path` is a numbered raw Bruker ParaVision scan folder, i.e. a
    directory directly containing a 'method' file (as opposed to a NIfTI
    file, a study root, or a pdata subfolder)."""
    p = Path(path)
    return p.is_dir() and (p / "method").is_file()


def _pe_sidecar_path(nifti_path):
    """Path of the phase-encoding-direction sidecar JSON written by
    bruker_to_nifti() alongside a converted NIfTI file."""
    p = Path(nifti_path)
    stem = p.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return p.parent / f"{stem}.detected_pe.json"


def _bids_phase_encoding_direction_to_axis(value):
    """Map a BIDS PhaseEncodingDirection value ('i', 'i-', 'j', 'j-', 'k',
    'k-') to a (raw_axis_index, sign) tuple, or (None, None) if the value is
    missing/unrecognized."""
    if not isinstance(value, str) or not value:
        return None, None
    axis_map = {"i": 0, "j": 1, "k": 2}
    letter = value[0].lower()
    if letter not in axis_map:
        return None, None
    sign = -1 if value.endswith("-") else 1
    return axis_map[letter], sign


def _propagate_axis_through_resample(raw_axis_idx, raw_sign, raw_nifti, resampled_nifti):
    """Given an axis index/sign defined in `raw_nifti`'s voxel space, work out
    the corresponding axis index/sign in `resampled_nifti`'s voxel space
    (which may have been reoriented, e.g. to LPI), using both files' NIfTI
    affines. Returns (axis_index, sign) in resampled_nifti's voxel space, or
    (None, None) if this cannot be determined (e.g. nibabel unavailable)."""
    try:
        import nibabel as nib
        from nibabel.orientations import io_orientation, ornt_transform
    except ImportError:
        return None, None

    try:
        raw_ornt = io_orientation(nib.load(str(raw_nifti)).affine)
        res_ornt = io_orientation(nib.load(str(resampled_nifti)).affine)
        transform = ornt_transform(raw_ornt, res_ornt)
        target_axis_idx, flip = transform[raw_axis_idx]
        return int(target_axis_idx), int(raw_sign * flip)
    except Exception:
        return None, None


def load_detected_pe_direction(nifti_path):
    """If `nifti_path` was produced by bruker_to_nifti() from a raw Bruker
    scan, load the auto-detected phase-encoding axis/sign written alongside
    it. Returns (axis, sign, bids_phase_encoding_direction), any of which may
    be None if detection was not possible or nifti_path did not come from a
    raw Bruker conversion (e.g. a plain user-supplied NIfTI file)."""
    sidecar = _pe_sidecar_path(nifti_path)
    if not sidecar.is_file():
        return None, None, None
    try:
        with open(sidecar) as f:
            info = json.load(f)
    except Exception:
        return None, None, None
    return info.get("axis"), info.get("sign"), info.get("bids_phase_encoding_direction")


def bruker_to_nifti(in_path, scan_number, work_dir, out_name):
    """Convert one raw Bruker ParaVision scan to NIfTI. Mirrors bruker_to_nifti()
    in electric_stim_stitching.py / mol_fMRI_v5.3L_newcoreg.ipynb: run
    `brkraw tonii` on the study root for the given scan number, merge echoes
    with fslmerge if the scan is multi-echo, then resample to LPI orientation
    with AFNI. Runs isolated in `work_dir` (created if needed) so multiple
    scans can be converted without their intermediate files colliding.
    Also attempts to auto-detect the phase-encoding axis (and a best-effort
    polarity) from the scan's BIDS metadata (see load_detected_pe_direction()).
    Returns the path to the converted, LPI-oriented NIfTI file
    (work_dir / out_name)."""
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

    subprocess.run(["brkraw", "tonii", f"{in_path}/", "-s", scan_number, "-b"], check=True, cwd=str(work_dir))

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

    # Skip macOS "._" AppleDouble files (SMB shares) and our own outputs from an earlier run in the same work_dir
    src_files = sorted(
        f for f in work_dir.glob(f"*{scan_number}*.nii*")
        if f.is_file() and not f.name.startswith("._")
        and not f.name.endswith(("_combined.nii.gz", "_epi_raw.nii.gz"))
    )
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

    # Best-effort auto-detection of the phase-encoding axis (and a candidate
    # polarity) from the scan's BIDS sidecar JSON, carried through the LPI
    # resample above so it is expressed in out_file's own voxel axes.
    bids_pe_value = None
    detected_axis = None
    detected_sign = None
    json_candidates = sorted(
        f for f in work_dir.glob(f"*{scan_number}*.json")
        if not f.name.startswith("._") and not f.name.endswith(".detected_pe.json")
    )
    if json_candidates:
        try:
            with open(json_candidates[0]) as jf:
                bids_pe_value = json.load(jf).get("PhaseEncodingDirection")
            raw_axis_idx, raw_sign = _bids_phase_encoding_direction_to_axis(bids_pe_value)
            if raw_axis_idx is not None:
                axis_idx, sign = _propagate_axis_through_resample(raw_axis_idx, raw_sign, combined_file, out_file)
                if axis_idx is not None:
                    detected_axis = {0: "x", 1: "y", 2: "z"}[axis_idx]
                    detected_sign = sign
        except Exception as e:
            print_statement(
                f"[Note] Could not parse phase-encoding metadata for scan {scan_number}: {e}", bcolors.NOTIFICATION
            )

    with open(_pe_sidecar_path(out_file), "w") as f:
        json.dump(
            {
                "bids_phase_encoding_direction": bids_pe_value,
                "axis": detected_axis,
                "sign": detected_sign,
            },
            f,
            indent=2,
        )

    if detected_axis is not None:
        print_statement(
            f"[Info] Auto-detected phase-encoding axis for scan {scan_number}: '{detected_axis}' "
            f"(candidate sign {detected_sign:+d}) from BIDS metadata PhaseEncodingDirection={bids_pe_value!r}. "
            "The axis is reliable; the sign is only a best-effort default - please verify.",
            bcolors.NOTIFICATION,
        )
    else:
        print_statement(
            f"[Note] Could not auto-detect phase-encoding direction for scan {scan_number} "
            "(no usable PhaseEncodingDirection metadata found) - --pe_axis/--main_blip_sign must be set manually.",
            bcolors.NOTIFICATION,
        )

    print_statement(f"[OK] Raw Bruker scan {scan_number} converted -> {out_file}", bcolors.OKGREEN)
    return out_file


def resolve_scan_input(entry, conversion_root):
    """Accepts either a NIfTI file path or a raw Bruker scan folder for one
    EPI input (--epi or --rev_pe_epi). NIfTI paths are returned unchanged;
    raw Bruker folders are converted to NIfTI on the fly (see
    bruker_to_nifti()), isolated in their own subfolder of `conversion_root`
    to avoid collisions between the two scans."""
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
        out_name = f"{scan_number}_epi_raw.nii.gz"
        converted = work_dir / out_name
        if converted.is_file() and _pe_sidecar_path(converted).is_file():
            print_statement(f"Scan {scan_number} already converted. Skipping conversion.", bcolors.OKGREEN)
            return converted
        return bruker_to_nifti(in_path, scan_number, work_dir, out_name)
    return p


def choose_bruker_scan_numbers_via_dialog(study_root, parent):
    """Secondary popup used by choose_epi_input_via_dialog(): given a Bruker
    study/raw-data root folder, list its numbered scan subfolders (any
    directory directly containing a 'method' file) and let the user
    multi-select which one(s) to add."""
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
    sub.title("Select scan number")
    sub.geometry("340x420")
    sub.transient(parent)

    tk.Label(
        sub,
        text=f"Raw Bruker scans found in:\n{study_root}\n\nSelect the scan to add:",
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


def choose_epi_input_via_dialog(title):
    """Popup for selecting ONE EPI input (--epi or --rev_pe_epi), either an
    already-converted NIfTI file or a raw Bruker scan folder (auto-converted
    later via resolve_scan_input()). Returns a path string, or None if
    cancelled."""
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass the file path manually.")

    selection = {"path": None}

    dialog = tk.Tk()
    dialog.title(title)
    dialog.geometry("560x180")

    tk.Label(
        dialog,
        text=title + "\n\nChoose EITHER an already-converted NIfTI file OR a raw Bruker scan\n"
             "folder straight off the scanner (a numbered folder such as '10' containing a\n"
             "'method' file and a pdata/ subfolder) - raw folders are converted to NIfTI\n"
             "automatically.",
        justify="left", pady=8,
    ).pack(anchor="w", padx=10)

    def pick_nifti():
        f = filedialog.askopenfilename(
            title=title,
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*")],
        )
        if f:
            selection["path"] = f
            dialog.destroy()

    def pick_bruker():
        selected_dir = filedialog.askdirectory(
            title="Select a raw Bruker scan folder, or its parent study folder"
        )
        if not selected_dir:
            return

        if is_bruker_raw_scan_dir(selected_dir):
            selection["path"] = selected_dir
            dialog.destroy()
            return

        scan_numbers = choose_bruker_scan_numbers_via_dialog(selected_dir, dialog)
        if scan_numbers:
            selection["path"] = str(Path(selected_dir) / scan_numbers[0])
            dialog.destroy()

    button_frame = tk.Frame(dialog, pady=8)
    button_frame.pack()
    tk.Button(button_frame, text="Select NIfTI File...", width=22, command=pick_nifti).pack(side="left", padx=8)
    tk.Button(button_frame, text="Select Raw Bruker Scan...", width=22, command=pick_bruker).pack(side="left", padx=8)

    cancel_frame = tk.Frame(dialog, pady=4)
    cancel_frame.pack()
    tk.Button(cancel_frame, text="Cancel", width=10, command=dialog.destroy).pack()

    dialog.mainloop()

    return selection["path"]


def choose_multiple_epi_inputs_via_dialog(title):
    """Popup for selecting ONE OR MORE --epi inputs, each either an
    already-converted NIfTI file or a raw Bruker scan folder (auto-converted
    later via resolve_scan_input()). NIfTI files and raw Bruker scans can be
    mixed and added incrementally. Returns a list of path strings (empty if
    cancelled or nothing was added)."""
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass the file path(s) manually.")

    selected = []

    dialog = tk.Tk()
    dialog.title(title)
    dialog.geometry("640x440")

    tk.Label(
        dialog,
        text=title + "\n\nAdd one or more EPI scans below (NIfTI files and/or raw Bruker scan\n"
             "folders can be mixed) - all of them will be corrected against the SAME\n"
             "reverse phase-encoded reference you select next.",
        justify="left", pady=8,
    ).pack(anchor="w", padx=10)

    list_frame = tk.Frame(dialog)
    list_frame.pack(fill="both", expand=True, padx=10, pady=6)
    listbox = tk.Listbox(list_frame, selectmode="extended")
    listbox.pack(fill="both", expand=True)

    def add_nifti():
        files = filedialog.askopenfilenames(
            title="Select one or more NIfTI files",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*")],
        )
        for f in files:
            if f not in selected:
                selected.append(f)
                listbox.insert("end", f)

    def add_bruker():
        selected_dir = filedialog.askdirectory(
            title="Select a raw Bruker scan folder, or its parent study folder"
        )
        if not selected_dir:
            return

        if is_bruker_raw_scan_dir(selected_dir):
            if selected_dir not in selected:
                selected.append(selected_dir)
                listbox.insert("end", selected_dir)
            return

        for scan_number in choose_bruker_scan_numbers_via_dialog(selected_dir, dialog):
            scan_path = str(Path(selected_dir) / scan_number)
            if scan_path not in selected:
                selected.append(scan_path)
                listbox.insert("end", scan_path)

    def remove_selected():
        for idx in reversed(listbox.curselection()):
            path = listbox.get(idx)
            listbox.delete(idx)
            if path in selected:
                selected.remove(path)

    def on_cancel():
        selected.clear()
        dialog.destroy()

    button_frame = tk.Frame(dialog, pady=6)
    button_frame.pack()
    tk.Button(button_frame, text="Add NIfTI File(s)...", width=20, command=add_nifti).pack(side="left", padx=6)
    tk.Button(button_frame, text="Add Raw Bruker Scan(s)...", width=22, command=add_bruker).pack(side="left", padx=6)
    tk.Button(button_frame, text="Remove Selected", width=16, command=remove_selected).pack(side="left", padx=6)

    footer_frame = tk.Frame(dialog, pady=8)
    footer_frame.pack()
    tk.Button(footer_frame, text="Continue", width=12, command=dialog.destroy).pack(side="left", padx=10)
    tk.Button(footer_frame, text="Cancel", width=12, command=on_cancel).pack(side="left", padx=10)

    dialog.mainloop()

    return selected


def choose_out_dir_via_dialog():
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available on this system. Please pass --out_dir manually.")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select output directory for distortion-corrected data")
    root.destroy()

    if not folder:
        return None
    return folder


def extract_volume(in_file, vol_index, out_file, size=1):
    extract = fsl.ExtractROI()
    extract.inputs.in_file = str(in_file)
    extract.inputs.t_min = vol_index
    extract.inputs.t_size = size
    extract.inputs.roi_file = out_file
    extract.run()
    print_statement(f"[OK] Extracted volume {vol_index} from {in_file} -> {out_file}", bcolors.OKGREEN)
    return out_file


def build_acqparams(pe_axis, readout_time, out_file, main_sign=1):
    """Write a 2-row FSL acqparams.txt: row 1 for the main EPI's phase-encode
    direction/readout time, row 2 for the reverse-PE EPI (same axis, opposite
    sign). Each row is "<x> <y> <z> <total_readout_time>"."""
    pe_axis = pe_axis.lower()
    if pe_axis not in _AXIS_VECTORS:
        raise ValueError(f"pe_axis must be one of 'x', 'y', 'z' (got {pe_axis!r}).")
    if main_sign not in (1, -1):
        raise ValueError(f"main_sign must be 1 or -1 (got {main_sign!r}).")

    vec = _AXIS_VECTORS[pe_axis]
    main_row = " ".join(str(main_sign * v) for v in vec) + f" {readout_time}"
    rev_row = " ".join(str(-main_sign * v) for v in vec) + f" {readout_time}"

    with open(out_file, "w") as f:
        f.write(main_row + "\n")
        f.write(rev_row + "\n")

    print_statement(f"[OK] Wrote acquisition parameters -> {out_file}", bcolors.OKGREEN)
    print_statement(f"  Row 1 (main EPI, --epi):        {main_row}", bcolors.OKBLUE)
    print_statement(f"  Row 2 (reverse EPI, --rev_pe_epi): {rev_row}", bcolors.OKBLUE)
    return out_file


def _resolve_topup_config_path(config):
    """Resolve a topup --config value (bare filename or full path) to an
    actual file path, mirroring how FSL itself looks up config files (as a
    literal path, or under $FSLDIR/etc/flirtsch/)."""
    p = Path(config)
    if p.is_file():
        return p
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        candidate = Path(fsldir) / "etc" / "flirtsch" / config
        if candidate.is_file():
            return candidate
    return None


def _ensure_topup_compatible_config(config, image_file, out_dir):
    """FSL topup's default multi-resolution schedule (e.g. b02b0.cnf)
    subsamples each spatial dimension by up to a factor of 2 at its coarser
    levels. If any spatial dimension of image_file is not evenly divisible
    by that factor (common with a small/odd number of slices, e.g. rodent
    EPI), topup fails immediately with "Subsampling levels incompatible
    with image data" instead of running. When that would happen, write a
    modified copy of the config with subsampling disabled (subsamp=1 at
    every level) and use that instead."""
    config_path = _resolve_topup_config_path(config)
    if config_path is None:
        return config  # can't introspect it; let FSL raise its own error if incompatible

    try:
        import nibabel as nib
        dims = nib.load(str(image_file)).shape[:3]
    except Exception:
        return config

    lines = config_path.read_text().splitlines()
    max_subsamp = 1
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--subsamp="):
            values = [int(v) for v in stripped.split("=", 1)[1].split(",")]
            max_subsamp = max(max_subsamp, max(values))

    if max_subsamp <= 1 or all(d % max_subsamp == 0 for d in dims):
        return config  # already compatible, use as-is

    print_statement(
        f"[Note] Image dimensions {tuple(dims)} are not evenly divisible by the subsampling factor "
        f"({max_subsamp}) used in '{config}' - topup would fail with 'Subsampling levels incompatible "
        f"with image data' (common with a small/odd number of slices). Using a modified config with "
        f"subsampling disabled instead.",
        bcolors.NOTIFICATION,
    )
    fixed_lines = []
    for line in lines:
        if line.strip().startswith("--subsamp="):
            n_levels = len(line.strip().split("=", 1)[1].split(","))
            fixed_lines.append("--subsamp=" + ",".join(["1"] * n_levels))
        else:
            fixed_lines.append(line)

    fixed_config = Path(out_dir) / "topup_config_nosubsamp.cnf"
    fixed_config.write_text("\n".join(fixed_lines) + "\n")
    return str(fixed_config)


def run_topup(merged_b0_file, acqparams_file, out_base, config="b02b0.cnf"):
    config = _ensure_topup_compatible_config(config, merged_b0_file, Path(merged_b0_file).resolve().parent)
    topup = fsl.TOPUP()
    topup.inputs.in_file = str(merged_b0_file)
    topup.inputs.encoding_file = str(acqparams_file)
    topup.inputs.config = config
    topup.inputs.out_base = out_base
    result = topup.run()
    print_statement("[OK] topup field estimation complete.", bcolors.OKGREEN)
    return result.outputs


def run_applytopup(in_files, acqparams_file, in_index, fieldcoef_file, movpar_file, out_file, method="jac"):
    applytopup = fsl.ApplyTOPUP()
    applytopup.inputs.in_files = [str(f) for f in in_files]
    applytopup.inputs.encoding_file = str(acqparams_file)
    applytopup.inputs.in_index = in_index
    applytopup.inputs.in_topup_fieldcoef = str(fieldcoef_file)
    applytopup.inputs.in_topup_movpar = str(movpar_file)
    applytopup.inputs.method = method
    applytopup.inputs.out_corrected = str(out_file)
    applytopup.run()
    print_statement(f"[OK] applytopup complete -> {out_file}", bcolors.OKGREEN)
    return out_file


def distortion_correct_epi(
    epi_file,
    rev_pe_file,
    out_dir,
    pe_axis="y",
    main_sign=1,
    readout_time=0.05,
    epi_vol_index=0,
    rev_vol_index=0,
    config="b02b0.cnf",
    interp_method="jac",
):
    """Recommended path: estimate the susceptibility field from one volume of
    --epi and one volume of --rev_pe_epi (via topup), then apply that field
    to the full --epi series (via applytopup). Either input may be a NIfTI
    file or a raw Bruker scan folder (auto-converted via resolve_scan_input())."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    conversion_dir = out_dir / "converted_raw"
    epi_file = str(Path(resolve_scan_input(epi_file, conversion_dir)).resolve())
    rev_pe_file = str(Path(resolve_scan_input(rev_pe_file, conversion_dir)).resolve())

    os.chdir(out_dir)

    print_header("Distortion correction via reverse phase-encoded EPI (FSL topup)", bcolors.HEADER)

    # FSL topup hard-codes voxel axis 2 ("z") as the through-plane/slice axis
    # and refuses to run if the phase-encode direction lands there (e.g. for
    # coronal-plane EPI, where PE legitimately ends up on the image's 3rd
    # voxel axis). Swap it onto axis 1 ("y") instead, run the whole
    # correction in that space, then swap the final corrected output back.
    pe_axis_swapped = pe_axis.lower() == "z"
    if pe_axis_swapped:
        print_statement(
            "[Note] --pe_axis is 'z' (the image's 3rd/slice voxel axis) - FSL topup does not support this "
            "directly. Temporarily swapping the y/z voxel axes for topup/applytopup, then swapping the "
            "corrected output back afterwards.",
            bcolors.NOTIFICATION,
        )
        swapped_epi_file = out_dir / "epi_yz_swapped.nii.gz"
        swapped_rev_file = out_dir / "rev_pe_yz_swapped.nii.gz"
        _swap_yz_voxel_axes(epi_file, swapped_epi_file)
        _swap_yz_voxel_axes(rev_pe_file, swapped_rev_file)
        epi_file = str(swapped_epi_file)
        rev_pe_file = str(swapped_rev_file)
        pe_axis = "y"

    extract_volume(epi_file, epi_vol_index, "blip_main.nii.gz", 1)
    extract_volume(rev_pe_file, rev_vol_index, "blip_reverse.nii.gz", 1)

    subprocess.run(
        ["fslmerge", "-t", "blip_pair.nii.gz", "blip_main.nii.gz", "blip_reverse.nii.gz"],
        check=True,
    )
    print_statement("[OK] Blip-up/blip-down pair merged -> blip_pair.nii.gz", bcolors.OKGREEN)

    build_acqparams(pe_axis, readout_time, "acqparams.txt", main_sign=main_sign)

    print_statement("Running FSL topup to estimate the susceptibility-induced field...", bcolors.NOTIFICATION)
    topup_outputs = run_topup("blip_pair.nii.gz", "acqparams.txt", "topup_results", config=config)

    print_statement("Applying topup correction to the full functional EPI series...", bcolors.NOTIFICATION)
    corrected_file = out_dir / "epi_distortion_corrected.nii.gz"
    run_applytopup(
        in_files=[epi_file],
        acqparams_file="acqparams.txt",
        in_index=[1],
        fieldcoef_file=topup_outputs.out_fieldcoef,
        movpar_file=topup_outputs.out_movpar,
        out_file=str(corrected_file),
        method=interp_method,
    )

    if pe_axis_swapped:
        _swap_yz_voxel_axes(corrected_file, corrected_file)
        print_statement("[OK] Swapped corrected output back to the original voxel-axis layout.", bcolors.OKGREEN)

    print_statement(f"[OK] Distortion-corrected functional data -> {corrected_file}", bcolors.OKGREEN)
    return corrected_file


def _epi_output_subdir_name(entry, used_names):
    """Derive a short, unique, filesystem-safe output subfolder name for one
    --epi input, based on its raw Bruker scan number (if any) or NIfTI
    filename stem. Appends a numeric suffix on collision."""
    p = Path(entry)
    if is_bruker_raw_scan_dir(p):
        base = f"scan_{p.name}"
    elif p.name.endswith("_epi_raw.nii.gz") and p.parent.name.startswith("scan_"):
        base = p.parent.name  # already converted by resolve_scan_input()
    else:
        name = p.name
        for suffix in (".nii.gz", ".nii"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        base = name or "epi"

    candidate = base
    i = 2
    while candidate in used_names:
        candidate = f"{base}_{i}"
        i += 1
    used_names.add(candidate)
    return candidate


def distortion_correct_multiple_epis(
    epi_files,
    rev_pe_file,
    out_dir,
    pe_axis="y",
    main_sign=1,
    readout_time=0.05,
    epi_vol_index=0,
    rev_vol_index=0,
    config="b02b0.cnf",
    interp_method="jac",
    skip_existing=False,
):
    """Correct one or more --epi scans (e.g. multiple functional runs that
    will later be stitched together) against a SINGLE shared --rev_pe_epi
    reference. The reverse-PE scan (including any raw-Bruker conversion it
    needs) is resolved once and reused for every main EPI; each main EPI
    gets its own output subfolder under out_dir (named after its scan
    number/filename, see _epi_output_subdir_name()) so per-scan
    intermediates (blip_main.nii.gz, topup_results*, etc.) never collide.
    With skip_existing=True, an EPI whose epi_distortion_corrected.nii.gz
    already exists is not recomputed.
    Returns a list of corrected-file paths, in the same order as
    epi_files. The working directory is restored afterwards."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    original_cwd = os.getcwd()

    conversion_dir = out_dir / "converted_raw"
    rev_pe_resolved = str(Path(resolve_scan_input(rev_pe_file, conversion_dir)).resolve())

    used_names = set()
    corrected_files = []
    n = len(epi_files)
    try:
        for i, epi_entry in enumerate(epi_files, start=1):
            subdir_name = _epi_output_subdir_name(epi_entry, used_names)
            existing = out_dir / subdir_name / "epi_distortion_corrected.nii.gz"
            if skip_existing and existing.is_file() and existing.stat().st_size > 0:
                print_statement(f"EPI {i}/{n} ({subdir_name}) already distortion corrected. Skipping.", bcolors.OKGREEN)
                corrected_files.append(existing)
                continue
            print_header(f"Distortion correction: EPI {i}/{n} ({subdir_name})", bcolors.HEADER)
            corrected_file = distortion_correct_epi(
                epi_entry,
                rev_pe_resolved,
                out_dir / subdir_name,
                pe_axis=pe_axis,
                main_sign=main_sign,
                readout_time=readout_time,
                epi_vol_index=epi_vol_index,
                rev_vol_index=rev_vol_index,
                config=config,
                interp_method=interp_method,
            )
            corrected_files.append(corrected_file)
    finally:
        os.chdir(original_cwd)

    print_header(f"All {n} EPI(s) corrected", bcolors.HEADER)
    for epi_entry, corrected_file in zip(epi_files, corrected_files):
        print_statement(f"  {epi_entry} -> {corrected_file}", bcolors.OKGREEN)

    return corrected_files


def run_with_confirmation(
    epi_files,
    rev_pe_file,
    out_dir,
    pe_axis=None,
    main_sign=None,
    readout_time=0.05,
    epi_vol_index=0,
    rev_vol_index=0,
    config="b02b0.cnf",
    interp_method="jac",
    skip_existing=True,
    confirm=True,
):
    """Entry point shared by the command line and the preprocessing notebooks.
    Resolves raw Bruker inputs (converted once, reused on later calls), auto-
    detects the phase-encoding axis/sign where possible (pe_axis/main_sign
    left as None), shows the parameters and asks for confirmation before
    topup/applytopup run. Scans that are already corrected are reused when
    skip_existing is set, and nothing is asked if there is nothing to compute.
    Returns the list of corrected files, in the order of epi_files."""
    out_dir = Path(out_dir)
    # Resolving raw inputs first is a no-op for the later calls inside
    # distortion_correct_multiple_epis()/distortion_correct_epi(), so the
    # scans are not converted twice.
    conversion_dir = out_dir / "converted_raw"
    epi_was_raw_flags = [is_bruker_raw_scan_dir(entry) for entry in epi_files]
    epi_files = [str(resolve_scan_input(entry, conversion_dir)) for entry in epi_files]
    rev_pe_file = str(resolve_scan_input(rev_pe_file, conversion_dir))

    used_names = set()
    pending = [
        entry for entry in epi_files
        if not (skip_existing and (out_dir / _epi_output_subdir_name(entry, used_names)
                                   / "epi_distortion_corrected.nii.gz").is_file())
    ]

    # Auto-detection only needs to run once - use the first raw-Bruker --epi
    # entry that yields a usable detection result.
    detected_axis, detected_sign, bids_pe_value = (None, None, None)
    for entry, was_raw in zip(epi_files, epi_was_raw_flags):
        if was_raw:
            detected_axis, detected_sign, bids_pe_value = load_detected_pe_direction(entry)
            if detected_axis is not None:
                break

    if pe_axis is None:
        pe_axis = detected_axis if detected_axis is not None else "y"
    if main_sign is None:
        main_sign = detected_sign if detected_sign is not None else 1

    if pending and confirm:
        print_header("Please confirm the phase-encoding parameters", bcolors.HEADER)
        if detected_axis is not None:
            print_statement(
                f"Auto-detected from raw Bruker BIDS metadata (PhaseEncodingDirection={bids_pe_value!r}): "
                f"axis='{detected_axis}', candidate sign={detected_sign:+d}.",
                bcolors.NOTIFICATION,
            )
            print_statement(
                "NOTE: the axis is derived reliably from the scanner header, but the blip POLARITY (sign) is "
                "NOT reliably encoded in standard Bruker metadata - the sign above is only a best-effort default.",
                bcolors.NOTIFICATION,
            )
        else:
            print_statement(
                "Automatic phase-encoding detection was not available (requires a raw Bruker --epi scan with "
                "readable BIDS metadata) - using manually specified / default values.",
                bcolors.NOTIFICATION,
            )
        print_statement(f"  --pe_axis        = {pe_axis}", bcolors.OKBLUE)
        print_statement(f"  --main_blip_sign = {main_sign}", bcolors.OKBLUE)
        print_statement(f"  --readout_time   = {readout_time}", bcolors.OKBLUE)
        print_statement(
            f"These parameters will be applied identically to all {len(pending)} --epi scan(s) being corrected "
            f"against the shared --rev_pe_epi.",
            bcolors.OKBLUE,
        )
        print_statement(
            "Getting these wrong will not crash the pipeline but will make the distortion WORSE, not better - "
            "always visually check the corrected output against the original in fsleyes.",
            bcolors.NOTIFICATION,
        )
        confirmation = input("Proceed with these phase-encoding parameters? [y/N]: ").strip().lower()
        if confirmation not in ("y", "yes"):
            raise SystemExit(
                "Aborted by user - re-run with the correct --pe_axis / --main_blip_sign (or fix the raw scan "
                "input) and try again."
            )

    return distortion_correct_multiple_epis(
        epi_files,
        rev_pe_file,
        out_dir,
        pe_axis=pe_axis,
        main_sign=main_sign,
        readout_time=readout_time,
        epi_vol_index=epi_vol_index,
        rev_vol_index=rev_vol_index,
        config=config,
        interp_method=interp_method,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Correct EPI susceptibility distortion using a reverse phase-encoded EPI (FSL topup + applytopup)."
    )
    ap.add_argument("--epi", type=str, nargs="+", required=False,
                     help="Path(s) to one or more main functional EPI series to be corrected (each a full 4D "
                          "NIfTI, or a raw Bruker scan folder - auto-converted). Pass several paths separated "
                          "by spaces to correct multiple runs (e.g. scans to be stitched later) against the "
                          "same --rev_pe_epi in one go. If omitted, a picker dialog will open to add one or more.")
    ap.add_argument("--rev_pe_epi", type=str, required=False,
                     help="Path to the short reverse phase-encoded EPI (same acquisition parameters as --epi, "
                          "opposite PE polarity; NIfTI file or raw Bruker scan folder - auto-converted). "
                          "If omitted, a file picker dialog will open.")
    ap.add_argument("--out_dir", type=str, required=False,
                     help="Directory to write the corrected output and intermediate files. If omitted, a "
                          "folder picker dialog will open.")
    ap.add_argument("--pe_axis", type=str, default=None, choices=["x", "y", "z"],
                     help="Phase-encoding axis of --epi. If omitted, auto-detected from raw Bruker BIDS "
                          "metadata when --epi is a raw scan folder (falls back to 'y' otherwise).")
    ap.add_argument("--main_blip_sign", type=int, default=None, choices=[1, -1],
                     help="Polarity of --epi's phase encoding along --pe_axis. --rev_pe_epi is assumed to "
                          "have the opposite polarity on the same axis. If omitted, a best-effort default is "
                          "derived from raw Bruker metadata when possible (falls back to 1 otherwise) - this is "
                          "always shown for confirmation before running, since polarity is not reliably encoded "
                          "in standard Bruker headers.")
    ap.add_argument("--readout_time", type=float, default=0.05,
                     help="Total EPI readout time in seconds (effective echo spacing * "
                          "(number of phase-encode steps - 1)). Default: 0.05.")
    ap.add_argument("--epi_vol_index", type=int, default=0,
                     help="Volume index (0-based) to extract from --epi as the topup reference volume. Default: 0.")
    ap.add_argument("--rev_vol_index", type=int, default=0,
                     help="Volume index (0-based) to extract from --rev_pe_epi as the topup reference volume. "
                          "Default: 0.")
    ap.add_argument("--config", type=str, default="b02b0.cnf",
                     help="FSL topup config file (default: b02b0.cnf, FSL's standard b0-to-b0 config).")
    ap.add_argument("--interp_method", type=str, default="jac", choices=["jac", "lsr"],
                     help="applytopup interpolation method (default: jac).")
    args = ap.parse_args()

    if args.epi is None:
        args.epi = choose_multiple_epi_inputs_via_dialog("Select the main functional EPI scan(s) (to be corrected)")
        if not args.epi:
            raise ValueError("No EPI file(s) selected. Please run the script again and select at least one.")
        for entry in args.epi:
            print_statement(f"Selected main EPI: {entry}", bcolors.OKBLUE)

    if args.rev_pe_epi is None:
        args.rev_pe_epi = choose_epi_input_via_dialog("Select the short reverse phase-encoded EPI scan")
        if not args.rev_pe_epi:
            raise ValueError("No reverse phase-encoded EPI file selected. Please run the script again and select one.")
        print_statement(f"Selected reverse phase-encoded EPI: {args.rev_pe_epi}", bcolors.OKBLUE)

    if args.out_dir is None:
        args.out_dir = choose_out_dir_via_dialog()
        if not args.out_dir:
            raise ValueError("No output directory selected. Please run the script again and choose one.")
        print_statement(f"Selected output directory: {args.out_dir}", bcolors.OKBLUE)

    run_with_confirmation(
        args.epi,
        args.rev_pe_epi,
        args.out_dir,
        pe_axis=args.pe_axis,
        main_sign=args.main_blip_sign,
        readout_time=args.readout_time,
        epi_vol_index=args.epi_vol_index,
        rev_vol_index=args.rev_vol_index,
        config=args.config,
        interp_method=args.interp_method,
        skip_existing=False,
    )
