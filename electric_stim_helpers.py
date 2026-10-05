"""Helper functions for the electrical stimulation fMRI notebooks (conversion, preprocessing, plotting, console output)."""
# The notebooks get these imports through 'from electric_stim_helpers import *'
import pandas as pd
import numpy as np
from nipype.interfaces import afni, fsl
import nibabel as nib
import matplotlib.pyplot as plt
import os
import csv
import shutil
import glob
import subprocess
from pathlib import Path
import re
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser
from tabulate import tabulate

# Pipeline functions used by the preprocessing and analysis notebooks

def bruker_to_nifti(in_path, scan_number, out_file):

    scan_dir = os.path.join(in_path, scan_number)
    method_file = os.path.join(scan_dir, "method")

    # -------------------------------------------------------
    # Helper: safely collect only produced NIfTI files
    # -------------------------------------------------------
    def get_nifti_files(scan_number):
        return [
            f for f in glob.glob(f"*{scan_number}*.nii*")
            if os.path.isfile(f)
        ]

    # ---------- 1) Run brkraw tonii ----------
    cmd = ["brkraw", "tonii", f"{in_path}/", "-s", str(scan_number)]
    subprocess.run(cmd, check=True)

    # ---------- 2) Detect echo count ----------
    NoOfEchoImages = None
    if os.path.exists(method_file):
        with open(method_file) as f:
            for line in f:
                if "PVM_NEchoImages=" in line:
                    echo_str = line.split("=")[1].strip()
                    try:
                        NoOfEchoImages = int(echo_str)
                    except:
                        NoOfEchoImages = 1
                    break

    # ---------- 3) Collect produced NIfTI files ----------
    src_files = get_nifti_files(scan_number)

    if not src_files:
        raise RuntimeError(
            f"No NIfTI files found after brkraw conversion for scan {scan_number}"
        )

    # ---------- 4) Single echo OR unknown echo ----------
    if NoOfEchoImages is None or NoOfEchoImages == 1:
        # Just copy first detected nifti
        shutil.copy(src_files[0], "G1_cp.nii.gz")

    # ---------- 5) Multi-echo ----------
    else:
        merged_file = f"{scan_number}_combined_images.nii.gz"

        # Merge all echoes
        subprocess.run(
            ["fslmerge", "-t", merged_file] + src_files,
            check=True
        )

        shutil.copy(merged_file, "G1_cp.nii.gz")

    # ---------- 6) Fix orientation to LPI ----------
    print(f"{bcolors.NOTIFICATION}Fixing orientation to LPI{bcolors.ENDC}")

    resample = afni.Resample()
    resample.inputs.in_file = "G1_cp.nii.gz"
    resample.inputs.out_file = out_file
    resample.inputs.orientation = "LPI"
    resample.run()

    # ---------- 7) Save NIfTI header info ----------
    with open("NIFTI_file_header_info.txt", "w") as out:
        subprocess.run(["fslhd", out_file], stdout=out, check=True)

    print_statement(
        f"[OK] Bruker → NIFTI workflow completed.",
        bcolors.OKGREEN
    )
def extract_middle_volume(in_file, reference_vol, out_file, size):
  extract_vol = fsl.ExtractROI()
  extract_vol.inputs.in_file=in_file 
  extract_vol.inputs.t_min=reference_vol 
  extract_vol.inputs.t_size=size 
  extract_vol.inputs.roi_file=out_file
  extract_vol.run()

  print("[OK] Intended Volumes extracted.")
  return out_file
def motion_correction(reference_vol, input_vol, output_prefix):

    # ---------- 1) 3dvolreg ----------
    
    volreg = afni.Volreg()  
    volreg.inputs.in_file = input_vol
    volreg.inputs.basefile = reference_vol
    volreg.inputs.out_file = f"{output_prefix}.nii.gz"
    volreg.inputs.oned_file = "motion.1D"
    volreg.inputs.args = '-linear'
    volreg.inputs.oned_matrix_save = "rmsabs.1D"
    volreg.inputs.verbose = True
    volreg.run()

    print("[INFO] Running 3dvolreg…")
    return output_prefix
def plot_motion_parameters(input_file):

    # ---------- 4) Plot motion parameters ----------
    print("[INFO] Creating motion plots…")

    # Translation plots
    data = np.loadtxt(input_file)

    # X-axis (row index / timepoints)
    x = np.arange(data.shape[0])

    # -------- Plot 1: first 3 columns --------
    plt.figure(figsize=(8, 4))
    for i in range(3):
        plt.plot(x, data[:, i], label=f"Column {i+1}")

    plt.title("Rotation")
    plt.xlabel("Volume Number")
    plt.ylabel("Rotation in degrees")
    plt.legend(["Pitch (x)", "Roll (y)", "Yaw (z)"])
    plt.tight_layout()
    plt.savefig("motion_rotations.svg", dpi=1200)
    # -------- Plot 2: next 3 columns --------
    plt.figure(figsize=(8, 4))
    for i in range(3, 6):
        plt.plot(x, data[:, i], label=f"Column {i+1}")

    plt.title("Translation")
    plt.xlabel("Volume Number")
    plt.ylabel("Translation in mm")
    plt.legend(["Read (x)", "Phase (y)", "Slice (z)"])
    plt.tight_layout()
    plt.savefig("motion_translations.svg", dpi=1200)
def masking_file(input_file, mask_file, output_file):
    
    math = fsl.maths.ApplyMask()
    math.inputs.in_file = input_file
    math.inputs.mask_file = mask_file
    math.inputs.out_file = output_file

    math.run()

    print_statement(f"[OK] Masked file saved → {output_file}", bcolors.OKGREEN)
    return output_file


# Helper functions, compulsory to run
def func_param_extract(scan_dir, export_env=True):

    scan_dir = Path(scan_dir)
    acqp_file = scan_dir / "acqp"
    method_file = scan_dir / "method"
    

    if not acqp_file.exists() or not method_file.exists():
        raise FileNotFoundError("acqp or method file not found")

    # -----------------------------
    # Read files
    # -----------------------------
    acqp_text = acqp_file.read_text()
    method_text = method_file.read_text()

    # -----------------------------
    # Sequence name (ACQ_protocol_name)
    # -----------------------------
    seq_match = re.search(
        r"ACQ_protocol_name=\(\s*64\s*\)\s*\n\s*<([^>]+)>",
        acqp_text
    )
    SequenceName = seq_match.group(1) if seq_match else None

    # -----------------------------
    # Extract numeric parameters
    # -----------------------------
    def get_value(pattern, text, cast=int):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    NoOfRepetitions = get_value(r"##\$PVM_NRepetitions=\s*(\d+)", method_text)
    TotalScanTime = get_value(r"##\$PVM_ScanTime=\s*(\d+)", method_text)

    Baseline_TRs = get_value(r"PreBaselineNum=\s*(\d+)", method_text)
    StimOn_TRs = get_value(r"StimNum=\s*(\d+)", method_text)
    StimOff_TRs = get_value(r"InterStimNum=\s*(\d+)", method_text)
    NoOfEpochs = get_value(r"NEpochs=\s*(\d+)", method_text)

    # -----------------------------
    # Derived values
    # -----------------------------
    VolTR_msec = None
    VolTR = None
    MiddleVolume = None

    if NoOfRepetitions and TotalScanTime:
        VolTR_msec = TotalScanTime / NoOfRepetitions
        VolTR = VolTR_msec / 1000
        MiddleVolume = NoOfRepetitions / 2

    # -----------------------------
    # Pack results
    # -----------------------------
    params = {
        "SequenceName": SequenceName,
        "NoOfRepetitions": NoOfRepetitions,
        "TotalScanTime": TotalScanTime,
        "VolTR_msec": VolTR_msec,
        "VolTR": VolTR,
        "Baseline_TRs": Baseline_TRs,
        "StimOn_TRs": StimOn_TRs,
        "StimOff_TRs": StimOff_TRs,
        "NoOfEpochs": NoOfEpochs,
        "MiddleVolume": MiddleVolume,
    }

    # -----------------------------
    # Export to environment (optional)
    # -----------------------------
    if export_env:
        for k, v in params.items():
            if v is not None:
                os.environ[k] = str(v)

    return params
def extract_subject_id(scan_dir):
    subject_file = Path(scan_dir) / "subject"

    if not subject_file.exists():
        raise FileNotFoundError(f"'subject' file not found in {scan_dir}")

    with subject_file.open("r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("##$SUBJECT_id"):
            # The value is expected on the next line
            value_line = lines[i + 1].strip()
            return value_line.strip("<>")

    raise ValueError("##$SUBJECT_id not found in subject file")
def print_header(message, color):
    line = "*" * 134   # same width everywhere
    width = len(line)

    print()
    print(f"{color}{line}{bcolors.ENDC}")
    print(f"{color}{message.center(width)}{bcolors.ENDC}")
    print(f"{color}{line}{bcolors.ENDC}")
    print()
def print_statement(message, color):
    print(f"{color}{message}{bcolors.ENDC}")
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
def view_images(input_image, cmap="gray"):
    img = nib.load(input_image)
    data = img.get_fdata()

    # Handle 4D vs 3D safely
    if data.ndim == 4:
        data = data[..., 0]   # take first volume
    elif data.ndim != 3:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    ny = data.shape[1]
    cols = 8
    rows = int(np.ceil(ny / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        if i < ny:
            ax.imshow(data[:, i, :].T, cmap=cmap, origin="lower")
            ax.set_title(f"y={i}", fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Copying freshly written files within the SMB share on macOS can yield a truncated 4096-byte copy; force client-side copies
if hasattr(shutil, "_HAS_FCOPYFILE"):
    shutil._HAS_FCOPYFILE = False


# Stitching helpers
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


def is_bruker_raw_scan_dir(path):
    p = Path(path)
    return p.is_dir() and (p / "method").is_file()


def bruker_scan_to_nifti(in_path, scan_number, work_dir, out_name):
    """Same steps as bruker_to_nifti(), but isolated in work_dir so several scans don't collide."""
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

    # Skip macOS "._" AppleDouble metadata files created on SMB shares
    src_files = sorted(f for f in work_dir.glob(f"*{scan_number}*.nii*") if f.is_file() and not f.name.startswith("._") and not f.name.endswith("_combined.nii.gz"))
    if not src_files:
        raise RuntimeError(f"No NIfTI files found after brkraw conversion for scan {scan_number} in {work_dir}")

    if n_echo_images is None or n_echo_images == 1:
        resample_input = src_files[0]
    else:
        resample_input = work_dir / f"{scan_number}_combined.nii.gz"
        subprocess.run(["fslmerge", "-t", str(resample_input)] + [str(f) for f in src_files], check=True)
        print_statement(f"[OK] Merged {n_echo_images} echoes -> {resample_input}", bcolors.OKGREEN)

    out_file = work_dir / out_name
    # Called without a shell: brkraw file names contain parentheses, e.g. "...-(E36).nii.gz", which nipype's shell call rejects
    subprocess.run(["3dresample", "-orient", "LPI", "-prefix", str(out_file), "-inset", str(resample_input)], check=True)

    with open(work_dir / "NIFTI_file_header_info.txt", "w") as f:
        subprocess.run(["fslhd", str(out_file)], stdout=f, check=True)

    print_statement(f"[OK] Raw Bruker scan {scan_number} converted -> {out_file}", bcolors.OKGREEN)
    return out_file


def mean_image(input_file, output_file):
    mean = fsl.maths.MeanImage()
    mean.inputs.in_file = input_file
    mean.inputs.out_file = output_file
    mean.run()
    print_statement(f"[OK] Mean image saved -> {output_file}", bcolors.OKGREEN)
    return output_file
