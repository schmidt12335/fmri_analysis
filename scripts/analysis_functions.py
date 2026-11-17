
def bruker_to_nifti(in_path, scan_number, out_file):

    import os, glob, shutil, subprocess
    from nipype.interfaces import afni

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