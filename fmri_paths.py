"""Location of the shared fMRI data folder (the one with RawData, AnalysedData, atlas, ...), without hardcoded paths.

The root is resolved in this order:
1. the environment variable FMRI_ROOT,
2. the folder saved in `fmri_root.txt` next to this module (machine specific, not tracked by git),
3. otherwise a folder selection popup opens once (typing the path is the fallback if no popup is possible) and the
   choice is saved in `fmri_root.txt`.
Delete `fmri_root.txt` to be asked again.
"""
import os
import subprocess
import sys
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "fmri_root.txt"
DATA_FOLDERS = ("RawData", "AnalysedData", "atlas")


def _saved_root():
    if os.environ.get("FMRI_ROOT"):
        return "environment variable FMRI_ROOT", os.environ["FMRI_ROOT"]
    if CONFIG_FILE.is_file():
        lines = CONFIG_FILE.read_text().strip().splitlines()
        if lines and lines[0].strip():
            return str(CONFIG_FILE), lines[0].strip()
    return None, None


_DIALOG_CODE = (
    "import tkinter as tk\n"
    "from tkinter import filedialog\n"
    "root = tk.Tk()\n"
    "root.withdraw()\n"
    "root.attributes('-topmost', True)\n"
    "print(filedialog.askdirectory(title='Select the fMRI data folder (contains RawData, AnalysedData, atlas)'))\n"
)


def _select_folder():
    """Folder selection popup (run outside the notebook kernel so it cannot disturb it).
    Returns '' when cancelled and None when no popup is possible."""
    if sys.platform == "darwin":
        command = ["osascript", "-e", 'POSIX path of (choose folder with prompt '
                   '"Select the fMRI data folder (contains RawData, AnalysedData, atlas)")']
    else:
        command = [sys.executable, "-c", _DIALOG_CODE]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        # osascript exits with an error when the dialog is cancelled
        return "" if sys.platform == "darwin" and "-128" in result.stderr else None
    return result.stdout.strip()


def _ask_for_root():
    print("Select the folder that contains your fMRI data folders (RawData, AnalysedData, atlas).")
    use_popup = True
    while True:
        answer = _select_folder() if use_popup else None
        if answer is None:
            use_popup = False
            answer = input("Popup not available. fMRI data folder path (empty to cancel): ").strip().strip("'\"")
        if not answer:
            raise FileNotFoundError("No fMRI data folder selected.")
        path = Path(answer).expanduser()
        if not path.is_dir():
            print(f"'{answer}' is not an existing folder, try again.")
            continue
        if not any((path / name).is_dir() for name in DATA_FOLDERS):
            print(f"Warning: none of {', '.join(DATA_FOLDERS)} found in '{path}'.")
        CONFIG_FILE.write_text(str(path) + "\n")
        print(f"Saved in {CONFIG_FILE}; delete this file to choose another folder.")
        return str(path)


def get_fmri_root():
    """Return the fMRI data root as a string; ask for it (and remember it) when it is not known yet."""
    source, value = _saved_root()
    if value is None:
        return _ask_for_root()
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"The fMRI data folder '{value}' from {source} does not exist. "
                                "Fix or delete it to be asked again.")
    return str(path)
