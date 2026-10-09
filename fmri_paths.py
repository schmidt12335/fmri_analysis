"""Location of the shared fMRI data folder (the one with RawData, AnalysedData, atlas, ...), without hardcoded paths.

The root is resolved in this order:
1. the environment variable FMRI_ROOT,
2. the folder saved in `fmri_root.txt` next to this module (machine specific, not tracked by git),
3. otherwise the user is asked for the folder once and it is saved in `fmri_root.txt`.
Delete `fmri_root.txt` to be asked again.
"""
import os
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


def _ask_for_root():
    print("Enter the path of the folder that contains your fMRI data folders (RawData, AnalysedData, atlas).")
    while True:
        answer = input("fMRI data folder (empty to cancel): ").strip().strip("'\"")
        if not answer:
            raise FileNotFoundError("No fMRI data folder given.")
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
