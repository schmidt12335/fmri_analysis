"""Session handling for the electrical stimulation notebooks.

The preprocessing notebook writes a session.json into its analysis folder. The analysis notebook loads it
to continue on the preprocessed data without re-running anything. Folders without a session.json (created
by older runs) are reconstructed from the folder layout.
"""
import csv
import datetime
import json
import re
from pathlib import Path

SESSION_FILE = "session.json"
STITCHED_PATTERN = re.compile(r"stitched_(\d+(?:-\d+)*)_(.+)")
PATH_KEYS = ("in_path", "analysed_func_dir", "analysed_struct_dir", "analysis_dir")

# Set by the preprocessing notebook; makes run_step recompute outputs that already exist
FORCE_RERUN = False


def save_session(session_dir, **state):
    out = {k: (str(v) if isinstance(v, Path) else v) for k, v in state.items()}
    out["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    path = Path(session_dir) / SESSION_FILE
    path.write_text(json.dumps(out, indent=2))
    return path


def read_scan_ranges(analysed_func_dir):
    """(scan, start_volume, end_volume) of every scan in the stitched series, end inclusive."""
    csv_path = Path(analysed_func_dir) / "scan_boundaries.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return [(int(r["scan"]), int(r["start_volume"]), int(r["end_volume"])) for r in csv.DictReader(f)]


def infer_session(analysis_dir):
    """Rebuild the session of a folder that was created before session.json existed."""
    analysis_dir = Path(analysis_dir)
    analysed_func_dir = analysis_dir.parent
    match = STITCHED_PATTERN.fullmatch(analysed_func_dir.name)
    if not match:
        raise ValueError(f"{analysis_dir} is not inside a 'stitched_<scans>_<sequence>' folder and has no {SESSION_FILE}.")
    func_scan_numbers = match.group(1).split("-")
    subject_dir = analysed_func_dir.parent

    # Structural folders are named <scan number><sequence>; take the one that was already processed
    candidates = [d for d in subject_dir.iterdir() if d.is_dir() and re.match(r"\d+", d.name)
                  and (d / "struct.nii.gz").exists()]
    candidates.sort(key=lambda d: not (d / "cleaned_struct.nii.gz").exists())
    if not candidates:
        raise FileNotFoundError(f"No processed structural folder (with struct.nii.gz) found in {subject_dir}.")
    analysed_struct_dir = candidates[0]
    struct_file = analysed_struct_dir / ("cleaned_struct.nii.gz" if (analysed_struct_dir / "cleaned_struct.nii.gz").exists()
                                         else "struct.nii.gz")

    parts = list(subject_dir.parts)
    if "AnalysedData" not in parts:
        raise ValueError(f"'AnalysedData' not found in {subject_dir}; cannot locate the raw data.")
    parts[parts.index("AnalysedData")] = "RawData"
    in_path = Path(*parts)
    if not in_path.is_dir():
        raise FileNotFoundError(f"Raw data folder {in_path} not found. It is needed for the stimulation timing in the Bruker method file.")

    return {
        "in_path": in_path,
        "subject_id": subject_dir.name,
        "func_scan_numbers": func_scan_numbers,
        "struct_scan_number": re.match(r"\d+", analysed_struct_dir.name).group(0),
        "analysed_func_dir": analysed_func_dir,
        "analysed_struct_dir": analysed_struct_dir,
        "analysis_dir": analysis_dir,
        "drop_vols": int(drop.group(1)) if (drop := re.search(r"_drop(\d+)", analysed_func_dir.name)) else 0,
        "mask_file": str(analysed_func_dir / "mask_mean_mc_func.nii.gz"),
        "structural_file_for_coregistration": str(struct_file),
    }


def resolve_analysis_dir(selected):
    """Accepts the analysis folder itself, its stitched_* folder or the subject folder above it."""
    selected = Path(selected)
    if (selected / SESSION_FILE).exists() or STITCHED_PATTERN.fullmatch(selected.parent.name):
        return selected
    if STITCHED_PATTERN.fullmatch(selected.name) and (selected / "analysis").is_dir():
        return selected / "analysis"
    candidates = sorted(d / "analysis" for d in selected.glob("stitched_*") if (d / "analysis").is_dir()
                        and STITCHED_PATTERN.fullmatch(d.name))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"{selected} contains several analysis folders, select one of them:\n  "
                         + "\n  ".join(map(str, candidates)))
    return selected


def load_session(analysis_dir):
    """Returns (session, source) where source is the session file or 'inferred from folder layout'."""
    analysis_dir = resolve_analysis_dir(analysis_dir)
    path = analysis_dir / SESSION_FILE
    if path.exists():
        session = json.loads(path.read_text())
        for key in PATH_KEYS:
            if key in session:
                session[key] = Path(session[key])
        return session, str(path)
    return infer_session(analysis_dir), "inferred from folder layout"


def require_files(paths, hint):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing) + f"\n{hint}")


def up_to_date(outputs, inputs=()):
    """True if every output exists, is not empty and is newer than all inputs."""
    outputs = [Path(o) for o in outputs]
    if FORCE_RERUN or not all(o.exists() and o.stat().st_size > 0 for o in outputs):
        return False
    newest_input = max((Path(i).stat().st_mtime for i in inputs if Path(i).exists()), default=0)
    return all(o.stat().st_mtime >= newest_input for o in outputs)


def run_step(outputs, inputs, func, *args, **kwargs):
    """Runs func unless its outputs are already up to date with respect to its inputs."""
    if up_to_date(outputs, inputs):
        print(f"\033[92mSkipping {func.__name__}: {', '.join(Path(o).name for o in outputs)} already up to date.\033[0m")
        return None
    return func(*args, **kwargs)
