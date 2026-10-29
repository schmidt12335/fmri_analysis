"""Helpers for the SORDINO (zero echo time, 3D radial) stimulation notebooks.

The sordino scans differ from the EPI scans of electric_stim_helpers.py in these points:
  * A volume is a 3D radial acquisition, so there are no slices and PVM_RepetitionTime (0.75 ms) is the time per spoke.
    The volume TR is PVM_ScanTime / PVM_NRepetitions (NPro spokes per volume, 2.4 s for NPro = 3200).
  * The method file has no stimulation paradigm (no PreBaselineNum, StimNum, NEpochs, FMRIYesNo). The scanner only sends a
    TTL pulse (to the Master-9, which runs the stimulation train) at the start of the scan (StartOfScanTrigger) and at the
    start of every block (BlockTriggerMode, NTrigBlocks, NVolumesPerBlock).
"""
import re
import subprocess
from pathlib import Path

from electric_stim_helpers import func_param_extract, print_statement, bcolors


def _method_value(text, key):
    """Raw text of a Bruker parameter (everything between '##$key=' and the next '##' or '$$' line), or None."""
    m = re.search(rf"^##\${key}=(.*?)(?=^##|^\$\$)", text, re.S | re.M)
    return m.group(1).strip() if m else None


def _parse_int_list(raw):
    """Bruker array like '( 10 ) 12 12 12' or '( 30 ) @30*(12)' -> list of ints."""
    if raw is None:
        return []
    body = re.sub(r"^\(\s*\d+\s*\)", "", raw).strip()
    values = []
    for token in re.findall(r"@(\d+)\*\((\d+)\)|(\d+)", body):
        if token[0]:
            values += [int(token[1])] * int(token[0])
        else:
            values.append(int(token[2]))
    return values


def sordino_param_extract(scan_dir):
    """Timing and trigger information of a sordino scan.
    block_onsets are the 0-based volume indices (within the scan) at which a block trigger is sent."""
    scan_dir = Path(scan_dir)
    method_text = (scan_dir / "method").read_text(errors="ignore")
    params = func_param_extract(scan_dir, export_env=False)

    def yes(key):
        return (_method_value(method_text, key) or "").lower() in ("yes", "on")

    method_name = re.search(r"^##\$Method=<(?:\w+:)?([^>]+)>", method_text, re.M)
    n_rep = params["NoOfRepetitions"]
    block_mode = yes("BlockTriggerMode")
    volumes_per_block = _parse_int_list(_method_value(method_text, "NVolumesPerBlock")) if block_mode else []
    block_onsets = []
    start = 0
    for n in volumes_per_block:
        block_onsets.append(start)
        start += n

    params.update({
        "Method": method_name.group(1) if method_name else None,
        "is_sordino": bool(method_name and method_name.group(1) == "sordino"),
        "n_spokes": int(_method_value(method_text, "NPro") or 0),
        "spoke_TR_ms": float(_method_value(method_text, "PVM_RepetitionTime") or 0),
        "BlockTriggerMode": block_mode,
        "StartOfScanTrigger": yes("StartOfScanTrigger"),
        "InterVolumeDummySpokes": yes("InterVolumeDummySpokes"),
        "TriggerPulseDuration_ms": float(_method_value(method_text, "TriggerPulseDuration") or 0),
        "DummyScansDur_ms": float(_method_value(method_text, "PVM_DummyScansDur") or 0),
        "volumes_per_block": volumes_per_block,
        "block_onsets": block_onsets,
    })
    if block_mode and sum(volumes_per_block) != n_rep:
        print_statement(f"{scan_dir.name}: NVolumesPerBlock sums to {sum(volumes_per_block)} but the scan has {n_rep} repetitions.", bcolors.FAIL)
    return params


def find_trigger_scan(in_path, scan_numbers):
    """Index of the scan with block triggers (the stimulation scan), or None."""
    for i, num in enumerate(scan_numbers):
        if sordino_param_extract(Path(in_path) / num)["BlockTriggerMode"]:
            return i
    return None


def merge_with_tr(scan_files, out_file, tr):
    """fslmerge -tr: concatenates the scans in time and writes the real volume TR to the header.
    The converted NIfTIs carry TR = 1 s because Bruker only stores the 0.75 ms spoke TR."""
    subprocess.run(["fslmerge", "-tr", str(out_file)] + [str(f) for f in scan_files] + [f"{tr:.6f}"], check=True)
    print_statement(f"[OK] Scans concatenated -> {Path(out_file).name} (TR {tr:.4f} s)", bcolors.OKGREEN)
    return out_file
