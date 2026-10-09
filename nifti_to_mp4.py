#!/usr/bin/env python
"""
Make an mp4 of the time series of one slice of a 4D NIfTI file.

Examples
--------
    python nifti_to_mp4.py                      (popups ask for the file, the slice and where to save)
    python nifti_to_mp4.py mc_func.nii.gz       (popups ask for the slice and where to save)
    python nifti_to_mp4.py mc_func.nii.gz slice4.mp4 --slice 4
    python nifti_to_mp4.py mc_func.nii.gz slice4.mp4 --slice 4 --axis y --start 300 --end 700 --fps 20
    python nifti_to_mp4.py scm_timeseries_from_sm_sm_mc_func.nii.gz psc.mp4 --slice 4 --cmap RdBu_r --symmetric

Whatever is not given on the command line is asked for in a popup (file, slice, output file). The slice number is the
0-based voxel index along the slice axis of the file (the voxel coordinate shown by fsleyes).
Needs numpy, nibabel, Pillow, tkinter (for the popups) and an ffmpeg executable (found on the PATH, next to this
Python, or via imageio-ffmpeg).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AXES = {"x": 0, "y": 1, "z": 2}
START_FOLDER = Path("/Volumes/pschmidt/fmri_analysis/AnalysedData")


def tk_root():
    try:
        import tkinter as tk
    except ImportError:
        sys.exit("tkinter is not available for the popups: give the input file and --slice on the command line.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def ask_input_file(root):
    from tkinter import filedialog
    path = filedialog.askopenfilename(
        parent=root, title="Choose a 4D NIfTI file", initialdir=str(START_FOLDER if START_FOLDER.exists() else Path.cwd()),
        filetypes=[("NIfTI files", ("*.nii", "*.gz")), ("All files", "*")])
    if not path:
        sys.exit("No file chosen.")
    return path


def ask_axis(root):
    from tkinter import simpledialog
    answer = simpledialog.askstring(
        "Slice axis", "The voxels are isotropic, so the slice axis cannot be guessed.\nWhich voxel axis is the slice axis (x, y or z)?", parent=root)
    if answer is None or answer.strip().lower() not in AXES:
        sys.exit("No valid slice axis given.")
    return AXES[answer.strip().lower()]


def ask_slice(root, n_slices, axis):
    from tkinter import simpledialog
    answer = simpledialog.askinteger(
        "Slice", f"Slice number (0 to {n_slices - 1}, along voxel axis {'xyz'[axis]}):", parent=root,
        minvalue=0, maxvalue=n_slices - 1, initialvalue=n_slices // 2)
    if answer is None:
        sys.exit("No slice chosen.")
    return answer


def ask_output_file(root, default):
    from tkinter import filedialog
    path = filedialog.asksaveasfilename(
        parent=root, title="Save the video as", initialdir=str(default.parent), initialfile=default.name,
        defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
    if not path:
        sys.exit("No output file chosen.")
    return Path(path)


def default_output(input_file, slice_index):
    stem = Path(input_file).name.removesuffix(".gz").removesuffix(".nii")
    return Path(input_file).with_name(f"{stem}_slice{slice_index}.mp4")


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    beside_python = Path(sys.executable).parent / "ffmpeg"
    if beside_python.exists():
        return str(beside_python)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found. Install it (conda install ffmpeg) or activate the environment that has it.")


def default_slice_axis(zooms):
    """The slice direction has the thickest voxels; None if the voxels are isotropic and it cannot be guessed."""
    zooms = np.asarray(zooms[:3], dtype=float)
    if zooms.max() / zooms.min() < 1.05:
        return None
    return int(np.argmax(zooms))


def orient_block(block, in_plane, codes, zooms):
    """
    Rearrange the (axis a, axis b, time) block so that it displays upright: superior (or anterior) up, and
    right (or anterior) to the right, i.e. the left side of the image is the left of the animal.
    Returns the block as (rows, columns, time) and the voxel size in mm of rows and columns.
    """
    vertical = 0
    for letters in ("SI", "AP"):
        found = [k for k, ax in enumerate(in_plane) if codes[ax] in letters]
        if found:
            vertical = found[0]
            break
    horizontal = 1 - vertical
    v_axis, h_axis = in_plane[vertical], in_plane[horizontal]

    if vertical == 1:
        block = block.transpose(1, 0, 2)
    if codes[v_axis] in "SA":  # index increases upwards, but row 0 is drawn at the top
        block = block[::-1]
    if codes[h_axis] not in "RA":  # index increases to the left
        block = block[:, ::-1]
    return block, zooms[v_axis], zooms[h_axis]


def pixel_repeats(z_rows, z_cols, n_rows, n_cols, longest_side):
    """Integer repeats per voxel so that pixels are square and the longest side is about `longest_side` pixels."""
    smallest = min(z_rows, z_cols)
    rel_rows, rel_cols = z_rows / smallest, z_cols / smallest
    scale = max(1, int(longest_side // max(n_rows * rel_rows, n_cols * rel_cols)))
    return max(1, round(rel_rows * scale)), max(1, round(rel_cols * scale))


def make_font(height):
    size = max(12, height // 22)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow without a size argument
        return ImageFont.load_default()


def render_frame(plane, lo, hi, cmap, rep_rows, rep_cols, label, font):
    x = np.clip((np.nan_to_num(plane.astype(np.float32)) - lo) / (hi - lo), 0, 1)
    x = np.repeat(np.repeat(x, rep_rows, axis=0), rep_cols, axis=1)
    if cmap is None:
        rgb = np.repeat((x * 255).astype(np.uint8)[..., None], 3, axis=-1)
    else:
        rgb = (cmap(x)[..., :3] * 255).astype(np.uint8)
    rgb = np.pad(rgb, ((0, rgb.shape[0] % 2), (0, rgb.shape[1] % 2), (0, 0)))  # h264 needs even sizes
    image = Image.fromarray(rgb)
    ImageDraw.Draw(image).text((8, 6), label, fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description="Make an mp4 of the time series of one slice of a 4D NIfTI file.")
    parser.add_argument("input", nargs="?", help="4D NIfTI file (asked for in a popup if missing)")
    parser.add_argument("output", nargs="?", help="output mp4 (default: <input>_slice<N>.mp4 next to the input; asked for in a popup if the file or slice was not given)")
    parser.add_argument("--slice", type=int, help="0-based voxel index of the slice, as in fsleyes (asked for in a popup if missing)")
    parser.add_argument("--axis", choices=AXES, help="voxel axis the slice index refers to (default: the axis with the thickest voxels)")
    parser.add_argument("--fps", type=float, default=50, help="frames per second (default 50)")
    parser.add_argument("--start", type=int, default=0, help="first volume (default 0)")
    parser.add_argument("--end", type=int, help="last volume, excluded (default: all)")
    parser.add_argument("--step", type=int, default=1, help="use every n-th volume (default 1)")
    parser.add_argument("--cmap", help="matplotlib colormap name (default: gray)")
    parser.add_argument("--symmetric", action="store_true", help="colour window symmetric around 0 (for signal change data)")
    parser.add_argument("--percentiles", type=float, nargs=2, default=(1, 99.5), metavar=("LOW", "HIGH"),
                        help="percentiles of the non-zero slice values that set the colour window (default 1 99.5)")
    parser.add_argument("--vmin", type=float, help="lowest displayed value (overrides the percentiles)")
    parser.add_argument("--vmax", type=float, help="highest displayed value (overrides the percentiles)")
    parser.add_argument("--size", type=int, default=640, help="longest side of the video in pixels (default 640)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite an existing output file")
    args = parser.parse_args()

    # Popups for everything that was not given on the command line
    root = tk_root() if args.input is None or args.slice is None else None
    if args.input is None:
        args.input = ask_input_file(root)

    img = nib.load(args.input)
    if img.ndim != 4:
        sys.exit(f"Expected a 4D NIfTI file, got {img.ndim}D (shape {img.shape}).")
    zooms = img.header.get_zooms()
    codes = nib.aff2axcodes(img.affine)

    axis = AXES[args.axis] if args.axis else default_slice_axis(zooms)
    if axis is None:
        if root is None:
            sys.exit("The voxels are isotropic, so the slice axis cannot be guessed: pass --axis x, y or z.")
        axis = ask_axis(root)
    n_slices = img.shape[axis]
    if args.slice is None:
        args.slice = ask_slice(root, n_slices, axis)
    if not 0 <= args.slice < n_slices:
        sys.exit(f"--slice must be between 0 and {n_slices - 1} for axis {'xyz'[axis]} (the file has {n_slices} slices along it).")
    n_vols = img.shape[3]
    end = n_vols if args.end is None else min(args.end, n_vols)
    volumes = range(args.start, end, args.step)
    if len(volumes) == 0:
        sys.exit(f"No volumes selected (the file has {n_vols}).")

    if args.output:
        output = Path(args.output)
    elif root is not None:
        output = ask_output_file(root, default_output(args.input, args.slice))
        args.overwrite = True  # the save dialog already asked about replacing an existing file
    else:
        output = default_output(args.input, args.slice)
    if root is not None:
        root.destroy()
    if output.exists() and not args.overwrite:
        sys.exit(f"{output} already exists. Use --overwrite to replace it.")

    print(f"{args.input}: shape {img.shape}, orientation {''.join(codes)}, voxel size {tuple(round(float(z), 3) for z in zooms[:3])} mm")
    print(f"Slice {args.slice} along axis {'xyz'[axis]}, {len(volumes)} volumes ({volumes.start}..{volumes[-1]}, step {volumes.step})")

    index = [slice(None)] * 4
    index[axis] = args.slice
    index[3] = slice(volumes.start, volumes[-1] + 1, volumes.step)
    block = np.asarray(img.dataobj[tuple(index)], dtype=np.float32)  # (in-plane a, in-plane b, time)
    in_plane = [a for a in range(3) if a != axis]
    block, z_rows, z_cols = orient_block(block, in_plane, codes, zooms)

    if args.vmin is not None and args.vmax is not None:
        lo, hi = args.vmin, args.vmax
    else:
        values = block[np.isfinite(block) & (block != 0)]  # masked data has a zero background
        if values.size == 0:
            values = block[np.isfinite(block)]
        lo, hi = np.percentile(values, args.percentiles)
        if args.symmetric:
            hi = np.percentile(np.abs(values), args.percentiles[1])
            lo = -hi
        lo = lo if args.vmin is None else args.vmin
        hi = hi if args.vmax is None else args.vmax
    if hi <= lo:
        hi = lo + 1
    print(f"Display window: {lo:.4g} to {hi:.4g}")

    cmap = None
    if args.cmap:
        from matplotlib import colormaps
        cmap = colormaps[args.cmap]

    rep_rows, rep_cols = pixel_repeats(z_rows, z_cols, block.shape[0], block.shape[1], args.size)
    height = block.shape[0] * rep_rows
    width = block.shape[1] * rep_cols
    height, width = height + height % 2, width + width % 2
    font = make_font(height)

    tr = float(zooms[3]) if len(zooms) > 3 else 0.0
    seconds = img.header.get_xyzt_units()[1] == "sec" and tr > 0

    command = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i, vol in enumerate(volumes):
            label = f"volume {vol}" + (f"   {vol * tr:.0f} s" if seconds else "")
            process.stdin.write(render_frame(block[:, :, i], lo, hi, cmap, rep_rows, rep_cols, label, font).tobytes())
        process.stdin.close()
    except BrokenPipeError:
        pass
    if process.wait() != 0:
        sys.exit("ffmpeg failed (see its message above).")
    print(f"Saved: {output} ({width}x{height} pixels, {len(volumes)} frames at {args.fps:g} fps)")


if __name__ == "__main__":
    main()
