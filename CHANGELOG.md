# Changelog

Two parts: the injection pipeline notebook `mol_fMRI_v5.x.ipynb` (below, newest version first; each version is a separate notebook, older notebooks stay unchanged) and the [electrical stimulation notebooks](#electrical-stimulation-notebooks) (at the end; not versioned, so the entries are dated).

## v5.8

Two options that are **still in testing**; both are off by default, so the notebook behaves as v5.7 unless they are switched on.

### Added
- Automatic brain masks, chosen separately for the structural and the functional mask in two dropdowns at the top (hand-drawn in fsleyes stays the default; an existing mask is never replaced).
  - Structural mask: `3dSkullStrip -rat` combined with `3dAutomask` (`3dAutomask` alone if 3dSkullStrip fails). 3dSkullStrip alone can cut off the darker ventral brain along a straight line in images with a strong surface-coil intensity fall-off; 3dAutomask reaches the ventral edge, so the union is used. In tests against hand-drawn masks of 14 rat sessions (inside the slices covered by the hand-drawn mask) the overlap (Dice) was 0.97, lowest 0.96 (the same as 3dSkullStrip alone). It can be used after a look at the quality-check figure.
  - Functional mask: made from the structural mask (hand-drawn or automatic), which is moved to the functional grid through the image headers and trimmed with an intensity threshold. Dice 0.89 for EPI (lowest 0.72, 13 sessions) and 0.85 for sordino (3 sessions) with an automatic or a hand-drawn structural mask. Clearly less accurate: it always needs to be checked, and it is wrong if the animal moved between the two scans. Without a drawn cannula mask the brain mask is also used as the outline.
  - Quality check for automatic masks (`mask_quality_check`): overlay figures `mask_qc_structural.png` and `mask_qc_functional.png` (red = mask, cyan dashed = structural mask before the trimming) and warnings for separate pieces, empty slices, a mask at the image border, low edge contrast and, for the functional mask, a large change by the intensity trimming. The checks only catch gross failures; for functional masks they could not tell the weaker masks from the good ones, so the figure has to be looked at. Automatic masks are recorded in `<mask>.source.json`; a hand-drawn mask saved over an automatic one counts as hand-drawn again.
  - Tested on rat data only. The test scripts (`scripts/automask_test.py`, `scripts/automask_coreg_test.py`) are in the repository. With automatic instead of hand-drawn masks the functional → structural registration fitted equally well in 6 sessions (transforms differed by 0.4 mm and 2° on average).
- 3D automatic ROIs: the dropdown "Block depth" in the automatic ROI section makes the ROI a 3×3×3 voxel block over three consecutive slices instead of a 3×3 block in one slice (default). The whole block has to lie inside the striatum and the despeckled mask. The z-scores of 3D blocks are not comparable with those of the 3×3 blocks.

## v5.7

### Added
- SORDINO scans (zero echo time, 3D radial) are supported next to EPI; the rest of the pipeline is unchanged. The sequence is detected from the `##$Method` entry of the Bruker `method` file of the functional scan (`detect_sequence_type`), can be forced with the new "Sequence" dropdown (auto/EPI/SORDINO), and is asked for if both give nothing.
- `set_volume_tr`: for a sordino scan the converted `func.nii.gz` gets the real volume TR written into its header (`PVM_ScanTime / PVM_NRepetitions`, e.g. 2.4 s for 3200 spokes). Bruker only stores the 0.75 ms spoke TR in `PVM_RepetitionTime`, so brkraw writes a wrong TR (1 s). The fix (AFNI `3drefit -TR`) changes the header in place and never touches the voxel data, so it also repairs an existing conversion and is safe to re-run.

### Changed
- Distortion correction cannot be selected for a sordino scan (no echo-planar readout, so no EPI distortion); the parameter cell stops with an error.
- The coregistration of a sordino scan is rigid body only: there is no phase-encoding axis, so the optional scaling along it (see v5.5) and its axis detection are skipped.
- The volume TR (`tr`) is a float instead of an integer, so 2.4 s is no longer truncated to 2 s.

### Notes
- The baseline/signal start indices and the window size are still given in volumes, and a sordino volume covers more time than an EPI volume. Adjust the defaults for sordino.
- Several back-to-back sordino scans are not stitched in this notebook; use the `mol_fMRI_sordino_*` notebooks for that.

## v5.6

- The brain edge for the exclusion mask and for the coregistration is the outline that includes the cannulas (`mask_outline_with_cannulas.nii.gz`, from `mask_mean_mc_func_cannulas.nii.gz`), so the notch where a cannula enters the brain is not trimmed as if it were a brain edge. The candidate voxels (`mask_exclusion_candidates.nii.gz`) are the brain voxels inside that outline after the 2-voxel line-wise shrink; the cannula path itself stays excluded. The coregistration uses `cleaned_outline_mean_mc_func.nii.gz`. Without a cannula mask the brain mask is used, as before.
- Despeckled maps are saved alongside the existing ones, which stay untouched. `mask_despeckled.nii.gz` is the exclusion mask without speckle: small in-plane clusters (fewer than 5 voxels) that reach 3 noise SDs of PSC, and voxels above 2 noise SDs whose time course correlates with their in-plane neighbours with r < 0.3 (gaps of up to 4 voxels that only this check would create are filled again). It gives `clean_despeckled_scm_from_sm_*.nii.gz` and, in structural space, `despeckled_signal_change_map_coregistered_structural_space.nii.gz`, where clusters smaller than one functional voxel (in mm²) are removed in addition. The limits are arguments of `create_despeckled_mask` in the signal change map cell.
- Existing results are not overwritten silently: if a result already exists (motion correction, smoothed data, coregistration), you are asked whether to overwrite it, and Enter keeps it. A result that is older than its input is marked OUTDATED and then Enter overwrites it. Set `OVERWRITE_POLICY` in the step-selection cell to `"never"` or `"always"` to skip the question. Derived maps and masks inside the dated result folder (signal change maps, exclusion mask, structural-space masks) are cheap and are always recomputed. Hand-drawn masks and the converted func/struct NIfTI files are never overwritten (an existing conversion is skipped).
- `coregistration_afni` removes its old output files before it runs. `3dAllineate` refuses to overwrite them without any error, so a re-run could leave an outdated coregistered signal change map next to a new transform.
- Optional cells at the end run `exclusion_diagnostics`: which exclusion rule (and the shrunk mask and structural restriction) removes which voxels, with a figure of all slices (`exclusion_diagnostics.png`), a reason map (`exclusion_reason_map.nii.gz`) and a report. If ROIs exist, it also shows how much of each ROI, and of voxels with a similar time course, is lost. It changes no map or mask.
- Optional automatic ROIs (before the ROI cell): per side, the 3×3 voxel block (one slice) with the strongest signal change inside the striatum of the SIGMA rat atlas replaces drawing the ROIs by hand. The protein/sensor, delivery route (AAV or direct) and the acutely injected substance are asked for in a small form and go into the ROI file names. The selection is described in the cell and written to `automatic_roi_summary.txt`; `automatic_roi_qc.png` shows the striatum outlines and the chosen blocks.

## v5.5

- Motion correction: two passes (cubic, then heptic against the mean of pass 1), brain-weighted, no more `-linear`.
- Spatial smoothing is done inside the brain mask (normalized convolution) before the signal change maps are computed.
- The exclusion mask (`mask_thresholded.nii.gz`) replaces the 15–80 percentile cut. A voxel is kept only if it is at least 3 voxels from the brain edge, has a baseline of at least 50% of the median, and a baseline tSNR (`tsnr_baseline_*.nii.gz`) of at least 60% of the median. The mask is moved to structural space with the coregistration affine (`struct_mask_from_functional.nii.gz`) and intersected with the structural brain (`struct_mask_final.nii.gz`), which removes the isolated dead voxels inside clusters and the voxels outside the structural brain.
- A coregistration quality check (`coregistration_qc`) compares the registration with a header-only alignment, warns about implausible transforms, and writes `coregistration_qc.png` (overlay) and `coregistration_qc.txt`. It only reports; it does not change the transform.
- Coregistration may scale the functional image along the phase-encoding axis (within 0.92–1.08, rotations ≤ 5°, shifts ≤ 2 mm) to compensate EPI stretching. The axis comes from the "Phase-encoding axis" dropdown or is detected from brkraw's BIDS sidecar (cached in `func.detected_pe.json`); the scaling is switched off when distortion correction is used or the axis is unknown.

## Scripts

### plot_slice_stack.py (2026-11-20)
Now one tool for three 3D views of a map on the atlas template (modelled on the figures of the deConfUSIon toolbox, SonerCgu/deConfUSIon), chosen with `--view` / the "View" dropdown of the form:
- **stack** with two layouts. `tiled` (new default, `--layout tiled`): the slices are flat tiles side by side in rows (n tiles evenly spread over the extent of the map, `--n-slices`, default 8), each a little offset downwards from its left neighbour, seen from a slightly oblique angle with an orthographic camera (`--tile-view` "paper oblique", "shallow oblique", "face-on" or "custom"), so that every slice is completely visible; the thickness is exaggerated (`--depth-gain`) so the sides show; spacing `--tile-gap`, offset `--tile-rise`, `--columns`, `--reverse-order`; every tile has a position label. `physical` (`--layout physical`): the previous view with slabs standing behind each other with gaps.
- **volume**: volume rendering, a translucent brain with the map as a glowing volume inside (`--volume-opacity`).
- **surface**: a translucent brain surface with nested iso-surfaces of the map (`--surface-levels`, positive and negative values).
- All views: figure (png/pdf/svg, with colorbar), interactive window (`--interactive`) and rotating video (`--movie file.mp4`, `--movie-seconds`; the stacks sway to both sides, volume and surface turn around once). Needs `imageio-ffmpeg` for the video (already in `requirements.txt`).
- The map is drawn with an opacity ramp (`--alpha-min`, default 0.55, up to 1 at the end of the colour scale): weak values stay translucent and the anatomy shows through. Colours of weak values are therefore blended with the template; the colorbar is exact at the top.
- The dark outline around the slices is off by default (`--outline` to switch it on); the form is scrollable and has headings. Old saved form entries are still read.
- Earlier today: rebuilt as a 3D rendering with pyvista / VTK (`pip install pyvista`, now in `requirements.txt`; the old matplotlib renderer was removed): smooth closed slab solids, depth buffer, lighting and anti-aliasing, gray shaded sides, the outline follows the bright tissue so the dark rim of the template does not show. Interactive viewer (drag = rotate, scroll = zoom, shift + drag = move, `s` = save the view, `q` = close); in the form the rendering and the viewer run in their own processes.
- Note: with "mean" over the slice thickness a small or sparse effect can drop below the threshold and disappear; use `--projection max` (the form says so).
- Tested on the mouse (0.1 mm) template in coronal, sagittal and axial direction, all three views, interactive screenshot and video export.

## Electrical stimulation notebooks

The notebooks for the analysis of electrical stimulation experiments (EPI: `mol_fMRI_electric_stim_*.ipynb`, SORDINO: `mol_fMRI_sordino_*.ipynb`, with `electric_stim_helpers.py`, `electric_stim_session.py`, `electric_stim_atlas.py`, `sordino_helpers.py`) have **one version number for the whole set**, EPI and SORDINO together. It is the constant `STIM_NOTEBOOKS_VERSION` in `electric_stim_session.py`, printed by the first code cell of every notebook and saved as `stim_notebooks_version` in the `session.json` of every preprocessing run (an analysis notebook notes if a session was preprocessed with another version). A change of the notebooks gets a new version here and a new value of the constant.

Versions up to 2.1 were assigned afterwards from the git history (the dates are those of the commits; several commit messages are only "minor changes", so those entries only say which notebooks were touched). See [docs/electric_stim_analysis.md](docs/electric_stim_analysis.md) for the description of the notebooks.

### v2.2 (2026-11-19)
- Version numbers for the stimulation notebooks (see above). Sessions made before this version show "unknown (made before version numbers)".
- `mol_fMRI_sordino_analysis.ipynb`: the stimulation timing that is not stored in the data is entered in a new input cell instead of being fixed in the code: Stimulation ON (required, no default, so a wrong value is never used silently), Stimulation OFF (default: the rest of the trigger block), hemodynamic lag and the OFF window at the end of the block. The trigger blocks read from the method files are shown next to it.
- `mol_fMRI_sordino_analysis.ipynb`: a trigger block that is at least twice as long as the shortest one (e.g. 500 volumes after ten blocks of 25 volumes) is the post-stimulation phase of the same scan and is no longer counted as an additional epoch. This allows one scan per stimulation with the stimulation blocks and the post-stimulation phase together.
- `mol_fMRI_sordino_poststim_analysis.ipynb`: if the last scan ends with such a long trigger block, the default baseline window starts at the beginning of the post-stimulation phase and the signal window ends at the end of the scan (the window length is at most half of the phase).
- 2026-11-17: standalone scripts (`mol_fMRI_electric_stim_stitching.py`, `mol_fMRI_sordino_stitching.py`, `roi_atlas_analysis.py`) moved to `scripts/`; self-contained old notebooks moved to `archive/`. The notebooks still import the helper modules from the root folder.

### v2.1 (2026-11-11)
- Improvements to the SORDINO analysis notebooks; `mol_fMRI_sordino_preprocessing.ipynb` added to the repository (2026-11-06). `electric_stim_session.py` and the EPI notebooks received small changes at the same time.
- Changes to the SORDINO post-stimulation notebook and to the EPI preprocessing notebook (2026-11-11).

### v2.0 (2026-10-29)
- SORDINO (zero echo time, 3D radial) stimulation notebooks added: `mol_fMRI_sordino_preprocessing.ipynb`, `mol_fMRI_sordino_analysis.ipynb`, `mol_fMRI_sordino_poststim_analysis.ipynb`, with `sordino_helpers.py` and `mol_fMRI_sordino_stitching.py`. The volume TR is `PVM_ScanTime / PVM_NRepetitions` (the converted NIfTIs carry a wrong TR) and the epochs come from the trigger blocks of the method file, because the stimulation paradigm is not stored in it.
- The older single-notebook version of the EPI analysis is kept as `mol_fMRI_elstim_old.ipynb`.

### v1.2 (2026-10-12)
- Option to normalise the signal level of the scans in the preprocessing, which removes jumps between stitched scans (`scan_normalization.csv`) (2026-10-08).
- The data folder is no longer hard-coded: `fmri_paths.py` asks once per workstation and saves the folder (`fmri_root.txt`, or the environment variable `FMRI_ROOT`) (2026-10-09).
- Changes to the coregistration in `mol_fMRI_electric_stim_analysis.ipynb` (2026-10-12).

### v1.1 (2026-10-07)
- Distortion correction (reverse phase-encoded EPI, FSL topup) added to the EPI preprocessing; adjustments to the EPI analysis notebook (2026-10-06).
- Fixes in the distortion correction and the new post-stimulation (LTP) analysis notebook, `mol_fMRI_electric_stim_poststim_analysis.ipynb` (2026-10-07).

### v1.0 (2026-10-05)
- New separate notebooks for the preprocessing (`mol_fMRI_electric_stim_preprocessing.ipynb`) and the analysis (`mol_fMRI_electric_stim_analysis.ipynb`) of electrical stimulation experiments.
