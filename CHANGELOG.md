# Changelog

Changes of the `mol_fMRI` pipeline notebook, newest version first. Each version is a separate notebook (`mol_fMRI_v5.x.ipynb`); older notebooks stay unchanged.

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
