# Electrical stimulation fMRI analysis

Two notebooks for EPI and SORDINO (zero echo time, 3D radial):

| Step | Notebook |
|---|---|
| 1. Preprocessing | `mol_fMRI_electric_stim_preprocessing_v2.4.ipynb` |
| 2. Analysis: 2a. acute stimulation response and/or 2b. post-stimulation change (LTP) | `mol_fMRI_electric_stim_analysis_v2.4.ipynb` |

Run the preprocessing notebook once per session, then the analysis notebook on its result. The sequence is detected from the Bruker `method` file of the first functional scan (or chosen in the "Sequence" dropdown) and saved in `session.json`; the analysis notebook takes it from there. The two parts of the analysis (2a, 2b) are ticked in its first cell and can be run separately; the atlas registration is done once for both. The analysis notebook does not recompute anything of the preprocessing, so it can be started at any time, also on sessions preprocessed with the older notebooks.

**Version:** the notebooks and the helper modules share one version number (currently 2.4, the constant `STIM_NOTEBOOKS_VERSION` in `electric_stim_session.py`). It is printed by the first code cell of both notebooks and saved in the `session.json` of every preprocessing run; the analysis notebook notes if the session was preprocessed with another version. The history is in the [CHANGELOG](../CHANGELOG.md#electrical-stimulation-notebooks).

**Older versions:** the six v2.3 notebooks (one preprocessing, stimulation response and post-stimulation notebook per sequence: `mol_fMRI_electric_stim_*.ipynb` for EPI, `mol_fMRI_sordino_*.ipynb` for SORDINO) are in `archive/v2.3/`. They import the helper modules from the root folder, so copy a notebook back to the root folder to run it. `mol_fMRI_elstim_old.ipynb` is the old single-notebook version.

Helper modules (in the root folder, imported by the notebooks): `electric_stim_helpers.py`, `electric_stim_session.py`, `electric_stim_atlas.py` (shared), `sordino_helpers.py` (sordino-specific, and the sequence detection), `brain_masks.py` (automatic brain masks). Command-line scripts in `scripts/`: `mol_fMRI_electric_stim_stitching.py` and `mol_fMRI_sordino_stitching.py` (the stitching step), `roi_atlas_analysis.py` (atlas ROI time courses).

## 1. Preprocessing

Input: the raw Bruker session folder (`RawData/...`) and the scan numbers of the functional scans **in acquisition order** (e.g. pre-stimulation, stimulation, post-stimulation) plus the structural scan.

1. **Convert and stitch.** The functional scans (same session, same subject, acquired back to back with identical parameters) are converted to NIfTI and concatenated into one series, `func.nii.gz`. `scan_boundaries.csv` records the volume range of each original scan.
   - EPI: the first volumes of every scan are dropped before stitching, because a new scan is not yet in steady state (a bright first volume that decays over a few volumes).
   - SORDINO: no volumes are dropped by default (10 s of dummy scans precede every scan; it can still be set). The converted NIfTIs carry a wrong TR (1 s), so the stitched series is written with the real volume TR, `PVM_ScanTime / PVM_NRepetitions` (2.4 s for 3200 spokes).
2. **Distortion correction (EPI only, optional).** Reverse phase-encoded EPI and FSL topup. Off by default. The corrected data goes to its own `..._distcorr` folder.
3. **Scan level normalisation (optional, off by default).** Scales each scan so the signal level continues across the stitching points. It also removes real differences between scans, so keep the baseline in the same scan as the signal window when it is used.
4. **Brain mask** (`mask_mean_mc_func.nii.gz`): drawn by hand in fsleyes on `middle_vol.nii.gz` (default), reused from another folder, or made automatically ("Functional mask" dropdown). The automatic functional mask is made from the structural brain mask, which is moved to the functional grid through the image headers (same session, no registration): EPI with nearest neighbour and an intensity trimming, SORDINO as a partial-volume fraction (≥ 30% of a voxel) without trimming, limited to a slab of structural slices ("SORDINO slab", default 20 slices, shifted +3 towards anterior). Overlap with hand-drawn masks in tests: EPI 0.89 (down to 0.7), SORDINO 0.87–0.93. Always look at the quality-check figure `mask_qc_functional.png`; if the outline does not follow the brain, delete the mask and draw it by hand. The mask is wrong if the animal moved between the structural and the functional scan. An existing mask is never replaced; a hand-drawn mask saved over an automatic one counts as hand-drawn (`<mask>.source.json`).
5. **Motion correction** of the whole stitched series against its middle volume (`mc_func.nii.gz`, `motion.1D`), then the mean image `mean_mc_func.nii.gz`.
6. **Structural scan** conversion and masking (`struct.nii.gz`, `struct_mask.nii.gz`, `cleaned_struct.nii.gz`). The structural mask is drawn by hand (default) or made automatically ("Structural mask" dropdown: `3dSkullStrip -rat` on the head, combined with `3dAutomask` near it; overlap with hand-drawn masks 0.94–0.97), with the quality-check figure `mask_qc_structural.png`. It needs a standard anatomy of the same session (e.g. 256 × 256, 25.6 mm FOV, 32 slices); on 700 × 700 anatomies 3dSkullStrip failed.

Every step is skipped if its output exists and is newer than its inputs; tick *Force re-run* to recompute everything. The processing options are part of the folder name (`stitched_<scans>_<sequence>_distcorr_drop10_norm`), so runs with different settings never overwrite each other. The results are written to the stitched folder and to an `analysis` folder inside it, which also gets a `session.json` that the analysis notebooks load.

## 2a. Stimulation response analysis

Pick the `analysis` folder of a finished preprocessing run (the one with `session.json`).

1. **Stimulation setup.** The stimulation scan is located and the epochs are derived.
   - EPI: the scan with `FMRIYesNo=Yes`; epoch timing from `PreBaselineNum`, `StimNum`, `InterStimNum`, `NEpochs` in the `method` file. Stimulation 5 Hz, 3 mA, 40 s ON / 20 s OFF per 60 s epoch (the ON/OFF split is an input field, default 40 / 20 s). Percent signal change is relative to the pre-stimulation baseline; the comparison window is the rest right before each onset.
   - SORDINO: the `method` file has no paradigm. The scanner only sends a trigger at the start of the scan and of every block (`BlockTriggerMode`, `NTrigBlocks`, `NVolumesPerBlock`); the stimulator (Master-9) runs the train. Each trigger block is one epoch; the ON length (**Stimulation ON**, `STIM_ON_S`) is not in the data, so you enter it in an input cell of the notebook (required; it **must match the Master-9 setting**, e.g. 40 s). The OFF time is the rest of the trigger block unless you enter it, and the hemodynamic lag and the length of the OFF window can be changed there as well. The block set in ParaVision must be longer than the train. A much longer final block (at least twice the shortest one) is the post-stimulation phase of the same scan and is not an epoch, so one scan can hold both the stimulation blocks and the post-stimulation phase; stitching several scans is optional.
2. **Atlas registration.** The structural scan is registered to the functional mean and to the atlas template; the two rigid transforms are combined into one functional → atlas matrix, and the atlas labels (not the functional data) are resampled onto the functional grid once. Choose the species in the widget: rat (SIGMA in vivo template) or mouse (Allen CCFv3 regions on the DevCCF P56 MRI template, from v2.3; before: DSURQE). Other templates can be entered. A folder `roi_atlas_<species>/` that was registered to another atlas is moved to `previous_atlas_<date>/` and the registration is done again.
3. **Region-wise check.** For every atlas region the mean time course is extracted, and the ON window (after a hemodynamic lag) is compared with the comparison window (EPI: the rest before the onset; SORDINO: the OFF window at the end of the same block) by a paired t-test across epochs, with multiple-comparison correction. No hand-drawn ROIs are needed.
4. **Voxelwise maps in atlas space** (for group analysis), in a single resampling step: `stim_diff_psc` (mean ON − rest, %) and `stim_tstat` (paired t across epochs), saved with an `_atlasspace` suffix. By default only signal increases are kept (`POSITIVE_ONLY`).

Main outputs: `atlas_roi_stim_response.csv`, `atlas_roi_responders.csv`, `atlas_roi_timeseries_all.csv`, `atlas_roi_stim_response_map.nii.gz`, `atlas_roi_stim_response_top_*.svg`, `stim_diff_psc*.nii.gz`, `stim_tstat*.nii.gz`.

## 2b. Post-stimulation analysis (LTP)

Looks at the slow signal change after the stimulation instead of the acute response.

1. Set a **baseline window** and a **signal window** of equal length on the stitched series (start indices in volumes, 0-based; the window duration in seconds, converted with the volume TR). By default the baseline starts 50 s into the last (post-stimulation) scan and the signal window at about 3/4 of it, with a window of 300 s. If that scan ends with a long trigger block (SORDINO: stimulation and post-stimulation phase in one scan), the baseline window starts at the beginning of that phase and the signal window ends at the end of the scan.
2. Percent signal change is relative to the mean of the baseline window. Both windows are cut into bins (`BIN_S`, 30 s) and a Welch t-test compares signal bins with baseline bins in every atlas region (same registration as in 2a, stored in `roi_atlas_<species>/` and reused).
3. Voxelwise maps in atlas space: `poststim_diff_psc` and `poststim_tstat`.

Main outputs: `poststim_roi_response.csv`, `poststim_roi_responders.csv`, `poststim_roi_psc_timeseries_all.csv`, `poststim_roi_response_map.nii.gz`, `poststim_roi_response_top_*.svg`, `poststim_diff_psc*.nii.gz`, `poststim_tstat*.nii.gz`.
