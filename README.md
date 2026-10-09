Setting up script for Data Analysis

# Prerequisites

## Make sure you have the following installed:

* Python (3.8 or newer)
* Visual Studio Code
* pip (comes with Python)

## To check Python installation:

* python --version
or
* python3 --version

# Follow the step to run Jupyter notebook in VSC
## Step 1: Install Required VS Code Extensions

* Open VS Code → Extensions (Ctrl+Shift+X) and install:
	* Python (by Microsoft)
	* Jupyter (by Microsoft)

Restart VS Code after installation.

## Step 2: Select Your Installed Python Interpreter

* Open VS Code
* Press Ctrl+Shift+P (macOS: Cmd+Shift+P)
* Search and select:
	* Python: Select Interpreter
* Choose your installed Python (example):
	* /usr/bin/python3
	* ~/miniforge3/bin/python
	* C:\Python311\python.exe

✔ This Python will be used by Jupyter.

## Step 3: Install Jupyter in That Python Environment

* Open VS Code Terminal:
* Click on Terminal → New Terminal
* Run:
	* pip install jupyter notebook ipykernel
* In terminal verify:
	* jupyter --version

## Step 4: Open or Create a Jupyter Notebook
* Option A: Open Existing Notebook
	* code example.ipynb
* Option B: Create New Notebook
	* In VS Code → File → New File
	* Save as: my_notebook.ipynb

VS Code will automatically open it in Jupyter mode.

## Step 5: Select Python Kernel for Notebook

* Open the .ipynb file
* Click Kernel Selector (top-right)
* Choose:
	* Python (your-selected-interpreter)
	- for example: Python 3.11 (miniforge)

✔ Your notebook is now linked to your installed Python.

## Step 6: Run Notebook Cells

* Run single cell: Shift + Enter
* Run all cells: Run → Run All
* Restart kernel: Kernel → Restart Kernel

## Step 7: Test Installation

* Run this in a notebook cell:
	- import sys
	- print(sys.executable)

Output should match your selected Python path.


# Installing python dependencies to run script

All the that are needed to run the data analysis script have been listed out in the 
requirements file. To install these dependencies, run the following command: 

 - conda install -r requirements.txt

brkraw can not be installed with conda, therefore install it with pip as shown below. Also, there is no version 0.4.0 currently, so i used 0.3.11. Newer versions do not seem to have brkraw tonii (probably different name).

pip uninstall brkraw
pip install --no-deps brkraw==0.3.11 

⚠️ Please note that AFNI and FSL must be installed separately and available in your system PATH.
Nipype only provides Python interfaces to these tools.

## Popup-driven filtering and decomposition workflow

`filtering_decomposition_v24.py` is the Python version of the BLuSH filtering,
QC, SCM, and decomposition workflow. It no longer requires
`cleaned_mc_func.nii.gz` or the mask to be in the current directory. Start it
from VS Code or a terminal:

```bash
python filtering_decomposition_v24.py
```

The script opens dialogs for the cleaned functional NIfTI, an optional mask,
and the output folder. Numeric settings are requested in popup dialogs, and
the component plots remain interactive: click components to drop them and
press Enter to save the selected decomposition.



## PS Group Analysis Script - 28.05.26


The repository includes a group-level analysis script for averaging subject-level percent signal change data:

- `FAP_group_analysis.py`

## What it does

- Averages subject-level percent signal change NIfTI maps into a single group map
- Averages ROI percent signal change time series across subjects
- Saves a group-average map, a CSV of ROI time series, and an optional plot

## Expected folder structure

Each subject should be stored in its own folder under a shared group directory. Example:

```
/group_data/
  subject01/
    subject01_scm.nii.gz
    PSC_time_series_roi_example.txt
  subject02/
    subject02_scm.nii.gz
    PSC_time_series_roi_example.txt
  ...
```

Match subject-level filenames using glob patterns in the script arguments.

## Example usage

Average PSC maps and ROI time series together:

```bash
python FAP_group_analysis.py \
  --group-root /path/to/group_data \
  --map-pattern "*scm*.nii.gz" \
  --map-output group_average_signal_change_map.nii.gz \
  --roi-pattern "PSC_time_series_*.txt" \
  --roi-output group_average_roi_timeseries.csv \
  --roi-plot group_average_roi_timeseries.png
```

Only average PSC maps:

```bash
python FAP_group_analysis.py --group-root /path/to/group_data --map-output group_average_signal_change_map.nii.gz --no-roi
```

Only average ROI time series:

```bash
python FAP_group_analysis.py --group-root /path/to/group_data --roi-output group_average_roi_timeseries.csv --no-map
```

Register each subject to a common atlas before averaging PSC maps:

```bash
python FAP_group_analysis.py \
  --group-root /path/to/group_data \
  --map-pattern "*scm*.nii.gz" \
  --atlas-reference /path/to/atlas_template.nii.gz \
  --struct-pattern "*struct*.nii.gz" \
  --atlas-cache-dir /path/to/group_data/atlas_registered_maps \
  --map-output group_average_signal_change_map_atlas.nii.gz
```

If you omit `--group-root`, the script will open a popup folder selection dialog so you can choose the subject data folder interactively.

# Data folder location

The notebooks find the shared data folder (the one containing `RawData`, `AnalysedData` and `atlas`) via `fmri_paths.py`, in this order:

1. the environment variable `FMRI_ROOT`
2. the folder saved in `fmri_root.txt` in the repository folder (ignored by git)
3. otherwise a folder selection popup opens and saves it in `fmri_root.txt`

So you are asked once per machine. Delete `fmri_root.txt` to choose another folder.
