#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


CONFIG = {
    # Subject data
    "subject_dir": None,  # Path to single subject or None
    "group_root": "/home/pschmidt/fmri_analysis/QC/QC_in/data",  # Path to group root (multiple subjects) or None
    "subject_glob": "sub*",  # Glob pattern for finding subjects
    
    # Anatomical and functional images
    "struct_pattern": "*struct*.nii*",  # Pattern to find structural image
    "func_pattern": "*func*.nii*",      # Pattern to find functional image
    
    # Atlas and ROIs
    "atlas_reference": "/home/pschmidt/fmri_analysis/QC/QC_in/atlas/SIGMA_InVivo_Brain_Template_Masked.nii",  # Path to atlas reference image or None
    "roi_masks_dir": "/home/pschmidt/fmri_analysis/QC/QC_in/rois",    # Path to ROI masks directory or None
    "roi_mask_glob": "*.nii.gz*",  # Glob pattern for finding ROI masks
    
    # Output directories
    "output_dir": "/home/pschmidt/fmri_analysis/QC/QC_out",  # Directory for aligned images and results or None
    "plot_dir": "/home/pschmidt/fmri_analysis/QC/QC_out",    # Directory for plots or None
}

def find_subject_dirs(group_root: Path, subject_glob: str = "*") -> list[Path]:
    direct_subject_paths = sorted(
        [Path(p) for p in glob.glob(str(group_root / subject_glob)) if Path(p).is_dir()]
    )

    if direct_subject_paths:
        return direct_subject_paths

    nested_subject_paths = sorted(
        [
            p
            for p in group_root.iterdir()
            if p.is_dir() and (any(p.glob("*struct*.nii*")) or any(p.glob("*func*.nii*")))
        ]
    )

    if nested_subject_paths:
        return nested_subject_paths

    raise FileNotFoundError(
        f"No subject directories found in {group_root} using pattern {subject_glob}"
    )

def get_incidence(r_data, r2_data):
    specific = 0
    non_specific = 0
    no_fc = 0
    for i in range(len(r_data)):
        if r_data[i] > 0.1 and r2_data[i] < 0.1:
            specific += 1
        if r_data[i] > 0.1 and r2_data[i] > 0.1:
            non_specific += 1
        if -0.1 < r_data[i] < 0.1 and -0.1 < r2_data[i] < 0.1:
            no_fc += 1
    spurious_fc = len(r_data) - (specific + non_specific + no_fc)
    return np.array([specific/len(r_data)*100, non_specific/len(r_data)*100, no_fc/len(r_data)*100, spurious_fc/len(r_data)*100])


corrs_S1r_S1l = []
corrs_S1l_ACA = []

subject_dirs = find_subject_dirs(Path(CONFIG["group_root"]), CONFIG["subject_glob"])


for i in range(len(subject_dirs)):
    subject_name = subject_dirs[i].name
    print(f'Processing subject: {subject_name}')
    S1bfl = (np.loadtxt(f"/home/pschmidt/fmri_analysis/QC/QC_out/{subject_name}/roi_timeseries_S1bf_l.csv", delimiter=",", skiprows=1)[:,1])
    S1bfr = (np.loadtxt(f"/home/pschmidt/fmri_analysis/QC/QC_out/{subject_name}/roi_timeseries_S1bf_r.csv", delimiter=",", skiprows=1)[:,1])
    ACAl = (np.loadtxt(f"/home/pschmidt/fmri_analysis/QC/QC_out/{subject_name}/roi_timeseries_ACA_l.csv", delimiter=",", skiprows=1)[:,1])
    corrs_S1r_S1l.append(np.corrcoef(S1bfl, S1bfr)[0,1])
    corrs_S1l_ACA.append(np.corrcoef(S1bfl, ACAl)[0,1])


data_x = corrs_S1r_S1l
data_y = corrs_S1l_ACA

xmin = -1
xmax = 1
ymin = -1
ymax = 2
dx = 0.0005
x = np.arange(xmin, xmax, dx)
y = np.arange(ymin, ymax, dx)
incidence_all = []


incidence_all.append(get_incidence(data_x, data_y))

total_incidence = np.sum(incidence_all, axis = 0)
fc_categories = ['Specific FC', 'Non-specific FC', 'No FC', 'Spurious FC']
print(f'total incidence = {total_incidence}', f'incidence ={incidence_all}')

labels = [f'AC QC GSR', f'AC GSR', f'AC QC Rabies', f'AC Rabies GSR']

"""all_fc = {}
for i in range(len(fc_categories)):
    all_fc[fc_categories[i]] = np.array([
        incidence_all[0][i],
        incidence_all[1][i],
        incidence_all[2][i],
        incidence_all[3][i]])"""



fig, ax = plt.subplots(figsize=(10, 5))
for i in range(len(subject_dirs)):
    subject_name = subject_dirs[i].name
    ax.plot(corrs_S1r_S1l[i], corrs_S1l_ACA[i], 'o', label=f'{subject_name}')
ax.set_xlabel('Correlation S1bf left to right (r)')
ax.set_ylabel('Correlation S1bf left to ACA (r)')
ax.set_xlim([-0.3, 1])
ax.set_ylim([-0.7, 1])
ax.legend(fontsize = 8, markerscale = 0.75, frameon = False, title = 'Dataset', title_fontsize = 9, labelspacing = 0.2, handlelength = 0, borderaxespad = 2, draggable = True)
ax.hlines(y = 0.1, xmin=-0.1, xmax=0.8, linestyle='--', color='k')
ax.vlines(x = 0.1, ymin=-0.4, ymax=0.8, linestyle='--', color='k')
ax.hlines(y = -0.1, xmin=-0.1, xmax=0.1, linestyle='--', color='k')
ax.vlines(x = -0.1, ymin=-0.1, ymax=0.1, linestyle='--', color='k')
ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8], labels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], labels = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
pos = [[0.85, -0.6], [0.85, 0.2], [0, 0.2], [-0.1, -0.6]]
for i in range(len(fc_categories)):
    if total_incidence[i] == 100 or total_incidence[i] == 0:
        ax.text(x = pos[i][0], y = pos[i][1], s = f'{total_incidence[i]:.0f}%\n{fc_categories[i]}', ha='center', va='center')
    else:
        ax.text(x = pos[i][0], y = pos[i][1], s = f'{total_incidence[i]:.2f}%\n{fc_categories[i]}', ha='center', va='center')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

ax.legend()
plt.show()