import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import uniform_filter1d
from scipy import stats
#import ipython

file_path = "/home/pschmidt/fmri_analysis/AnalysedData/Sero/group_data/"

timewindow = [20,2400]

base_start = 300
base_end = 500

inj_start = 600
inj_end = 1200

sig_start = 1800
sig_end = 2000

def semi_std(data, axis=0):
    """Asymmetric (semi) standard deviation.

    Returns (upper_std, lower_std), where upper_std is computed only from
    samples above the mean and lower_std only from samples below the mean,
    instead of a single symmetric std applied to both sides.
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    diff = data - mean
    upper_sq = np.where(diff > 0, diff ** 2, np.nan)
    lower_sq = np.where(diff < 0, diff ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        upper_std = np.sqrt(np.nanmean(upper_sq, axis=axis))
        lower_std = np.sqrt(np.nanmean(lower_sq, axis=axis))
    upper_std = np.nan_to_num(upper_std, nan=0.0)
    lower_std = np.nan_to_num(lower_std, nan=0.0)
    return upper_std, lower_std

sero_aav_sero1mM = np.array([
    np.loadtxt(file)[timewindow[0]:timewindow[1]]
    for file in glob.glob(os.path.join(file_path, "1mM/*.txt"))
])
print(sero_aav_sero1mM)

sero_aav_sero150uM = np.array([
    np.loadtxt(file)[timewindow[0]:timewindow[1]]
    for file in glob.glob(os.path.join(file_path, "150uM/*.txt"))
])
print(sero_aav_sero150uM)

    
mean_sero_aav_sero1mM = np.mean(sero_aav_sero1mM, axis=0)[timewindow[0]:timewindow[1]]
std_upper_sero_aav_sero1mM, std_lower_sero_aav_sero1mM = semi_std(sero_aav_sero1mM, axis=0)
std_upper_sero_aav_sero1mM = std_upper_sero_aav_sero1mM[timewindow[0]:timewindow[1]]
std_lower_sero_aav_sero1mM = std_lower_sero_aav_sero1mM[timewindow[0]:timewindow[1]]

mean_sero_aav_sero150uM = np.mean(sero_aav_sero150uM, axis=0)[timewindow[0]:timewindow[1]]
std_upper_sero_aav_sero150uM, std_lower_sero_aav_sero150uM = semi_std(sero_aav_sero150uM, axis=0)
std_upper_sero_aav_sero150uM = std_upper_sero_aav_sero150uM[timewindow[0]:timewindow[1]]
std_lower_sero_aav_sero150uM = std_lower_sero_aav_sero150uM[timewindow[0]:timewindow[1]]




mean_psc_sero_aav_sero1mM_smooth = uniform_filter1d(mean_sero_aav_sero1mM, size=40, axis=0, mode="nearest")
std_upper_psc_sero_aav_sero1mM_smooth = uniform_filter1d(std_upper_sero_aav_sero1mM, size=40, axis=0, mode="nearest")
std_lower_psc_sero_aav_sero1mM_smooth = uniform_filter1d(std_lower_sero_aav_sero1mM, size=40, axis=0, mode="nearest")

mean_psc_sero_aav_sero150uM_smooth = uniform_filter1d(mean_sero_aav_sero150uM, size=40, axis=0, mode="nearest")
std_upper_psc_sero_aav_sero150uM_smooth = uniform_filter1d(std_upper_sero_aav_sero150uM, size=40, axis=0, mode="nearest")
std_lower_psc_sero_aav_sero150uM_smooth = uniform_filter1d(std_lower_sero_aav_sero150uM, size=40, axis=0, mode="nearest")


"""
plt.figure(figsize=(10, 6))
plt.plot(mean_fap_aav_fap, label="Mean Signal Change")
plt.plot(mean_control_aav_FAP, label="Mean Control AAV FAP", linestyle="--")
plt.plot(mean_fap_aav_PBS, label="Mean FAP AAV PBS", linestyle=":")
plt.fill_between(range(len(mean_fap_aav_fap)), mean_fap_aav_fap - std_fap_aav_fap, 
                 mean_fap_aav_fap + std_fap_aav_fap, alpha=0.3, label="Standard Deviation")
plt.fill_between(range(len(mean_control_aav_FAP)), mean_control_aav_FAP - std_control_aav_FAP,
                mean_control_aav_FAP + std_control_aav_FAP, alpha=0.3, 
                label="Control AAV FAP Std Dev", color="orange")
plt.fill_between(range(len(mean_fap_aav_PBS)), mean_fap_aav_PBS - std_fap_aav_PBS,
                mean_fap_aav_PBS + std_fap_aav_PBS, alpha=0.3, label="FAP AAV PBS Std Dev", color="green")
plt.title("Mean Group PSC")
plt.xlabel("Time [s]")
plt.ylabel("PSC [%]")
plt.legend()

plt.figure(figsize=(10, 6))

plt.plot(mean_psc_exp_smooth, label="FAP-AAV + FAP")
plt.plot(mean_control_aav_FAP, label="Ctrl-AAV + FAP")
plt.plot(mean_fap_aav_PBS, label="FAP-AAV + PBS")

plt.fill_between(range(len(mean_psc_exp_smooth)), mean_psc_exp_smooth - std_psc_exp_smooth, mean_psc_exp_smooth + std_psc_exp_smooth, 
                 alpha=0.3)
plt.fill_between(range(len(mean_psc_control_fap_smooth)), mean_psc_control_fap_smooth - std_psc_control_fap_smooth,
                mean_psc_control_fap_smooth + std_psc_control_fap_smooth, alpha=0.3, color="orange")
plt.fill_between(range(len(mean_psc_fap_pbs_smooth)), mean_psc_fap_pbs_smooth - std_psc_fap_pbs_smooth,
                mean_psc_fap_pbs_smooth + std_psc_fap_pbs_smooth, alpha=0.3, color="green")

plt.title("Mean Group PSC Smoothed")
plt.xlabel("Time [s]")
plt.ylabel("PSC [%]")
plt.legend()
"""



# Define baseline and stimulation windows
baseline_window = slice(base_start, base_end)
stim_window = slice(sig_start, sig_end)

labels = ["FAP", "Ctrl"]
x = np.arange(len(labels))  # [0, 1]

# Subject-wise signal changes
sero_aav_sero1mM_signal_changes = (
    np.mean(sero_aav_sero1mM[:, stim_window], axis=1)
    - np.mean(sero_aav_sero1mM[:, baseline_window], axis=1)
)

sero_aav_sero150uM_signal_changes = (
    np.mean(sero_aav_sero150uM[:, stim_window], axis=1)
    - np.mean(sero_aav_sero150uM[:, baseline_window], axis=1)
)

# Group means
exp_group_mean = np.mean(sero_aav_sero1mM_signal_changes)
control_group_mean = np.mean(sero_aav_sero150uM_signal_changes)

# Group SEMs
exp_group_sem = np.std(sero_aav_sero1mM_signal_changes, ddof=1) / np.sqrt(len(sero_aav_sero1mM_signal_changes))
control_group_sem = np.std(sero_aav_sero150uM_signal_changes, ddof=1) / np.sqrt(len(sero_aav_sero150uM_signal_changes))

means = [exp_group_mean, control_group_mean]
sems = [exp_group_sem, control_group_sem]


from scipy import stats

t_stat, p_value = stats.ttest_ind(
    sero_aav_sero1mM_signal_changes,
    sero_aav_sero150uM_signal_changes,
    equal_var=False  # Welch's t-test, safer
)

print("t =", t_stat)
print("p =", p_value)


fig, ax = plt.subplots(figsize=(10, 6))

# Main time-course plot
ax.plot(mean_psc_sero_aav_sero1mM_smooth, label="1mM", color="C0", zorder=10)
ax.plot(mean_psc_sero_aav_sero150uM_smooth, label="Sero 150µM", color="C1", zorder=10)

ax.fill_between(
    range(len(mean_psc_sero_aav_sero1mM_smooth)),
    mean_psc_sero_aav_sero1mM_smooth - std_lower_psc_sero_aav_sero1mM_smooth,
    mean_psc_sero_aav_sero1mM_smooth + std_upper_psc_sero_aav_sero1mM_smooth,
    color="C0",
    alpha=0.2,
    zorder=9
)

ax.plot(mean_psc_sero_aav_sero1mM_smooth + std_upper_psc_sero_aav_sero1mM_smooth, color="C0", alpha=0.4, zorder=9)
ax.plot(mean_psc_sero_aav_sero1mM_smooth - std_lower_psc_sero_aav_sero1mM_smooth, color="C0", alpha=0.4, zorder=9)

ax.fill_between(
    range(len(mean_psc_sero_aav_sero150uM_smooth)),
    mean_psc_sero_aav_sero150uM_smooth - std_lower_psc_sero_aav_sero150uM_smooth,
    mean_psc_sero_aav_sero150uM_smooth + std_upper_psc_sero_aav_sero150uM_smooth,
    color="C1",
    alpha=0.2,
    zorder=8
)

ax.plot(mean_psc_sero_aav_sero150uM_smooth + std_upper_psc_sero_aav_sero150uM_smooth, color="C1", alpha=0.4, zorder=8)
ax.plot(mean_psc_sero_aav_sero150uM_smooth - std_lower_psc_sero_aav_sero150uM_smooth, color="C1", alpha=0.4, zorder=8)
plt.axvspan(base_start, base_end, color='green', alpha=0.3, label='Baseline Period')
plt.axvspan(sig_start, sig_end, color='blue', alpha=0.3, label='Signal Period')
plt.axvspan(inj_start, inj_end, color='red', alpha=0.3, label='Injection Period')

ax.set_xlabel("Time [s]")
ax.set_ylabel("PSC [%]")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)



groups = [sero_aav_sero1mM_signal_changes, sero_aav_sero150uM_signal_changes]
x_inset = np.arange(2)

ax_inset = ax.inset_axes([0.075, 0.7, 0.125, 0.25])  # [left, bottom, width, height]

# Bars
ax_inset.bar(
    x_inset,
    means,
    yerr=sems,
    capsize=4,
    color=["C0", "C1"],
    edgecolor="black",
    linewidth=1,
    alpha=0.75,
    zorder=2
)

# Individual subject dots
for i, group in enumerate(groups):
    ax_inset.scatter(
        np.full(len(group), x_inset[i]),
        group,
        color="black",
        s=18,
        zorder=3
    )

# Statistical test
t_stat, p_value = stats.ttest_ind(
    sero_aav_sero1mM_signal_changes,
    sero_aav_sero150uM_signal_changes,
    equal_var=False
)

def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."

stars = p_to_stars(p_value)

# Significance bracket position
all_values = np.concatenate([sero_aav_sero1mM_signal_changes, sero_aav_sero150uM_signal_changes])

y_min = np.min(all_values)
y_max = np.max(all_values)
y_range = y_max - y_min

if y_range == 0:
    y_range = 1

y_sig = y_max + 0.20 * y_range
h = 0.08 * y_range

# Expand ylim BEFORE/AFTER drawing so bracket is visible
ax_inset.set_ylim(y_min - 0.15 * y_range, y_sig + h + 0.25 * y_range)

# Significance bracket
ax_inset.plot(
    [x_inset[0], x_inset[0], x_inset[1], x_inset[1]],
    [y_sig, y_sig + h, y_sig + h, y_sig],
    color="black",
    linewidth=1,
    zorder=10
)

# Stars / n.s.
ax_inset.text(
    np.mean(x_inset),
    y_sig + h,
    stars,
    ha="center",
    va="bottom",
    fontsize=10,
    zorder=10
)

# Inset style
ax_inset.axhline(0, color="black", linewidth=0.8)

ax_inset.set_xticks(x_inset)
ax_inset.set_xticklabels([f"Sero \n1mM", f"Sero \n150µM"], fontsize=8)
ax_inset.set_ylabel("ΔPSC [%]", fontsize=8)

ax_inset.tick_params(axis="both", labelsize=8)

ax_inset.spines["top"].set_visible(False)
ax_inset.spines["right"].set_visible(False)

print("p =", p_value)

plt.tight_layout()
plt.show()


fig.savefig("/home/pschmidt/fmri_analysis/AnalysedData/Sero/group_data/Sero_group_analysis.png", dpi=300)
fig.savefig("/home/pschmidt/fmri_analysis/AnalysedData/Sero/group_data/Sero_group_analysis.svg", dpi=300)