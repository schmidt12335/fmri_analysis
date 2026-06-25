import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import uniform_filter1d
from scipy import stats
#import ipython

file_path = "/home/pschmidt/fmri_analysis/AnalysedData/FAP/timeseries_group_analysis/FAP_AAV/"

timewindow = [20,1300]

fap_aav_fap = np.array([
    np.loadtxt(file)[timewindow[0]:timewindow[1]]
    for file in glob.glob(os.path.join(file_path, "FAP_AAV_FAP/*.txt"))
])
print(fap_aav_fap)
control_aav_FAP = np.array([
    np.loadtxt(file)[timewindow[0]:timewindow[1]]
    for file in glob.glob(os.path.join(file_path, "Ctrl_AAV_FAP/*.txt"))
])
print(control_aav_FAP)
fap_aav_PBS = np.array([
    np.loadtxt(file)[timewindow[0]:timewindow[1]]
    for file in glob.glob(os.path.join(file_path, "FAP_AAV_PBS/*.txt"))
])
print(fap_aav_PBS)

"""
plt.figure(figsize=(10, 6))
for i in range(fap_aav_fap.shape[0]):
    plt.plot(fap_aav_fap[i], label=f"FAP AAV FAP {i+1}")
for i in range(control_aav_FAP.shape[0]):
    plt.plot(control_aav_FAP[i], label=f"Ctrl AAV FAP {i+1}", linestyle="--")
for i in range(fap_aav_PBS.shape[0]):
    plt.plot(fap_aav_PBS[i], label=f"FAP AAV PBS {i+1}", linestyle=":")
plt.title("Single Subject PSC")
plt.xlabel("Time [s]")
plt.ylabel("PSC [%]")
plt.legend()
"""

mean_fap_aav_fap = np.mean(fap_aav_fap, axis=0)[timewindow[0]:timewindow[1]]
std_fap_aav_fap = np.std(fap_aav_fap, axis=0)[timewindow[0]:timewindow[1]]

mean_control_aav_FAP = np.mean(control_aav_FAP, axis=0)[timewindow[0]:timewindow[1]]
std_control_aav_FAP = np.std(control_aav_FAP, axis=0)[timewindow[0]:timewindow[1]]

mean_fap_aav_PBS = np.mean(fap_aav_PBS, axis=0)[timewindow[0]:timewindow[1]]
std_fap_aav_PBS = np.std(fap_aav_PBS, axis=0)[timewindow[0]:timewindow[1]]

control_grouped = np.concatenate([control_aav_FAP, fap_aav_PBS], axis=0)
mean_control = np.mean(control_grouped, axis=0)[timewindow[0]:timewindow[1]]
std_control = np.std(control_grouped, axis=0)[timewindow[0]:timewindow[1]]


mean_psc_exp_smooth = uniform_filter1d(mean_fap_aav_fap, size=40, axis=0, mode="nearest")
std_psc_exp_smooth = uniform_filter1d(std_fap_aav_fap, size=40, axis=0, mode="nearest")

mean_psc_fap_pbs_smooth = uniform_filter1d(mean_fap_aav_PBS, size=40, axis=0, mode="nearest")
std_psc_fap_pbs_smooth = uniform_filter1d(std_fap_aav_PBS, size=40, axis=0, mode="nearest")

mean_psc_control_fap_smooth = uniform_filter1d(mean_control_aav_FAP, size=40, axis=0, mode="nearest")
std_psc_control_fap_smooth = uniform_filter1d(std_control_aav_FAP, size=40, axis=0, mode="nearest")

mean_psc_control_smooth = uniform_filter1d(mean_control, size=40, axis=0, mode="nearest")
std_psc_control_smooth = uniform_filter1d(std_control, size=40, axis=0, mode="nearest")

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
baseline_window = slice(0, 200)
stim_window = slice(1100, 1300)

labels = ["FAP", "Ctrl"]
x = np.arange(len(labels))  # [0, 1]

# Subject-wise signal changes
fap_aav_fap_signal_changes = (
    np.mean(fap_aav_fap[:, stim_window], axis=1)
    - np.mean(fap_aav_fap[:, baseline_window], axis=1)
)

control_signal_changes = (
    np.mean(control_grouped[:, stim_window], axis=1)
    - np.mean(control_grouped[:, baseline_window], axis=1)
)

# Group means
exp_group_mean = np.mean(fap_aav_fap_signal_changes)
control_group_mean = np.mean(control_signal_changes)

# Group SEMs
exp_group_sem = np.std(fap_aav_fap_signal_changes, ddof=1) / np.sqrt(len(fap_aav_fap_signal_changes))
control_group_sem = np.std(control_signal_changes, ddof=1) / np.sqrt(len(control_signal_changes))

means = [exp_group_mean, control_group_mean]
sems = [exp_group_sem, control_group_sem]


from scipy import stats

t_stat, p_value = stats.ttest_ind(
    fap_aav_fap_signal_changes,
    control_signal_changes,
    equal_var=False  # Welch's t-test, safer
)

print("t =", t_stat)
print("p =", p_value)


fig, ax = plt.subplots(figsize=(10, 6))

# Main time-course plot
ax.plot(mean_psc_exp_smooth, label="FAP", color="C0", zorder=10)
ax.plot(mean_psc_control_fap_smooth, label="Control", color="C1", zorder=10)

ax.fill_between(
    range(len(mean_psc_exp_smooth)),
    mean_psc_exp_smooth - std_psc_exp_smooth,
    mean_psc_exp_smooth + std_psc_exp_smooth,
    color="C0",
    alpha=0.2,
    zorder=9
)

ax.plot(mean_psc_exp_smooth + std_psc_exp_smooth, color="C0", alpha=0.4, zorder=9)
ax.plot(mean_psc_exp_smooth - std_psc_exp_smooth, color="C0", alpha=0.4, zorder=9)

ax.fill_between(
    range(len(mean_psc_control_fap_smooth)),
    mean_psc_control_fap_smooth - std_psc_control_smooth,
    mean_psc_control_fap_smooth + std_psc_control_smooth,
    color="C1",
    alpha=0.2,
    zorder=8
)

ax.plot(mean_psc_control_fap_smooth + std_psc_control_smooth, color="C1", alpha=0.4, zorder=8)
ax.plot(mean_psc_control_fap_smooth - std_psc_control_smooth, color="C1", alpha=0.4, zorder=8)

ax.set_xlabel("Time [s]")
ax.set_ylabel("PSC [%]")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)



groups = [fap_aav_fap_signal_changes, control_signal_changes]
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
    fap_aav_fap_signal_changes,
    control_signal_changes,
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
all_values = np.concatenate([fap_aav_fap_signal_changes, control_signal_changes])

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
ax_inset.set_xticklabels(["FAP", "Control"], fontsize=8)
ax_inset.set_ylabel("ΔPSC [%]", fontsize=8)

ax_inset.tick_params(axis="both", labelsize=8)

ax_inset.spines["top"].set_visible(False)
ax_inset.spines["right"].set_visible(False)

print("p =", p_value)

plt.tight_layout()
plt.show()


fig.savefig("FAP_group_analysis.png", dpi=300)
fig.savefig("FAP_group_analysis.svg", dpi=300)