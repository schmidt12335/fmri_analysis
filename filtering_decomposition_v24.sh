#!/bin/bash
# =====================================================================
#  BLuSH v24-fix1
#
#  ✔ No LPI enforcement
#  ✔ Timestamped PNGs (global + grids)
#  ✔ Robust TR detection
#  ✔ Works for BOTH 3D + 4D datasets
#  ✔ Full pipeline: trimming → smoothing → detrending → filtering → SCM → decomposition
#  ✔ Filtering step repaired (HP/LP/BP/NOTCH fully works)
#  ✔ SAFE DEFAULTS prevent “unbound variable” errors
#  ✔ SCM allowed for 3D datasets
#  ✔ Component GUI with hover + CTRL marker + timestamped grid PNGs
#  ✔ HTML QC summary (timestamps included)
# =====================================================================

set -Eeuo pipefail
trap 'echo "[ERROR] line ${LINENO}" >&2' ERR
shopt -s nullglob
export LC_NUMERIC=C
export PYTHONWARNINGS="ignore"

# ---------------- Colors ----------------
RED="\033[38;5;160m"; PURPLE="\033[38;5;135m"; GREEN="\033[92m"
BLUE="\033[38;5;33m"; RESET="\033[0m"; BOLD="\033[1m"

timestamp(){ date +"%Y%m%d_%H%M%S"; }

# ---------------- Folders ----------------
mkdir -p soner_pipeline/{filtering,scm_outputs,logs,detrended}
mkdir -p soner_pipeline/pca_outputs/{raw,hp,lp,bp,notch,detrended,trim,smoothed}
mkdir -p soner_pipeline/qc/{global,grid}

MASK="mask_mean_mc_func.nii.gz"
[[ ! -f cleaned_mc_func.nii.gz ]] && { echo "[ERR] cleaned_mc_func.nii.gz not found"; exit 1; }
[[ ! -f "$MASK" ]] && MASK=""

# ---------------- Helpers ----------------
center_line(){
  printf "\n────────────────────────────────────────────────────────\n✳️  %s ✳️\n────────────────────────────────────────────────────────\n" "$1"
}

# --- TR detection (robust, avoids tuple errors) ---
get_tr(){
python3 - "$1" <<'PY'
import nibabel as nib, sys
try:
    hdr = nib.load(sys.argv[1]).header
    zooms = list(hdr.get_zooms())
    if len(zooms) < 4:
        print("1.000000")
    else:
        print(f"{float(zooms[3]):.6f}")
except:
    print("1.000000")
PY
}

# =====================================================================
# GLOBAL MEAN PNG
# =====================================================================
save_global_png(){
  local nii="$1"; local mask="$2"; local tr="$3"; local label="$4"
  python3 - "$nii" "$mask" "$tr" "$label" <<'PY'
import sys, os, nibabel as nib, numpy as np, matplotlib.pyplot as plt, time

f,m,tr,label=sys.argv[1:5]
tr=float(tr)

img=nib.load(f)
d=img.get_fdata()

# Handle 3D case safely
if d.ndim == 3:
    print("[WARN] 3D data → global signal computed as mean of whole volume (single value).")
    sig = np.array([d.mean()])
    x = np.array([0])
else:
    try:
        mask = nib.load(m).get_fdata()>0
        d = d[mask]
    except:
        d = d.reshape(-1, d.shape[-1])
    sig = d.mean(0)
    x = np.arange(len(sig)) * tr

plt.figure(figsize=(10,4))
plt.plot(x, sig, 'k', lw=1)
plt.title(f"Global Mean – {os.path.basename(f)}")
plt.xlabel("Time (s)")
plt.ylabel("Mean Intensity")
plt.grid(True, alpha=0.3)
plt.tight_layout()

out=f"soner_pipeline/qc/global/global_{label}_{time.strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(out, dpi=160)
plt.close()
print("[OK] Global PNG:", out)
PY
}

# =====================================================================
# INTERACTIVE GLOBAL SIGNAL PLOT
# =====================================================================
plot_global_signal(){
python3 - "$1" "$2" "$3" <<'PY'
import sys, nibabel as nib, numpy as np, matplotlib.pyplot as plt

f, m, tr = sys.argv[1], sys.argv[2], float(sys.argv[3])
img = nib.load(f)
d = img.get_fdata()

if d.ndim == 3:
    print("[WARN] 3D file → Only one global value available.")
    sig = np.array([d.mean()])
    x = np.array([0])
else:
    try:
        mask = nib.load(m).get_fdata()>0
        d = d[mask]
    except:
        d = d.reshape(-1, d.shape[-1])
    sig = d.mean(0)
    x = np.arange(len(sig))*tr

plt.figure(figsize=(10,4))
plt.plot(x, sig, 'k')
plt.title(f"Global Mean Signal – {f}")
plt.xlabel("Time (s)")
plt.ylabel("Intensity")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
PY
}

# =====================================================================
# SCM CREATOR
# =====================================================================
scm_blush_fast(){
  local func="$1"; base="$2"; bend="$3"; sig="$4"; send="$5"; mask="$6"; outd="$7"
  mkdir -p "$outd"

  # 3D handling
  if [[ $(python3 - <<EOF
import nibabel as nib
print(1 if nib.load("$func").get_fdata().ndim==3 else 0)
EOF
) == 1 ]]; then
    echo "[WARN] SCM on 3D → Only single-volume operations."
    cp "$func" "$outd/baseline_${base}_${bend}.nii.gz"
    cp "$func" "$outd/signal_${sig}_${send}.nii.gz"
    cp "$func" "$outd/signal_change_map_${base}_${bend}_${sig}_${send}.nii.gz"
    cp "$func" "$outd/norm_func_${base}_${bend}.nii.gz"
    return
  fi

  3dTstat -mean -prefix "$outd/baseline_${base}_${bend}.nii.gz" "$func[${base}..${bend}]"
  3dTstat -mean -prefix "$outd/signal_${sig}_${send}.nii.gz"   "$func[${sig}..${send}]"

  fslmaths "$outd/signal_${sig}_${send}" -sub "$outd/baseline_${base}_${bend}" \
           -div "$outd/baseline_${base}_${bend}" -mul 100 ${mask:+ -mas "$mask"} \
           "$outd/signal_change_map_${base}_${bend}_${sig}_${send}.nii.gz"

  fslmaths "$func" -sub "$outd/baseline_${base}_${bend}" \
           -div "$outd/baseline_${base}_${bend}" -mul 100 ${mask:+ -mas "$mask"} \
           "$outd/norm_func_${base}_${bend}.nii.gz"
}

# =====================================================================
#  STEP 1: TRIMMING
# =====================================================================
TR=$(get_tr cleaned_mc_func.nii.gz)
TMP="cleaned_mc_func.nii.gz"

echo -e "${RED}${BOLD}====== BLuSH v24-fix1 ======${RESET}"
# ============================================================
#  ORIENTATION CHECK (AFNI vs FSL)
# ============================================================

echo -e "${BLUE}${BOLD}=== ORIENTATION REPORT ===${RESET}"

# --- SAFE: get FSL orientation ---
if FSL_MODE=$(fslorient cleaned_mc_func.nii.gz 2>/dev/null || true); then
    :
fi
[[ -z "$FSL_MODE" ]] && FSL_MODE="UNKNOWN"

# --- SAFE: get AFNI voxel axes ---
if AFNI_ORIENT=$(3dinfo -orient cleaned_mc_func.nii.gz 2>/dev/null || true); then
    :
fi
[[ -z "$AFNI_ORIENT" ]] && AFNI_ORIENT="UNKNOWN"

# --- SAFE: detect AFNI display mode ---
if [[ "${AFNI_LEFT_IS_LEFT:-NO}" == "NO" ]]; then
    AFNI_MODE="Radiological (Left=Right)"
else
    AFNI_MODE="Neurological (Left=Left)"
fi

echo -e "FSL display:      ${GREEN}$FSL_MODE${RESET}"
echo -e "AFNI voxel axes:  ${GREEN}$AFNI_ORIENT${RESET}"
echo -e "AFNI display:     ${GREEN}$AFNI_MODE${RESET}"

echo -e "${BLUE}${BOLD}=============================${RESET}\n"


# =====================================================================
# STEP 1: TRIMMING
# =====================================================================
echo -e "${PURPLE}${BOLD}STEP 1: TRIMMING${RESET}"
center_line "Trim initial/end volumes for stable baseline"

read -p "Trim from START (vols): " TSTART
read -p "Trim from END   (vols): " TEND

IS3D=$(python3 - <<EOF
import nibabel as nib
print(1 if nib.load("cleaned_mc_func.nii.gz").get_fdata().ndim==3 else 0)
EOF
)

if [[ "$IS3D" == "1" ]]; then
    echo "[WARN] 3D file → trimming skipped."
else
    if [[ "${TSTART:-0}" != "0" || "${TEND:-0}" != "0" ]]; then
        NV=$(python3 - <<EOF
import nibabel as nib
print(nib.load("cleaned_mc_func.nii.gz").shape[-1])
EOF
)
        END=$(( NV - TEND - 1 ))
        OUT="trimmed_${TSTART}-${TEND}_cleaned_mc_func_$(timestamp).nii.gz"
        3dTcat -prefix "$OUT" cleaned_mc_func.nii.gz"[${TSTART}..${END}]"
        TMP="$OUT"
    fi
fi

save_global_png "$TMP" "$MASK" "$TR" "trim"

# =====================================================================
# STEP 2: SMOOTHING
# =====================================================================
echo -e "${PURPLE}${BOLD}STEP 2: SMOOTHING${RESET}"
center_line "Temporal smoothing + Spatial smoothing"

AVAILABLE=($(ls -tr cleaned_mc_func.nii.gz trimmed_*.nii.gz 2>/dev/null))
for i in "${!AVAILABLE[@]}"; do printf "%3d) %s\n" $((i+1)) "${AVAILABLE[$i]}"; done

read -p "Select file to smooth [1-${#AVAILABLE[@]}]: " idx
TMP="${AVAILABLE[$((idx-1))]}"

read -p "Temporal smoothing window (seconds, 0=skip): " TSM
read -p "Spatial σ blur (mm, 0=skip): " SSM

# Temporal smoothing
if [[ "${TSM:-0}" != "0" ]]; then
python3 - "$TMP" "$TSM" "$TR" <<'PY'
import sys, nibabel as nib, numpy as np
from scipy.ndimage import uniform_filter1d

f=sys.argv[1]
win_sec=float(sys.argv[2])
tr=float(sys.argv[3])

img=nib.load(f)
d=img.get_fdata()

if d.ndim==3:
    print("[WARN] 3D file → temporal smoothing skipped.")
    sys.exit(0)

win=max(1,int(round(win_sec/tr)))
sm=uniform_filter1d(d,size=win,axis=-1,mode="nearest")

out=f"temporal_smoothed_{win_sec}s_{f.rsplit('/',1)[-1]}"
nib.save(nib.Nifti1Image(sm.astype(np.float32),img.affine,img.header),out)
print("[OK] Temporal smoothing →",out)
PY
    if [[ -f temporal_smoothed_${TSM}s_${TMP##*/} ]]; then
        TMP="temporal_smoothed_${TSM}s_${TMP##*/}"
        save_global_png "$TMP" "$MASK" "$TR" "tsmooth"
    fi
fi

# Spatial smoothing
if [[ "${SSM:-0}" != "0" ]]; then
    OUT="spatial_smoothed_${SSM}mm_${TMP%.nii.gz}_$(timestamp).nii.gz"
    fslmaths "$TMP" -s "$SSM" "$OUT"
    TMP="$OUT"
    echo "[OK] Spatial smoothing → $OUT"
    save_global_png "$TMP" "$MASK" "$TR" "ssmooth"
fi

# =====================================================================
# STEP 2.5: DETRENDING
# =====================================================================
echo -e "${PURPLE}${BOLD}STEP 2.5: DETRENDING${RESET}"
center_line "Remove scanner drift (polynomial detrending)"

read -p "Do you want to detrend? (y/n): " DO_DET

if [[ "$DO_DET" == "y" ]]; then
    AVAILABLE=($(ls -tr cleaned_mc_func.nii.gz trimmed_*.nii.gz temporal_smoothed_*.nii.gz spatial_smoothed_*.nii.gz 2>/dev/null))
    for i in "${!AVAILABLE[@]}"; do printf "%3d) %s\n" $((i+1)) "${AVAILABLE[$i]}"; done

    read -p "Select file for detrending [1-${#AVAILABLE[@]}]: " idx
    TMP="${AVAILABLE[$((idx-1))]}"

    plot_global_signal "$TMP" "$MASK" "$TR"

    read -p "Apply detrending? (y/n): " DETR
    if [[ "$DETR" == "y" ]]; then
        while true; do
            echo "1) Linear"
            echo "2) Quadratic"
            echo "3) Custom polynomial"
            echo "4) Cancel"
            read -p "Choice [1–4]: " CHD

            case "$CHD" in
                1) POL=1 ;;
                2) POL=2 ;;
                3) read -p "Polynomial order: " POL ;;
                4) break ;;
                *) continue ;;
            esac

            BASE=$(basename "$TMP" .nii.gz)
            OUT="soner_pipeline/detrended/detrended_${BASE}_pol${POL}_$(timestamp).nii.gz"

            3dDetrend -polort $POL -prefix "$OUT" "$TMP"

            save_global_png "$OUT" "$MASK" "$TR" "detrended_pol${POL}"
            plot_global_signal "$OUT" "$MASK" "$TR"

            read -p "Use this detrended file? (y/n): " OK
            [[ "$OK" == "y" ]] && TMP="$OUT" && break
        done
    fi
fi

# =====================================================================
# STEP 3: SCM QC
# =====================================================================
echo -e "${PURPLE}${BOLD}STEP 3: SCM QC${RESET}"
center_line "Baseline / Signal mean + %change maps"

read -p "Run SCM QC? (y/n): " DO_SCM

if [[ "$DO_SCM" == "y" ]]; then
    AVAILABLE=($(ls -tr cleaned_mc_func.nii.gz trimmed_*.nii.gz spatial_smoothed_*.nii.gz temporal_smoothed_*.nii.gz 2>/dev/null))
    for i in "${!AVAILABLE[@]}"; do printf "%3d) %s\n" $((i+1)) "${AVAILABLE[$i]}"; done

    read -p "Select file [1-${#AVAILABLE[@]}]: " idx
    SCM_FILE="${AVAILABLE[$((idx-1))]}"

    plot_global_signal "$SCM_FILE" "$MASK" "$TR"

    read -p "Baseline vols (start end): " B1 B2
    read -p "Signal vols (start end): "   S1 S2

    OUT="soner_pipeline/scm_outputs/$(basename "$SCM_FILE" .nii.gz)_B${B1}-${B2}_S${S1}-${S2}_$(timestamp)"
    mkdir -p "$OUT"

    scm_blush_fast "$SCM_FILE" "$B1" "$B2" "$S1" "$S2" "$MASK" "$OUT"

    nohup fsleyes \
        "$OUT/norm_func_${B1}_${B2}.nii.gz" \
        "$OUT/baseline_${B1}_${B2}.nii.gz" \
        "$OUT/signal_change_map_${B1}_${B2}_${S1}_${S2}.nii.gz" >/dev/null 2>&1 &
fi

# =====================================================================
# ⭐ SAFE DEFAULTS FOR FILTERING — FIXES “UNBOUND VARIABLE”
# =====================================================================
HPF=""; LPF=""; BPL=""; BPH=""; NC=""; NW=""
# =====================================================================
#  STEP 4: FILTERING (HP / LP / BP / NOTCH)
# =====================================================================
echo -e "${PURPLE}${BOLD}STEP 4: FILTERING${RESET}"
center_line "High-pass / Low-pass / Band-pass / Notch filtering"

read -p "Do you want to apply a temporal filter? (y/n): " DO_FILT

if [[ "$DO_FILT" == "y" ]]; then

    AVAILABLE=($(ls -tr \
        cleaned_mc_func.nii.gz \
        trimmed_*.nii.gz \
        temporal_smoothed_*.nii.gz \
        spatial_smoothed_*.nii.gz \
        soner_pipeline/detrended/*.nii.gz \
        2>/dev/null))

    for i in "${!AVAILABLE[@]}"; do
        printf "%3d) %s\n" $((i+1)) "${AVAILABLE[$i]}"
    done

    read -p "Select file to filter [1-${#AVAILABLE[@]}]: " idx
    FILT_FILE="${AVAILABLE[$((idx-1))]}"

    echo -e "\nChoose filter type:"
    echo "1) High-pass"
    echo "2) Low-pass"
    echo "3) Band-pass"
    echo "4) Notch"
    echo "5) Cancel"
    read -p "Choice [1–5]: " FT

    # assign safe empty defaults (prevents unbound variable)
    HPF=""; LPF=""; BPL=""; BPH=""; NC=""; NW=""

    case "$FT" in

        1)
            read -p "High-pass cutoff (Hz): " HPF
        ;;

        2)
            read -p "Low-pass cutoff (Hz): " LPF
        ;;

        3)
            read -p "Band-pass LOW cutoff (Hz): " BPL
            read -p "Band-pass HIGH cutoff (Hz): " BPH
        ;;

        4)
            read -p "Notch center (Hz): " NC
            read -p "Notch width  (Hz): " NW
        ;;

        5)
            echo "[INFO] Filtering cancelled."
            FT=0
        ;;

        *)
            echo "[ERR] Invalid option"
            FT=0
        ;;
    esac

    # skip Python if FT was cancelled
    if [[ "$FT" != "0" ]]; then

python3 <<PY
import numpy as np, nibabel as nib, sys
from scipy.signal import butter, filtfilt
import datetime

fname = "$FILT_FILE"
ftype = "$FT"
tr = float("$TR")

img = nib.load(fname)
d = img.get_fdata().astype(np.float32)

if d.ndim == 3:
    print("[ERR] Cannot filter 3D data.")
    sys.exit(0)

X,Y,Z,T = d.shape
flat = d.reshape(-1, T)

nyq = 0.5 / tr

# ----------------------
# Filter selection
# ----------------------
if ftype == "1":
    HPF = float("$HPF")
    b,a = butter(4, HPF/nyq, "highpass")

elif ftype == "2":
    LPF = float("$LPF")
    b,a = butter(4, LPF/nyq, "lowpass")

elif ftype == "3":
    BPL = float("$BPL")
    BPH = float("$BPH")
    b,a = butter(4, [BPL/nyq, BPH/nyq], "bandpass")

elif ftype == "4":
    NC = float("$NC")
    NW = float("$NW")
    low  = (NC - NW/2) / nyq
    high = (NC + NW/2) / nyq
    b,a = butter(4, [low, high], "bandstop")
else:
    print("Filter cancelled.")
    sys.exit(0)

# ----------------------
# Apply filter
# ----------------------
good = np.isfinite(flat).all(axis=1) & (flat.std(axis=1) > 0)
out = flat.copy()

if good.any():
    out[good] = filtfilt(b, a, flat[good], axis=1, method="gust")

out = out.reshape(X, Y, Z, T)
out[np.isnan(out)] = d[np.isnan(out)]

base = fname.rsplit('/',1)[-1].replace(".nii.gz","")
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if ftype == "1":
    outname = f"soner_pipeline/filtering/hp_${HPF}Hz_{base}_{ts}.nii.gz"
elif ftype == "2":
    outname = f"soner_pipeline/filtering/lp_${LPF}Hz_{base}_{ts}.nii.gz"
elif ftype == "3":
    outname = f"soner_pipeline/filtering/bp_${BPL}-${BPH}Hz_{base}_{ts}.nii.gz"
else:
    outname = f"soner_pipeline/filtering/notch_${NC}Hz_{base}_{ts}.nii.gz"

nib.save(nib.Nifti1Image(out.astype(np.float32), img.affine, img.header), outname)
print("[OK] Saved:", outname)
PY

        FILT_OUT=$(ls -t soner_pipeline/filtering/*.nii.gz | head -n1)
        TMP="$FILT_OUT"

        save_global_png "$TMP" "$MASK" "$TR" "filter"
        plot_global_signal "$TMP" "$MASK" "$TR"

    fi
fi


# =====================================================================
#  STEP 5: DECOMPOSITION  (PCA / TICA / SICA / PLS)
# =====================================================================
while true; do
  echo -e "${PURPLE}${BOLD}STEP 5: DECOMPOSITION${RESET}"
  center_line "Decompose signal via PCA/ICA/PLS and drop unwanted components"

  echo "1) PCA"
  echo "2) Temporal ICA"
  echo "3) Spatial ICA"
  echo "4) PLS"
  echo "5) Exit"
  read -p "Choice [1–5]: " CH

  [[ "$CH" == "5" ]] && break

  case "$CH" in
    1) METHOD="pca" ;;
    2) METHOD="tica" ;;
    3) METHOD="sica" ;;
    4) METHOD="pls" ;;
    *) continue ;;
  esac

  AVAILABLE=($(ls -tr \
    cleaned_mc_func.nii.gz \
    trimmed_*.nii.gz \
    spatial_smoothed_*.nii.gz \
    temporal_smoothed_*.nii.gz \
    soner_pipeline/detrended/*.nii.gz \
    soner_pipeline/filtering/*.nii.gz \
    2>/dev/null))

  if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
    echo "[WARN] No candidate NIfTI files found for decomposition."
    break
  fi

  for i in "${!AVAILABLE[@]}"; do
    printf "%3d) %s\n" $((i+1)) "${AVAILABLE[$i]}"
  done

  read -p "Select file for ${METHOD^^} [1-${#AVAILABLE[@]}]: " idx
  TMP="${AVAILABLE[$((idx-1))]}"

  # PLS extra input
  if [[ "$METHOD" == "pls" ]]; then
    read -p "PLS regressor start/end vols (e.g. 1300 1500): " DS1 DS2
  else
    DS1=0; DS2=0
  fi

  # category (for saving inside pca_outputs/)
  SUB="raw"
  [[ "$TMP" == *"hp_"* ]]               && SUB="hp"
  [[ "$TMP" == *"lp_"* ]]               && SUB="lp"
  [[ "$TMP" == *"bp_"* ]]               && SUB="bp"
  [[ "$TMP" == *"detrended"* ]]         && SUB="detrended"
  [[ "$TMP" == trimmed_* ]]             && SUB="trim"
  [[ "$TMP" == temporal_smoothed_* ]]   && SUB="smoothed"
  [[ "$TMP" == spatial_smoothed_* ]]    && SUB="smoothed"

  save_global_png "$TMP" "$MASK" "$TR" "${METHOD}_before"

  # =====================================================================
  #  PYTHON DECOMPOSITION + GUI + TIMESTAMPED GRID PNGS
  # =====================================================================
python3 <<PY
import os, sys, nibabel as nib, numpy as np, matplotlib.pyplot as plt, datetime
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

fname = "$TMP"
method = "$METHOD"
sub = "$SUB"
reg_start = int("$DS1")
reg_end   = int("$DS2")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

img = nib.load(fname)
data = img.get_fdata().astype(np.float32)

if data.ndim != 4:
    print("[ERR] Decomposition requires 4D data.")
    sys.exit(0)

X, Y, Z, T = data.shape
flat = data.reshape(-1, T)

out_grid = f"soner_pipeline/qc/grid/{method}_grid_{ts}.png"
os.makedirs(f"soner_pipeline/pca_outputs/{sub}", exist_ok=True)

# =====================================================================
# Component GUI
# =====================================================================
def click_gui(comp, title, grid_path):
    n_show = min(comp.shape[1], 25)
    taxis = np.arange(comp.shape[0])
    drop = set()

    fig, axes = plt.subplots(int(np.ceil(n_show/5)), 5, figsize=(18,9))
    axes = axes.ravel()

    coord_text = fig.text(0.5, 0.01, "", ha="center", fontsize=10)
    markers = {}

    def refresh(ax, i):
        for sp in ax.spines.values():
            sp.set_edgecolor('red' if i in drop else 'black')
        ax.set_title(f"C{i+1}" + (" ✗" if i in drop else ""), fontsize=9)

    for i in range(n_show):
        ax = axes[i]
        tsig = (comp[:, i] - comp[:, i].mean()) / (comp[:, i].std() + 1e-6)
        ax.plot(taxis, tsig, lw=0.8)
        refresh(ax, i)

    plt.suptitle(title + " | click=drop | CTRL=marker | ENTER=save", fontsize=14)

    def on_click(event):
        if event.inaxes not in axes:
            return
        ax = event.inaxes
        idx = int(np.where(axes == ax)[0])

        # CTRL marker
        if event.key in ("control", "ctrl"):
            if event.xdata is None:
                return
            t = int(round(event.xdata))
            if not (0 <= t < comp.shape[0]):
                return
            val = comp[t, idx]
            print(f"[MARK] C{idx+1}: t={t}, val={val:.4f}")

            if idx in markers:
                for obj in markers[idx]:
                    obj.remove()

            vline = ax.axvline(t, color='red', lw=1)
            dot   = ax.plot(t, val, 'ro', markersize=4)[0]
            txt   = ax.text(0.02, 0.95, f"t={t}, v={val:.3f}",
                            transform=ax.transAxes, color='red',
                            va='top', fontsize=8)
            markers[idx] = (vline, dot, txt)
            fig.canvas.draw_idle()
            return

        # Toggle drop
        if idx in drop:
            drop.remove(idx)
        else:
            drop.add(idx)
        refresh(ax, idx)
        fig.canvas.draw_idle()

    def on_move(event):
        if event.inaxes not in axes:
            coord_text.set_text("")
            fig.canvas.draw_idle()
            return
        ax = event.inaxes
        idx = int(np.where(axes == ax)[0])
        if event.xdata is None:
            coord_text.set_text(f"C{idx+1}")
        else:
            t = int(round(event.xdata))
            if 0 <= t < comp.shape[0]:
                v = comp[t, idx]
                coord_text.set_text(f"C{idx+1}   t={t}   v={v:.3f}")
            else:
                coord_text.set_text(f"C{idx+1}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            plt.savefig(grid_path, dpi=160)
            print("[OK] Saved grid:", grid_path)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    return sorted(drop)

# =====================================================================
# Compute components
# =====================================================================
if method == "pca":
    ncomp = min(50, T)
    pca = PCA(n_components=ncomp)
    comps = pca.fit_transform(flat.T)
    drops = click_gui(comps, "PCA Components", out_grid)
    for d in drops:
        if 0 <= d < comps.shape[1]:
            comps[:, d] = 0
    recon = pca.inverse_transform(comps).T.reshape(X, Y, Z, T)

elif method == "tica":
    ncomp = min(25, T)
    ica = FastICA(n_components=ncomp, random_state=0, max_iter=500)
    S = ica.fit_transform(flat.T)
    drops = click_gui(S, "Temporal ICA Components", out_grid)
    for d in drops:
        if 0 <= d < S.shape[1]:
            S[:, d] = 0
    recon = (S @ ica.mixing_.T).T.reshape(X, Y, Z, T)

elif method == "sica":
    ncomp = min(25, T)
    flat2 = flat - flat.mean(axis=1, keepdims=True)
    Xn = StandardScaler(with_mean=False).fit_transform(flat2.T)
    ica = FastICA(n_components=ncomp, random_state=0, max_iter=500, whiten="unit-variance")
    S = ica.fit_transform(Xn)
    drops = click_gui(S, "Spatial ICA Components", out_grid)
    for d in drops:
        if 0 <= d < S.shape[1]:
            S[:, d] = 0
    recon = (S @ ica.mixing_.T).T.reshape(X, Y, Z, T)

elif method == "pls":
    Yv = np.zeros((T, 1), dtype=float)
    if 0 <= reg_start < T and 0 <= reg_end < T and reg_start <= reg_end:
        Yv[reg_start:reg_end+1, 0] = 1.0
    else:
        print(f"[WARN] PLS regressor window {reg_start}-{reg_end} invalid for T={T}; using all zeros.")
    ncomp = min(25, T)
    pls = PLSRegression(n_components=ncomp)
    pls.fit(flat.T, Yv)
    S = pls.x_scores_
    drops = click_gui(S, "PLS Components", out_grid)
    for d in drops:
        if 0 <= d < S.shape[1]:
            S[:, d] = 0
    recon = (S @ pls.x_loadings_.T).T.reshape(X, Y, Z, T)

# =====================================================================
# Save output
# =====================================================================
outf = (
    f"soner_pipeline/pca_outputs/{{sub}}/"
    f"{{sub}}_{{method}}_func_denoised_{{ts}}.nii.gz"
).format(sub=sub, method=method, ts=ts)

nib.save(nib.Nifti1Image(recon.astype(np.float32), img.affine, img.header), outf)
print("[OK] Saved →", outf)
PY

  # ---------------------------------
  # QC after decomposition
  # ---------------------------------
  DEN=$(ls -t soner_pipeline/pca_outputs/${SUB}/${SUB}_${METHOD}_func_denoised_*.nii.gz 2>/dev/null | head -n1 || true)
  if [[ -n "$DEN" ]]; then
    save_global_png "$DEN" "$MASK" "$TR" "${METHOD}_after"
    plot_global_signal "$DEN" "$MASK" "$TR"

    echo -e "\n${BLUE}${BOLD}Define baseline and signal windows for denoised output:${RESET}"
    read -p "Baseline vols (start end): " XB1 XB2
    read -p "Signal vols (start end): "   XS1 XS2

    OUTD="soner_pipeline/scm_outputs/${SUB}_${METHOD}_B${XB1}-${XB2}_S${XS1}-${XS2}_$(timestamp)"
    mkdir -p "$OUTD"

    scm_blush_fast "$DEN" "$XB1" "$XB2" "$XS1" "$XS2" "$MASK" "$OUTD"

    nohup fsleyes "$OUTD/norm_func_${XB1}_${XB2}.nii.gz" \
                  "$OUTD/baseline_${XB1}_${XB2}.nii.gz" \
                  "$OUTD/signal_change_map_${XB1}_${XB2}_${XS1}_${XS2}.nii.gz" >/dev/null 2>&1 &
  else
    echo "[WARN] No denoised file found to run SCM on."
  fi

  read -p "Run another decomposition? (y/n): " AGAIN
  [[ "$AGAIN" != "y" ]] && break
done

# =====================================================================
#  HTML SUMMARY (global + grid only, timestamped)
# =====================================================================
echo -e "${BLUE}${BOLD}Generating HTML QC summary…${RESET}"

python3 <<'PY'
import os, glob, datetime, html, platform, subprocess

ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

QC_GLOBAL = sorted(glob.glob("soner_pipeline/qc/global/*.png"))
QC_GRID   = sorted(glob.glob("soner_pipeline/qc/grid/*.png"))

def make_section(title, files):
    if not files:
        return f"<h2>{title}</h2><p>No images found.</p>"
    rows = "\n".join(
        f"<tr><td>{html.escape(os.path.basename(p))}</td>"
        f"<td><img src='{p}' width='360'></td></tr>"
        for p in files
    )
    return f"<h2>{title}</h2><table>{rows}</table>"

html_doc = f"""
<html>
<head>
<meta charset='utf-8'>
<title>BLuSH QC Summary</title>
<style>
body {{
    background:#0f1116;
    color:#eaeef2;
    font-family:Arial, sans-serif;
    padding:24px;
}}
h1 {{ color:#8bd3ff; }}
h2 {{ color:#62b0ff; margin-top:30px; }}
td, th {{
    border:1px solid #2a2f3a;
    padding:6px;
}}
table {{
    border-collapse:collapse;
    margin-bottom:24px;
}}
</style>
</head>
<body>

<h1>BLuSH QC Summary</h1>
<p><i>Generated: {ts}</i></p>

{make_section("Global Mean Plots", QC_GLOBAL)}
{make_section("Component Grid Plots", QC_GRID)}

</body>
</html>
"""

OUT = f"soner_pipeline/qc_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html_doc)

print("[OK] Summary HTML →", OUT)

try:
    if platform.system() == "Linux":
        subprocess.Popen(["xdg-open", OUT])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", OUT])
    elif platform.system() == "Windows":
        subprocess.Popen(["start", OUT], shell=True)
except Exception as e:
    print("[WARN] Could not auto-open HTML:", e)
PY

