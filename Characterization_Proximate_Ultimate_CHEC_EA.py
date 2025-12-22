import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Bookkeeping (original labels)
feedstocks = ['BRF', 'WS', 'PW']
temps = [350, 500, 700]
labels9 = [f"{f}_{t}" for f in feedstocks for t in temps]  # brf_350,...,pw_700 (9)
labels_feed = ['BRF_feed', 'WS_feed', 'PW_feed']
labels_all_original = labels9 + labels_feed  # current order: brf350,...,pw700,BRF_feed,WS_feed,PW_feed

# -----------------------------
# RAW TRIPLICATE DATA (CHN EA) for the original 9 groups (3x9)
C_reps = np.array([
    [47.286, 52.756, 55.220, 67.735, 73.329, 71.905, 76.239, 86.489, 85.179],  # rep 1
    [47.645, 47.409, 47.413, 67.431, 70.849, 74.383, 69.715, 89.259, 93.540],  # rep 2
    [39.115, 51.387, 56.207, 66.934, 69.489, 71.497, 78.989, 86.489, 95.456]   # rep 3
], dtype=float)

H_reps = np.array([
    [2.372, 1.373, 0.477, 3.268, 2.151, 0.612, 3.831, 2.674, 0.620],  # rep 1
    [2.423, 1.252, 0.390, 3.316, 1.965, 0.696, 3.580, 2.719, 0.927],  # rep 2
    [1.924, 1.485, 0.529, 3.247, 1.926, 0.696, 3.707, 2.657, 0.928]   # rep 3
], dtype=float)

N_reps = np.array([
    [1.593, 1.529, 1.308, 0.676, 0.676, 0.317, 0.210, 0.132, 0.000],  # rep 1
    [1.652, 1.383, 0.764, 0.671, 0.486, 0.465, 0.137, 0.162, 0.100],  # rep 2
    [1.251, 1.498, 1.116, 0.687, 0.198, 0.566, 0.187, 0.125, 0.000]   # rep 3
], dtype=float)

# -----------------------------
# PROXIMATE triplicates for the original 9 groups (3x9)
Ash_reps = np.array([
    [34.51, 39.94, 41.72, 13.02, 17.87, 17.21, 1.13, 1.46, 1.54],
    [35.42, 40.64, 42.21, 14.99, 16.92, 18.42, 1.14, 1.45, 1.52],
    [34.70, 39.67, 41.86, 14.09, 17.06, 19.06, 1.15, 1.48, 1.56]
], dtype=float)

Moisture_reps = np.array([
    [3.15, 3.98, 4.88, 6.69, 5.02, 6.94, 1.55, 2.28, 2.52],
    [3.14, 3.90, 4.86, 2.95, 4.45, 6.77, 1.35, 2.15, 2.29],
    [3.09, 3.82, 4.52, 2.82, 4.25, 6.66, 2.89, 2.03, 2.06]
], dtype=float)

Volatile_reps = np.array([
    [21.04, 15.09, 8.44, 26.97, 14.15, 9.75, 29.08, 12.82, 4.90],
    [18.01, 14.84, 8.39, 27.09, 14.39, 9.83, 29.02, 12.78, 4.65],
    [17.00, 14.96, 8.42, 27.03, 14.26750227, 9.79, 29.05057249, 12.80055318, 4.77]
], dtype=float)

# -----------------------------
# NEW FEEDSTOCK DATA (triplicates) — values from your table (DB)
brf_moist = [14.16, 13.25, 13.31]
brf_vol   = [62.42, 62.04, 62.15]
brf_ash   = [18.64, 21.26, 19.95]

ws_moist = [8.25, 8.77, 8.59]
ws_vol   = [76.78, 77.34, 77.47]
ws_ash   = [5.89, 5.00, 5.29]

pw_moist = [10.54, 10.49, 14.01]
pw_vol   = [85.84, 84.95, 85.54]
pw_ash   = [0.23, 0.37, 0.36]

# Stack new proximate columns as (3 x 3) arrays where columns = feedstocks in order BRF, WS, PW
Ash_reps_feed = np.array([brf_ash, ws_ash, pw_ash], dtype=float).T     # shape (3,3)
Moisture_reps_feed = np.array([brf_moist, ws_moist, pw_moist], dtype=float).T
Volatile_reps_feed  = np.array([brf_vol, ws_vol, pw_vol], dtype=float).T

# -----------------------------
# NEW FEEDSTOCK ULTIMATE (Dry basis) values — treated AS DB
C_brf_db = [42.64076233, 42.42800903, 41.48335266]
H_brf_db = [5.040123463, 5.0705266, 5.845505714]
N_brf_db = [1.361185193, 1.343269706, 1.341135208]

C_ws_db = [44.26855469, 44.51888657, 44.68572998]
H_ws_db = [5.708453178, 5.672017574, 5.64115572]
N_ws_db = [0.470543295, 0.3844181, 0.405320346]

C_pw_db = [48.73808289, 50.43639755, 48.18469238]
H_pw_db = [6.066213131, 6.386624813, 6.002144814]
N_pw_db = [0.117241606, 0.12249478, 0.117871963]

# Build 3x3 arrays (rows=reps, cols=feedstocks in order BRF,WS,PW)
C_reps_feed = np.array([C_brf_db, C_ws_db, C_pw_db], dtype=float).T
H_reps_feed = np.array([H_brf_db, H_ws_db, H_pw_db], dtype=float).T
N_reps_feed = np.array([N_brf_db, N_ws_db, N_pw_db], dtype=float).T

# -----------------------------
# Append feedstock columns to the original arrays (hstack)
C_reps = np.hstack([C_reps, C_reps_feed])     # shape (3,12)
H_reps = np.hstack([H_reps, H_reps_feed])
N_reps = np.hstack([N_reps, N_reps_feed])

Ash_reps = np.hstack([Ash_reps, Ash_reps_feed])
Moisture_reps = np.hstack([Moisture_reps, Moisture_reps_feed])
Volatile_reps = np.hstack([Volatile_reps, Volatile_reps_feed])

# -----------------------------
# REORDER COLUMNS so that feedstocks come first, then the biochars in each feedstock group:
# desired order: BRF_feed, brf350, brf500, brf700, WS_feed, ws350, ws500, ws700, PW_feed, pw350, pw500, pw700
# Current column indices (before reorder): 
# 0 brf350,1 brf500,2 brf700,3 ws350,4 ws500,5 ws700,6 pw350,7 pw500,8 pw700,9 BRF_feed,10 WS_feed,11 PW_feed
reorder_idx = [9, 0, 1, 2, 10, 3, 4, 5, 11, 6, 7, 8]

# Apply reorder to all reps arrays and proximate arrays
C_reps = C_reps[:, reorder_idx]
H_reps = H_reps[:, reorder_idx]
N_reps = N_reps[:, reorder_idx]

Ash_reps = Ash_reps[:, reorder_idx]
Moisture_reps = Moisture_reps[:, reorder_idx]
Volatile_reps = Volatile_reps[:, reorder_idx]

# New labels in the requested order
labels_ordered = ['BRF'] + [f'brf_{t}' for t in temps] + ['WS'] + [f'ws_{t}' for t in temps] + ['PW'] + [f'pw_{t}' for t in temps]

# -----------------------------
# STATS HELPER
ddof = 1
def mean_sd(arr):
    mean = np.nanmean(arr, axis=0)
    sd   = np.nanstd(arr, axis=0, ddof=ddof)
    return mean, sd

# -----------------------------
# 1. CALCULATE PROXIMATE (Dry Basis)
Moist_mean, Moist_sd = mean_sd(Moisture_reps)
Vol_mean, Vol_sd = mean_sd(Volatile_reps)
Ash_mean, Ash_sd = mean_sd(Ash_reps)

# FC by difference of means (No SD)
FC_mean = 100.0 - (Vol_mean + Ash_mean)

# -----------------------------
# 2. CALCULATE ULTIMATE (Dry Basis)
C_mean, C_sd = mean_sd(C_reps)
H_mean, H_sd = mean_sd(H_reps)
N_mean, N_sd = mean_sd(N_reps)

# Oxygen by difference of means (No SD)
O_mean = 100.0 - (C_mean + H_mean + N_mean + Ash_mean)

# -----------------------------
# 3. CALCULATE ULTIMATE (Dry Ash-Free Basis)
daf_factor = 100.0 / (100.0 - Ash_reps)   # shape (3,12)
C_reps_daf = C_reps * daf_factor
H_reps_daf = H_reps * daf_factor
N_reps_daf = N_reps * daf_factor

C_mean_daf, C_sd_daf = mean_sd(C_reps_daf)
H_mean_daf, H_sd_daf = mean_sd(H_reps_daf)
N_mean_daf, N_sd_daf = mean_sd(N_reps_daf)

# Oxygen DAF by difference of means (No SD)
O_mean_daf = 100.0 - (C_mean_daf + H_mean_daf + N_mean_daf)

# -----------------------------
# 4. CALCULATE MOLAR RATIOS (From Means, No SD)
AW_C = 12.011
AW_H = 1.008
AW_O = 15.999

moles_H_mean = H_mean / AW_H
moles_C_mean = C_mean / AW_C
moles_O_mean = O_mean / AW_O

HC_ratio = moles_H_mean / moles_C_mean
OC_ratio = moles_O_mean / moles_C_mean

# -----------------------------
# 5. PREPARE TABLES (format helpers)
def fmt_sd(m, s):
    return f"{m:.3f} ± {s:.3f}"

def fmt_no_sd(m):
    return f"{m:.3f}"

# Proximate Table (ordered)
prox_data = []
for i, lab in enumerate(labels_ordered):
    prox_data.append({
        "Condition": lab,
        "Moisture (%)": fmt_sd(Moist_mean[i], Moist_sd[i]),
        "Volatile Matter (%)": fmt_sd(Vol_mean[i], Vol_sd[i]),
        "Fixed Carbon (%)": fmt_no_sd(FC_mean[i]), # No SD
        "Ash (%)": fmt_sd(Ash_mean[i], Ash_sd[i])
    })
df_prox = pd.DataFrame(prox_data)

# Ultimate Table (Dry) (ordered)
ult_data = []
for i, lab in enumerate(labels_ordered):
    ult_data.append({
        "Condition": lab,
        "C (%)": fmt_sd(C_mean[i], C_sd[i]),
        "H (%)": fmt_sd(H_mean[i], H_sd[i]),
        "N (%)": fmt_sd(N_mean[i], N_sd[i]),
        "O (%)": fmt_no_sd(O_mean[i]), # No SD
        "Ash (%)": fmt_sd(Ash_mean[i], Ash_sd[i])
    })
df_ult = pd.DataFrame(ult_data)

# Ultimate Table (DAF) with RATIOS (ordered)
ult_daf_data = []
for i, lab in enumerate(labels_ordered):
    ult_daf_data.append({
        "Condition": lab,
        "C (DAF %)": fmt_sd(C_mean_daf[i], C_sd_daf[i]),
        "H (DAF %)": fmt_sd(H_mean_daf[i], H_sd_daf[i]),
        "N (DAF %)": fmt_sd(N_mean_daf[i], N_sd_daf[i]),
        "O (DAF %)": fmt_no_sd(O_mean_daf[i]), # No SD
        "H/C (molar)": fmt_no_sd(HC_ratio[i]), # No SD
        "O/C (molar)": fmt_no_sd(OC_ratio[i])  # No SD
    })
df_ult_daf = pd.DataFrame(ult_daf_data)

# -----------------------------
# 6. PLOTTING (helpers)
def display_table(df, title):
    fig, ax = plt.subplots(figsize=(14, len(df)*0.5 + 1.5))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')
    plt.tight_layout()
    plt.show()

def plot_stacked_bar(
    data_dict,
    labels,
    title,
    ylabel="Weight %",
    show_values=True,      # <- set True for Ultimate plots
    decimals=2,             # <- change to 0 if you want whole numbers
    min_height=2          # <- don't label tiny segments (set 0 to label all)
):
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.7
    bottom = np.zeros(len(labels), dtype=float)
    colors = plt.cm.tab10.colors

    for i, (comp_name, values) in enumerate(data_dict.items()):
        values_arr = np.array(values, dtype=float)

        bars = ax.bar(
            labels, values_arr, width,
            label=comp_name,
            bottom=bottom,
            color=colors[i % len(colors)]
        )

        if show_values:
            for bar, val in zip(bars, values_arr):
                if np.isnan(val):
                    continue

                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + bar.get_height() / 2

                # Auto text color (white on dark bars, black on light bars)
                r, g, b, _ = bar.get_facecolor()
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = "black" if luminance > 0.6 else "white"
                if abs(val) < min_height:
                    ax.text(
                        x, y,
                        f"{val:.{decimals}f}%",
                        ha="center", va="center",
                        fontsize=5, fontweight="bold",
                        color=txt_color
                    )
                else:
                    ax.text(
                        x, y,
                        f"{val:.{decimals}f}%",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=txt_color
                    )

        bottom += values_arr

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=45, ha='right')

    # Give a little headroom above the tallest stack
    ax.set_ylim(0, np.max(bottom) * 1.05)

    plt.tight_layout()
    plt.show()


# Plot Data (ordered)
prox_plot_data = {
    'Fixed Carbon': FC_mean,
    'Volatile Matter': Vol_mean,
    'Ash': Ash_mean
}

ult_dry_plot_data = {
    'C': C_mean,
    'H': H_mean,
    'N': N_mean,
    'O': O_mean,
    'Ash': Ash_mean
}

ult_daf_plot_data = {
    'C (DAF)': C_mean_daf,
    'H (DAF)': H_mean_daf,
    'N (DAF)': N_mean_daf,
    'O (DAF)': O_mean_daf
}

# -----------------------------
# 7. BET SURFACE AREA DATA (NEW ADDITION)
bet_data = {
    'Sample': ['BRF_350', 'BRF_500', 'BRF_700', 'WS_350', 'WS_500', 'WS_700', 'PW_350', 'PW_500', 'PW_700'],
    'BET_SSA (m²/g)': [1.067, 4.224, 2.38, 1.956, 2.665, 5.067, 1.119, 202.525, 356.22]
}

df_bet = pd.DataFrame(bet_data)

def display_bet_results():
    import numpy as np
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("BET SURFACE AREA ANALYSIS")
    print("=" * 60)

    # -------------------------
    # Table view
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(
        "BET Specific Surface Area (SSA) of Biochars",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    table_data = []
    for _, row in df_bet.iterrows():
        table_data.append([row["Sample"], f"{float(row['BET_SSA (m²/g)']):.3f}"])

    table = ax.table(
        cellText=table_data,
        colLabels=["Sample", "BET SSA (m²/g)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6e6e6")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Helper: label bars on log axis
    # -------------------------
    def _label_bars_log(ax, bars, values, fmt="{:.1f}", mult=1.08):
        """
        Add value labels above bars when y-axis is log-scaled.
        Uses multiplicative offset (height * mult) which behaves well on log scale.
        """
        for bar, v in zip(bars, values):
            if v is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            if not np.isfinite(vv) or vv <= 0:
                continue

            x = bar.get_x() + bar.get_width() / 2.0
            y = vv * mult
            ax.text(x, y, fmt.format(vv), ha="center", va="bottom", fontsize=9)

    # -------------------------
    # Bar plots (log scale)
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Plot 1: All samples (log y) ----
    samples_all = df_bet["Sample"].astype(str)
    ssa_all = pd.to_numeric(df_bet["BET_SSA (m²/g)"], errors="coerce")

    # log-scale requires positive values
    mask_pos = np.isfinite(ssa_all.to_numpy(dtype=float)) & (ssa_all.to_numpy(dtype=float) > 0)
    samples = samples_all[mask_pos].reset_index(drop=True)
    ssa_values = ssa_all[mask_pos].reset_index(drop=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(samples)))
    bars1 = ax1.bar(samples, ssa_values, color=colors, edgecolor="black")

    ax1.set_xlabel("Sample", fontsize=12)
    ax1.set_ylabel("BET SSA (m²/g)", fontsize=12)
    ax1.set_title("BET Surface Area - All Samples (log scale)", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_yscale("log")

    # labels on bars (log-friendly)
    _label_bars_log(ax1, bars1, ssa_values.to_numpy(dtype=float), fmt="{:.1f}", mult=1.08)

    # ---- Plot 2: Grouped by feedstock & temperature (log y) ----
    feedstocks = ["BRF", "WS", "PW"]
    colors_feed = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    temps = [350, 500, 700]

    for i, feedstock in enumerate(feedstocks):
        feedstock_data = df_bet[df_bet["Sample"].astype(str).str.contains(feedstock, na=False)]

        ssa_vals = []
        for temp in temps:
            sample_name = f"{feedstock}_{temp}"
            if sample_name in feedstock_data["Sample"].values:
                val = feedstock_data.loc[
                    feedstock_data["Sample"] == sample_name, "BET_SSA (m²/g)"
                ].values[0]
                val = float(val) if np.isfinite(val) else np.nan
                # log-scale requires >0
                if not np.isfinite(val) or val <= 0:
                    val = np.nan
                ssa_vals.append(val)
            else:
                ssa_vals.append(np.nan)  # missing values cannot be plotted on log axis

        x_pos = np.arange(len(temps)) + i * 0.25

        # Only plot bars where we have finite positive values
        ssa_arr = np.array(ssa_vals, dtype=float)
        mask = np.isfinite(ssa_arr) & (ssa_arr > 0)

        bars2 = ax2.bar(
            x_pos[mask],
            ssa_arr[mask],
            width=0.25,
            label=feedstock,
            color=colors_feed[i],
            edgecolor="black",
        )
        _label_bars_log(ax2, bars2, ssa_arr[mask], fmt="{:.1f}", mult=1.08)

    ax2.set_xlabel("Pyrolysis Temperature (°C)", fontsize=12)
    ax2.set_ylabel("BET SSA (m²/g)", fontsize=12)
    ax2.set_title("BET Surface Area by Feedstock & Temperature (log scale)", fontsize=14, fontweight="bold")
    ax2.set_xticks(np.arange(len(temps)) + 0.25)
    ax2.set_xticklabels(temps)
    ax2.legend(title="Feedstock")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Summary stats
    # -------------------------
    print("\nBET SSA Analysis by Feedstock:")
    print("-" * 40)
    for feedstock in feedstocks:
        feedstock_ssa = pd.to_numeric(
            df_bet[df_bet["Sample"].astype(str).str.contains(feedstock, na=False)]["BET_SSA (m²/g)"],
            errors="coerce",
        ).dropna()
        feedstock_ssa = feedstock_ssa[feedstock_ssa > 0]

        print(f"\n{feedstock}:")
        if feedstock_ssa.empty:
            print("  No positive BET SSA values available.")
            continue

        print(f"  Range: {feedstock_ssa.min():.3f} - {feedstock_ssa.max():.3f} m²/g")
        print(f"  Mean: {feedstock_ssa.mean():.3f} m²/g")
        if feedstock == "PW":
            print("  Note: Exceptional increase at higher temperatures")


# -----------------------------
# EXECUTE OUTPUTS (ordered labels)
print("Generating Tables...")
display_table(df_prox, "Proximate Analysis (Dry Basis) - FC by difference of means (no SD for FC)")
display_table(df_ult, "Ultimate Analysis (Dry Basis) - O by difference of means (no SD for O)")
display_table(df_ult_daf, "Ultimate Analysis (DAF Basis) & Atomic Ratios")

print("Generating Stacked Bar Plots...")
plot_stacked_bar(prox_plot_data, labels_ordered, "Proximate Analysis (Dry Basis)")
plot_stacked_bar(ult_dry_plot_data, labels_ordered, "Ultimate Analysis (Dry Basis)")
plot_stacked_bar(ult_daf_plot_data, labels_ordered, "Ultimate Analysis (DAF Basis)")

print("\n" + "="*60)
print("ADDITIONAL ANALYSIS: BET SURFACE AREA RESULTS")
print("="*60)
display_bet_results()