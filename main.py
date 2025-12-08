import numpy as np
from matplotlib import pyplot as plt

from tg_loader import load_thermogravimetric_data
from tg_math import estimate_segment_rate_zero_order, estimate_segment_rate_first_order, \
    estimate_arrhenius_from_segments, arrhenius_plot_data, estimate_EA_A_nonisothermal_coats_redfern, \
    estimate_EA_A_nonisothermal_coats_redfern_global, estimate_global_coats_redfern_with_o2, simulate_alpha_ramp, \
    alpha_to_mass_pct
from tg_plotting import plot_ln_r_vs_time, plot_arrhenius, plot_arrhenius_groups, plot_coats_redfern_overlays, \
    plot_coats_redfern_global, plot_global_coats_redfern_o2_fit

ODIN_PATH = "./TG_Data/Odin Data/"
SIF_PATH = "./TG_Data/Sif Data/"

#########
# DATA LOADING todo: find better data loading scheme
#########

# BRF
bfr200 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_BRF500.xlsx")
bfr225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_BRF500.xlsx") #useless
bfr250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_BRF500.xlsx")

# WS
ws200 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_WS500.xlsx")
ws225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_WS500.xlsx")
ws250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_WS500.xlsx")

# PW
pw200 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_PW500.xlsx")
pw225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_PW500.xlsx")
pw250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_PW500.xlsx")

#####
# Parse time segments todo: better scheme
#####

# BRF
seg_bfr_200 = estimate_EA_A_nonisothermal_coats_redfern(bfr200, time_window=(32, 195.5), label="BRF 10%")
#seg_bfr_225 = estimate_EA_A_nonisothermal_coats_redfern(bfr225, time_window=(88, 290), label="BRF 5%")
seg_bfr_250 = estimate_EA_A_nonisothermal_coats_redfern(bfr250, time_window=(32, 195), label="BRF 20%")

# WS
seg_ws_200 = estimate_EA_A_nonisothermal_coats_redfern(ws200, time_window=(32, 195), label="WS 10%")
seg_ws_225 = estimate_EA_A_nonisothermal_coats_redfern(ws225, time_window=(32, 195), label="WS 5%")
seg_ws_250 = estimate_EA_A_nonisothermal_coats_redfern(ws250, time_window=(32, 195), label="WS 20%")

# PW
seg_pw_200 = estimate_EA_A_nonisothermal_coats_redfern(pw200, time_window=(32, 195), label="PW 10%")
seg_pw_225 = estimate_EA_A_nonisothermal_coats_redfern(pw225, time_window=(32, 195), label="PW 5%")
seg_pw_250 = estimate_EA_A_nonisothermal_coats_redfern(pw250, time_window=(32, 195), label="PW 20%")

res = estimate_global_coats_redfern_with_o2(
    [pw225, pw200, pw250],
    o2_fractions=[0.05, 0.10, 0.20],
    time_window=(32.0, 195.0),      # ramp region
    n_solid=1.0,                   # 1st order in solid assumption
    alpha_range=(0.20, 0.80),
    beta_fixed_K_per_time=3.0,     # 3 K/min (since time_min)
    label="PW ramps global O2 fit",
)

print("Ea [kJ/mol]:", res.E_A_J_per_mol/1000)
print("A  [1/min]: ", res.A)
print("m_O2 order:", res.m_o2)
print("R²:", res.r2)

plot_global_coats_redfern_o2_fit(res, save_path="pw_global_cr", title="PW global CR fit")

df = pw200
t0 = 32.0
t1 = 195.0 # same time window
seg = df[(df["time_min"] >= t0) & (df["time_min"] <= t1)].copy()

t = seg["time_min"].to_numpy(dtype=float)
T = seg["temp_C"].to_numpy(dtype=float)
m_exp = seg["mass_pct"].to_numpy(dtype=float)

# estimate m0 and m_inf in THIS window (same as you do elsewhere)
N = len(m_exp)
k_head = max(3, int(round(0.10 * N)))
k_tail = max(3, int(round(0.20 * N)))
m0 = float(np.nanmedian(m_exp[:k_head]))
m_inf = float(np.nanmedian(m_exp[-k_tail:]))

# simulate alpha(t) using your global fitted parameters (res = global fit result)
alpha_sim = simulate_alpha_ramp(
    time_min=t,
    temp_C=T,
    yO2=0.1,
    E_A_J_per_mol=res.E_A_J_per_mol,
    A=res.A,
    m_o2=res.m_o2,
    solid_order=1,          # match n_solid you assumed
    alpha0=0.0,
)

# convert simulated alpha -> mass%
m_sim = alpha_to_mass_pct(alpha_sim, m0=m0, m_inf=m_inf, loss=True)

# overlay plot
plt.figure()
plt.plot(t, m_exp, "o", label="exp mass%")
plt.plot(t, m_sim, "-", label="sim mass%")
plt.xlabel("time [min]")
plt.ylabel("mass [%]")
plt.legend()
plt.tight_layout()
plt.savefig("pw_mass_overlay.png", dpi=150)
plt.close()