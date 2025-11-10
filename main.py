from tg_loader import load_thermogravimetric_data
from tg_math import estimate_segment_rate, estimate_arrhenius_from_segments, arrhenius_plot_data
from tg_plotting import plot_ln_r_vs_time, plot_arrhenius

ODIN_PATH = "./TG_Data/Odin Data/"
SIF_PATH = "./TG_Data/Sif Data/"
df200 = load_thermogravimetric_data(f"{ODIN_PATH}200C, 20% O2, Isothermal 4H, Odin/ExpDat_BRF 500C.xlsx")
df225 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_BRF.csv")
df250 = load_thermogravimetric_data(f"{SIF_PATH}TG TEST 3 - ISOTHERMAL 250C 20% O2/ExpDat_BRF 500.csv")


seg200 = estimate_segment_rate(df200, time_window=(63, 290), label="200C")
seg225 = estimate_segment_rate(df225, time_window=(88, 290), label="225C")
seg250 = estimate_segment_rate(df250, time_window=(90, 290), label="250C")

plot_ln_r_vs_time(df200, time_window=(63, 290), label="200C", overlay_constant_r=seg200.r_abs, save_path="lnr_vs_t_200C.png")

res = estimate_arrhenius_from_segments([seg200, seg225, seg250])
x, y = arrhenius_plot_data([seg200, seg225, seg250])

plot_arrhenius(x, y, slope=res.slope, intercept=res.intercept, save_path="arrhenius.png")

print("Ea [kJ/mol]:", res.E_A_J_per_mol/1000)
print("A [1/min]:", res.A)
