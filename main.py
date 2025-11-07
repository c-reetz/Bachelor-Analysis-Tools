from tg_loader import load_thermogravimetric_data
from tg_math import fit_isothermal_k, fit_arrhenius_from_isotherms_corrected

ODIN_PATH = "./TG_Data/Odin Data/"
SIF_PATH = "./TG_Data/Sif Data/"
df200 = load_thermogravimetric_data("")
df225 = load_thermogravimetric_data("./TG_Data/Sif Data/TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_BRF.csv")
df250 = load_thermogravimetric_data("Dataset_250C.csv")

iso = [
    fit_isothermal_k(df200, time_window=(t0_200, t1_200), n=1.0),
    fit_isothermal_k(df225, time_window=(t0_225, t1_225), n=1.0),
    fit_isothermal_k(df250, time_window=(t0_250, t1_250), n=1.0),
]

# O2 correction: n=1, 20% for all
res = fit_arrhenius_from_isotherms_corrected(
    iso, gas_order=1.0, o2_fractions=[0.20, 0.20, 0.20]
)

print("Ea [kJ/mol]:", res.E_J_per_mol/1000)
print("A (intrinsic) [1/time]:", res.A)
# If you want the apparent A' (including O2^n), check res.A_prime
