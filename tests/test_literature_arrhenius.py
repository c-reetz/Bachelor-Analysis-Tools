# tests/test_literature_arrhenius.py

import math
from pathlib import Path
import sys

# --- ensure project root is importable (so `tg_math` works) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from tg_math import SegmentRate, estimate_arrhenius_from_segments, R_DEFAULT


def _vitazek_segments() -> list[SegmentRate]:
    """
    Build synthetic SegmentRate objects based on Vitázek et al. (2021),
    "Isothermal Kinetic Analysis of the Thermal Decomposition of
    Wood Chips from an Apple Tree", Processes 9, 195.

    Table 3 (F1 model, 0.10 <= alpha <= 0.85) gives:

        T (°C):  250    270    290
        ln(k):  -1.87  -1.54  -1.32

    Units: k in min^-1, T in °C. We convert to K and compute k=exp(ln k),
    then feed those as r_abs to your Arrhenius fitter.
    """
    T_C = np.array([250.0, 270.0, 290.0], dtype=float)
    ln_k = np.array([-1.87, -1.54, -1.32], dtype=float)

    T_K = T_C + 273.15
    k = np.exp(ln_k)  # k [1/min]

    segments: list[SegmentRate] = []
    for T, ki, Tc_val in zip(T_K, k, T_C):
        seg = SegmentRate(
            label=f"{int(Tc_val)}C_isothermal",
            T_mean_K=float(T),
            T_span_K=0.0,
            r_abs=float(ki),          # treat k as "rate" for Arrhenius
            slope_signed=float(-ki),  # sign is irrelevant for the fit
            intercept=0.0,
            r2_mass_vs_time=1.0,
            n_points=100,
            time_window=(0.0, 10.0),
        )
        segments.append(seg)

    return segments


def test_arrhenius_from_vitazek_apple_tree():
    """
    Check that estimate_arrhenius_from_segments reproduces the Arrhenius
    parameters reported by Vitázek et al. (2021).

    Lit. values (Arrhenius fit of ln k vs 1/T):

        Ea = 34 ± 3 kJ/mol
        A  = 391 ± 2 min^-1

    We allow ~10–15% relative tolerance to account for rounding of ln(k).
    """
    segments = _vitazek_segments()

    res = estimate_arrhenius_from_segments(
        segments,
        R=R_DEFAULT
    )

    Ea_kJ = res.E_A_J_per_mol / 1000.0

    # Basic sanity
    assert math.isfinite(Ea_kJ)
    assert math.isfinite(res.A)
    assert res.A > 0
    assert res.r2_ln_r_vs_invT > 0.99

    # Compare to literature
    assert Ea_kJ == pytest.approx(34.0, rel=0.10)   # 34 ± ~3.4 kJ/mol
    assert res.A == pytest.approx(391.0, rel=0.15)  # 391 ± ~58 min^-1
