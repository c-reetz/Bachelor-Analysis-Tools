import math
import numpy as np
from pprint import pprint

def print_global_cr_o2_result(res):
    """
    Pretty-print a tg_math.GlobalCR_O2_Result without dumping huge arrays.
    """

    # Scalars / metadata
    summary = {
        "label": res.label,
        "n_solid": res.n_solid,                 # your model order (you set this to 1)
        "alpha_range_used": res.alpha_range,    # conversion window actually used (may be clipped)
        "time_window_used": res.time_window,
        "beta_K_per_time": res.beta_K_per_time, # heating rate in K / (time unit)

        # Main fit results
        "E_A_kJ_per_mol": res.E_A_J_per_mol / 1000.0,
        "A": res.A,
        "lnA": (math.log(res.A) if res.A > 0 else float("nan")),
        "m_o2": res.m_o2,
        "r2": res.r2,

        # Regression internals (useful for debugging / appendix)
        "intercept": res.intercept,
        "coef_lnO2": res.coef_lnO2,
        "coef_invT": res.coef_invT,

        # Point counts
        "N_pts_total": res.n_points,                 # total points used across all datasets
        "N_pts_per_dataset": res.dataset_point_counts,
    }

    pprint(summary)

    # Array summaries (so you can verify content without printing all values)
    def _arr_stats(name, a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            print(f"{name}: EMPTY")
            return
        print(
            f"{name}: shape={a.shape}, "
            f"min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}, "
            f"finite={np.isfinite(a).sum()}/{a.size}"
        )

  #  _arr_stats("x_invT_all (1/T)", res.x_invT_all)
  #  _arr_stats("z_lnO2_all (ln O2)", res.z_lnO2_all)
  #  _arr_stats("y_all (ln(g/T^2))", res.y_all)
