# bach tools
Utilised to analyse TG data for bachelors. Structure:
```
- TG_DATA/         # directory containing TG data files, from different TGs, only "Sif" data used.
    - Sif Data/
    - Odin Data/  # Should stay unused due to difference in Odin and Sif data
    - Samlede Data/ # Combined data from multiple TGs and multiple runs
        - AiO.xlsx       # "All in one" combined data-file for fast lookups of TG graphs and comparing various TGs.
- test/
    - test_literature_arrhenius.py   # test to see if data from an academic source would result in same results using functions in tg_math
- CHNS/        # CHNS analysis data, unused after making `Characterization_Proximate_Ultimate_CHEC_EA` which was another CHNS analysis machine
- BET Data + Reports/ # Data from Anton Nova Paar800 BET analyser
- tg_math.py      # module containing mathematical functions for TG data analysis
- tg_loader.py    # module containing functions to load TG data from excel files
- tg_plotting.py  # module containing functions to plot, save & show TG data
- main.py      # main script to run analyses (Should be cleaned up...)
```

## Used math functions:
### Non-isothermal TG Data:
- `estimate_global_coats_redfern_with_o2`: Uses the model 
  - `y = ln(g(w)/T^2) = ln(A*R/(beta*Ea)) + m*ln(yO2) - Ea/R * (1/T)`
  - Estimates activation energy and pre-exponential factor using Coats-Redfern method with oxygen data, needs multiple oxygen levels to estimate the oxygen order. Fits a line to a corrected oxygen plot done by doing a vertical shift by: `y*=y-m*ln(y(O2))`. Can obviously be used for other gas-types as long as composition is known. easy to expand.
- `simulate_alpha_ramp`: Simulate conversion alpha(t) for a non-isothermal ramp using:
  - `dα/dt = k(T,yO2) * (1-α)^(solid_order)` 
  - `k = A * yO2^m_o2 * exp(-Ea/(R*T))`
- `alpha_to_mass_pct`: Convert conversion alpha to mass percentage remaining using:
  - If mass loss:
    - `alpha=(m0-m)/(m0-m_inf) -> m = m0 - alpha*(m0-m_inf)`
  - If mass gain:
    -  `alpha=(m-m0)/(m_inf-m0) -> m = m0 + alpha*(m_inf-m0)`

### Isothermal TG Data:


## Plot saving
Plots save under the folder `Plots/` when using tg_plotting functions.