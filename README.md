# TUBAF Soil Respiration Chamber Analysis

Python package for GHG flux calculation from closed static respiration chamber data.
Ports the core model-selection logic of the R [goFlux](https://github.com/Qepanna/goFlux) package to pure Python.

## Quick start

```python
from chamber_analysis import load_directory, ChamberParams
from chamber_analysis.pyflux import analyze_directory

params = ChamberParams(volume_L=13.57, area_m2=0.04524)
dfs    = load_directory("path/to/csv/files")
results, summary = analyze_directory(dfs, params)
summary.to_csv("flux_results.csv", index=False)
```

## Interactive app

```bash
streamlit run app.py
```

Upload chamber CSV files, configure chamber geometry and analysis parameters, and export results — no coding required.

## Demo notebook

Open `chamber_analysis_demo.ipynb` for a step-by-step walkthrough of the full pipeline.

## Methods

See [METHODS.md](METHODS.md) for the flux calculation and model-selection algorithms with full citations.

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `pandas`, `scipy`, `plotly`.  
Optional (app): `streamlit`.

## Package structure

```
chamber_analysis/
├── io.py           # CSV loading
├── metadata.py     # auxiliary measurements (GPS, T, P, humidity, soil, battery)
├── plots.py        # Plotly figures
└── pyflux/         # physics engine (self-contained)
    ├── chamber.py      # ChamberParams
    ├── regression.py   # adaptive sliding-window R² optimisation
    ├── constants.py    # sensor metadata (GAS_META)
    ├── flux_term.py    # flux unit conversion + MDF
    ├── models.py       # LM and HM flux models
    ├── selection.py    # penalty-point model selection
    └── core.py         # analyze_sensor / analyze_file / analyze_directory
```

## Supported sensors

| Column | Gas | Reporting unit |
|---|---|---|
| `CO2 Dynament (ppm)` | CO₂ | µmol m⁻² s⁻¹ |
| `CO2 SCD30 (ppm)` | CO₂ | µmol m⁻² s⁻¹ |
| `CO2 Vaisala GMP252 (ppm)` | CO₂ | µmol m⁻² s⁻¹ |
| `N2O Dynament (ppm)` | N₂O | nmol m⁻² s⁻¹ |
| `CH4 Dynament (%)` | CH₄ | nmol m⁻² s⁻¹ |
| `O2 LuminOx (%)` | O₂ | µmol m⁻² s⁻¹ |

## References

Rheault et al. (2024). goFlux. *JOSS* 9(96), 6393. https://doi.org/10.21105/joss.06393  
Hutchinson & Mosier (1981). *SSSAJ* 45(2), 311–316.  
Hurvich & Tsai (1989). *Biometrika* 76(2), 297–307.

The code has been interactively optimized using Anthropic Claude Code Sonnet 4.6 and Caveman.
(cc) conrad.jackisch@tbt.tu-freiberg.de