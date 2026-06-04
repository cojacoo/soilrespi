"""pyFlux – pure-Python GHG flux calculation for closed respiration chambers.

Ports the core model-selection logic of the R goFlux package to Python
with no R dependency.

Reference
---------
Rheault, M., Defriez, D.-C., Perryman, C. R., Virkkala, A.-M., & Nuria, M.
(2024). goFlux: A user-friendly way to calculate GHG fluxes yourself,
regardless of user experience. *Journal of Open Source Software*, *9*(96),
6393. https://doi.org/10.21105/joss.06393
"""

from .chamber import ChamberParams
from .regression import find_best_window
from .constants import GAS_META, R_GAS
from .flux_term import flux_term, mdf
from .models import lm_flux, hm_flux, LMResult, HMResult
from .selection import best_flux, FluxResult
from .core import analyze_sensor, analyze_file, analyze_directory, SensorFluxResult

__all__ = [
    "ChamberParams",
    "find_best_window",
    "GAS_META",
    "R_GAS",
    "flux_term",
    "mdf",
    "lm_flux",
    "hm_flux",
    "LMResult",
    "HMResult",
    "best_flux",
    "FluxResult",
    "analyze_sensor",
    "analyze_file",
    "analyze_directory",
    "SensorFluxResult",
]
