"""chamber_analysis – Respiration chamber data processing and GHG flux calculation.

Quickstart
----------
>>> from chamber_analysis import load_directory, ChamberParams
>>> from chamber_analysis.pyflux import analyze_directory
>>>
>>> params = ChamberParams(volume_L=13.57, area_m2=0.04524)
>>> dfs    = load_directory("/path/to/csv/files")
>>> results, summary = analyze_directory(dfs, params)
>>> summary.to_csv("flux_results.csv", index=False)
"""

from .pyflux.chamber import ChamberParams
from .io import load_csv, load_directory
from .metadata import MeasurementMetadata, extract_metadata
from .pyflux.regression import find_best_window
from . import plots
from . import pyflux

__all__ = [
    "ChamberParams",
    "load_csv",
    "load_directory",
    "MeasurementMetadata",
    "extract_metadata",
    "find_best_window",
    "plots",
    "pyflux",
]
