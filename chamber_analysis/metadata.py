"""Auxiliary measurement metadata extraction from chamber DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# Maps raw CSV column names to clean attribute names.
# All non-GPS fields are averaged over the measurement period.
# GPS position columns use the median (robust to occasional fixes).
_COLUMN_MAP: Dict[str, str] = {
    "GPS (latitude)":                             "gps_latitude",
    "GPS (longitude)":                            "gps_longitude",
    "GPS (altitude_m)":                           "gps_altitude_m",
    "GPS (satellites)":                           "gps_satellites",
    "GPS (hdop)":                                 "gps_hdop",
    "GPS (fix)":                                  "gps_fix",
    "Temperature intern (degC)":                  "temperature_intern_C",
    "Humidity intern (%relH)":                    "humidity_intern_pct",
    "Pressure intern (mbar)":                     "pressure_intern_mbar",
    "Temperature extern (degC)":                  "temperature_extern_C",
    "Humidity extern (%relH)":                    "humidity_extern_pct",
    "Pressure extern (mbar)":                     "pressure_extern_mbar",
    "Temperature SCD30 (degC)":                   "temperature_scd30_C",
    "Humidity SDC30 (%relH)":                     "humidity_scd30_pct",
    "WaterContent SMT100 (vol %)":                "soil_water_content_pct",
    "Temperature SMT100 (degC)":                  "soil_temperature_C",
    "Permittivity dielectric coefficient SMT100": "soil_permittivity",
    "BatteryLevel (%)":                           "battery_level_pct",
}

# Columns for which median (not mean) is computed
_MEDIAN_ATTRS = {"gps_latitude", "gps_longitude", "gps_altitude_m"}

# Attributes where zero is physically impossible / means "sensor not connected".
# Temperature, humidity, and soil columns must NOT have zeros filtered out.
_ZERO_FILTER_ATTRS = {
    "gps_latitude", "gps_longitude", "gps_altitude_m",
    "gps_satellites", "gps_hdop", "gps_fix",
    "pressure_intern_mbar", "pressure_extern_mbar",
    "battery_level_pct",
}


@dataclass
class MeasurementMetadata:
    """Auxiliary data from a single chamber measurement file.

    All numeric fields are either mean or median values computed over
    the full measurement period. Zero readings are excluded only for
    pressure, GPS, and battery fields.
    """

    filename: str
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    duration_s: float
    n_samples: int

    # GPS
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude_m: Optional[float] = None
    gps_satellites: Optional[float] = None
    gps_hdop: Optional[float] = None
    gps_fix: Optional[float] = None

    # Internal sensor conditions
    temperature_intern_C: Optional[float] = None
    humidity_intern_pct: Optional[float] = None
    pressure_intern_mbar: Optional[float] = None

    # External atmospheric conditions
    temperature_extern_C: Optional[float] = None
    humidity_extern_pct: Optional[float] = None
    pressure_extern_mbar: Optional[float] = None

    # SCD30 auxiliary channels
    temperature_scd30_C: Optional[float] = None
    humidity_scd30_pct: Optional[float] = None

    # Soil (SMT100)
    soil_water_content_pct: Optional[float] = None
    soil_temperature_C: Optional[float] = None
    soil_permittivity: Optional[float] = None

    # Device health
    battery_level_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat dict; timestamps are ISO-8601 strings."""
        d = asdict(self)
        d["timestamp_start"] = str(self.timestamp_start)
        d["timestamp_end"] = str(self.timestamp_end)
        return d


def extract_metadata(
    df: pd.DataFrame,
    filename: str = "",
) -> MeasurementMetadata:
    """Extract auxiliary metadata by aggregating a loaded chamber DataFrame.

    Zero values are replaced with NaN for pressure and GPS columns (zero means
    sensor not connected or no fix). Temperature and humidity columns retain
    zero readings (0 °C / 0 % are physically valid).
    """

    def _agg(col: str, attr: str) -> Optional[float]:
        if col not in df.columns:
            return None
        s = df[col]
        if attr in _ZERO_FILTER_ATTRS:
            s = s.replace(0.0, np.nan)
        series = s.dropna()
        if series.empty:
            return None
        return float(series.median() if attr in _MEDIAN_ATTRS else series.mean())

    kwargs: Dict[str, Any] = {
        attr: _agg(col, attr) for col, attr in _COLUMN_MAP.items()
    }

    return MeasurementMetadata(
        filename=filename,
        timestamp_start=df["Timestamp"].iloc[0],
        timestamp_end=df["Timestamp"].iloc[-1],
        duration_s=float(df["time_sec"].iloc[-1]),
        n_samples=len(df),
        **kwargs,
    )
