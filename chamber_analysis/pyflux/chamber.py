"""Chamber physical parameters."""

from dataclasses import dataclass
from typing import Union


@dataclass
class ChamberParams:
    """Physical parameters of a closed static respiration chamber.

    Attributes
    ----------
    volume_L      : headspace volume [L]
    area_m2       : soil surface area enclosed by the chamber [m²]
    temperature_C : air temperature [°C], or ``"auto"`` to derive
                    from ``Temperature intern (degC)`` column in the data
    pressure_mbar : atmospheric pressure [mbar], or ``"auto"`` to derive
                    from ``Pressure intern (mbar)`` column in the data
    window_frac   : fraction of measurement duration used for the regression
                    window (passed to :func:`~chamber_analysis.regression.find_best_window`)
    """

    volume_L: float
    area_m2: float
    temperature_C: Union[float, str] = "auto"
    pressure_mbar: Union[float, str] = "auto"
    window_frac:    float = 0.8
    window_undercut: float = 0.3
    g_factor_max:   float = 4.0

    def __post_init__(self) -> None:
        if isinstance(self.temperature_C, (int, float)) and self.temperature_C < -90:
            raise ValueError(f"temperature_C={self.temperature_C} is below -90 °C")
        if isinstance(self.pressure_mbar, (int, float)) and not (
            500 < self.pressure_mbar < 1100
        ):
            raise ValueError(
                f"pressure_mbar={self.pressure_mbar} is outside the plausible range [500, 1100]"
            )
        if not 0 < self.window_frac <= 1:
            raise ValueError("window_frac must be in (0, 1]")
        if not 0 <= self.window_undercut < 1:
            raise ValueError("window_undercut must be in [0, 1)")
        if self.g_factor_max <= 0:
            raise ValueError("g_factor_max must be positive")
