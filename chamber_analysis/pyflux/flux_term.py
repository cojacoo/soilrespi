"""Flux unit-conversion factor and minimal detectable flux."""

from .constants import R_GAS


def flux_term(
    volume_L: float,
    area_m2: float,
    temperature_C: float,
    pressure_mbar: float,
    xH2O: float = 0.0,
) -> float:
    """Compute the flux unit-conversion factor.

    Multiplying a concentration slope [ppm s⁻¹] by ``flux_term``
    yields the molar gas flux [µmol m⁻² s⁻¹]:

        F [µmol m⁻² s⁻¹] = slope [ppm s⁻¹] × flux_term

    Derivation
    ----------
    Starting from the ideal-gas law (n/V = P / RT) and the definition
    of a ppm mole fraction (c_ppm = 1 × 10⁻⁶ mol mol⁻¹):

        dn/dt = dc/dt [ppm s⁻¹] × 1e-6 × (P V) / (R T)  [mol s⁻¹]

    Dividing by area A and converting mol → µmol:

        F = slope_ppm_s × (P_Pa × V_m³) / (R × T_K × A_m²)  [µmol m⁻² s⁻¹]

    Parameters
    ----------
    volume_L      : chamber headspace volume [L]
    area_m2       : soil surface area [m²]
    temperature_C : chamber air temperature [°C]
    pressure_mbar : atmospheric pressure [mbar]
    xH2O          : water-vapour mole fraction [mol mol⁻¹];
                    applies a dry-air correction (0 = already dry)
    """
    T_K  = temperature_C + 273.15
    P_Pa = pressure_mbar * 100.0
    V_m3 = volume_L * 1e-3
    return (P_Pa * V_m3) / (R_GAS * T_K * area_m2 * (1.0 - xH2O))


def mdf(
    precision_ppm: float,
    duration_s: float,
    ft: float,
) -> float:
    """Minimal Detectable Flux [µmol m⁻² s⁻¹].

    The smallest flux distinguishable from instrument noise given the
    sensor precision and measurement duration.

        MDF = (precision [ppm] / duration [s]) × flux_term

    Parameters
    ----------
    precision_ppm : instrument precision in ppm (convert native units first)
    duration_s    : total measurement duration [s]
    ft            : flux term from :func:`flux_term`
    """
    if duration_s <= 0.0:
        return float("inf")
    return (precision_ppm / duration_s) * ft
