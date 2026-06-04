"""Physical constants and per-sensor metadata for pyFlux."""

# Universal gas constant [J mol⁻¹ K⁻¹]
R_GAS: float = 8.314

# Per-sensor configuration.
#
# Keys are the exact column names produced by load_csv().
# Fields:
#   factor_to_ppm  – multiply native sensor reading by this to obtain ppm
#   precision      – instrument precision in native sensor units (used for MDF)
#   report_factor  – multiply the raw flux [µmol m⁻² s⁻¹] to reach the
#                    reporting unit (e.g. 1000 → nmol m⁻² s⁻¹)
#   report_unit    – SI unit string for output tables and plots
GAS_META: dict = {
    "CO2 Dynament (ppm)": {
        "factor_to_ppm": 1.0,
        "precision":     10.0,       # ±10 ppm (typical NDIR)
        "report_factor": 1.0,
        "report_unit":   "µmol m⁻² s⁻¹",
    },
    "CO2 SCD30 (ppm)": {
        "factor_to_ppm": 1.0,
        "precision":     2.5,        # ±(30 ppm + 3 %) – use conservative 2.5 ppm
        "report_factor": 1.0,
        "report_unit":   "µmol m⁻² s⁻¹",
    },
    "CO2 Vaisala GMP252 (ppm)": {
        "factor_to_ppm": 1.0,
        "precision":     1.0,        # ±1 ppm (high-quality NDIR)
        "report_factor": 1.0,
        "report_unit":   "µmol m⁻² s⁻¹",
    },
    "N2O Dynament (ppm)": {
        "factor_to_ppm": 1.0,
        "precision":     0.1,        # ±0.1 ppm
        "report_factor": 1_000.0,    # µmol → nmol
        "report_unit":   "nmol m⁻² s⁻¹",
    },
    "CH4 Dynament (%)": {
        "factor_to_ppm": 10_000.0,   # % → ppm
        "precision":     0.001,      # ±0.001 % ≈ ±10 ppm after conversion
        "report_factor": 1_000.0,
        "report_unit":   "nmol m⁻² s⁻¹",
    },
    "O2 LuminOx (%)": {
        "factor_to_ppm": 10_000.0,
        "precision":     0.01,       # ±0.01 % ≈ ±100 ppm
        "report_factor": 1.0,
        "report_unit":   "µmol m⁻² s⁻¹",
    },
}
