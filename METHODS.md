# Technical Methods – Respiration Chamber Flux Analysis

## 1. Closed Chamber Principle

Greenhouse gas (GHG) fluxes are estimated using the static closed-chamber method. A chamber of known headspace volume $V$ is placed on the soil surface, enclosing area $A$. Gas concentrations inside the chamber are recorded continuously over a defined measurement period. The rate of concentration change is used to infer the soil–atmosphere exchange flux.

## 2. Concentration Time Series and Window Selection

Raw concentration time series may exhibit non-linear behaviour at the start (chamber not yet equilibrated) and at the end (approaching atmospheric saturation or diffusion limitation). To isolate the most linear segment, a sliding-window algorithm is applied. All combinations of window size $w \in [f(1-u)n,\; fn]$ and start position are evaluated, where $f$ is the target window fraction, $u$ is the maximum allowed reduction (default 0.30), and $n$ is the total number of observations. A linear regression is fitted at each position. The combination with the highest R² is selected; ties are resolved in favour of the larger window to honour the requested fraction wherever possible. This adaptive approach allows the window to retreat from plateau regions in strongly saturating fluxes while still preferring the full target length when the data are adequately linear.

## 3. Flux Calculation

The molar flux $F$ is derived from the ideal gas law:

$$F \;[\mu\text{mol}\,\text{m}^{-2}\,\text{s}^{-1}] = k \cdot \frac{P \cdot V}{R \cdot T \cdot A}$$

| Symbol | Description | Unit |
|--------|-------------|------|
| $k$ | Concentration slope (from regression) | ppm s⁻¹ |
| $P$ | Atmospheric pressure | Pa |
| $V$ | Chamber headspace volume | m³ |
| $R$ | Universal gas constant (8.314) | J mol⁻¹ K⁻¹ |
| $T$ | Chamber air temperature | K |
| $A$ | Enclosed soil surface area | m² |

A water-vapour correction factor $(1 - x_{\mathrm{H_2O}})$ is applied when humidity data are available.

## 4. Linear Model (LM)

Ordinary least-squares regression of concentration $c$ on elapsed time $t$:

$$c(t) = \beta_0 + k \cdot t$$

The flux standard error is propagated directly from the OLS slope standard error: $\mathrm{SE}_F = \mathrm{SE}_k \cdot \frac{PV}{RTA}$.

## 5. Hutchinson-Mosier Model (HM)

When concentrations show curvature (diffusion limitation, back-pressure), the non-linear Hutchinson-Mosier model provides a more accurate slope estimate [1, 2]:

$$c(t) = C_0 + (C_\infty - C_0)\left(1 - e^{-\kappa t}\right)$$

The initial flux rate (at $t = 0$) is:

$$k_{\mathrm{HM}} = (C_\infty - C_0)\cdot\kappa$$

Parameters $C_0$, $C_\infty$, and $\kappa$ are fitted by non-linear least squares (Levenberg–Marquardt). The flux standard error is obtained via the delta method from the parameter covariance matrix:

$$\mathrm{SE}_F = \sqrt{\mathbf{J}\,\Sigma\,\mathbf{J}^{\top}} \cdot \frac{PV}{RTA}$$

where $\mathbf{J} = [-\kappa,\; \kappa,\; C_\infty - C_0]$ is the Jacobian with respect to $[C_0, C_\infty, \kappa]$.

## 6. Model Selection

Both models are fitted to the same window. The better model is selected by a penalty-point scoring system following Rheault et al. [2]: each model receives one penalty point for being worse on each of four criteria — MAE, RMSE, AICc, and SE. The model with fewer penalty points is retained; ties are resolved in favour of HM. If the g-factor $|F_{\mathrm{HM}} / F_{\mathrm{LM}}|$ exceeds a threshold (default 2.0), the HM result is considered physically implausible and LM is used instead.

The corrected Akaike information criterion (AICc) penalises model complexity in small samples [3]:

$$\mathrm{AICc} = n\ln\!\left(\frac{\mathrm{RSS}}{n}\right) + 2k + \frac{2k(k+1)}{n - k - 1}$$

where $k$ is the number of free parameters (LM: $k=2$; HM: $k=3$) and $n$ is the number of observations in the window.

## 7. Minimal Detectable Flux (MDF)

The smallest flux distinguishable from instrument noise is:

$$\mathrm{MDF} = \frac{\sigma_{\mathrm{instr}}}{\Delta t} \cdot \frac{PV}{RTA}$$

where $\sigma_{\mathrm{instr}}$ is the instrument precision [ppm] and $\Delta t$ is the measurement duration [s].

## 8. Quality Flags

Each result is assigned one of three quality flags:

| Flag | Criterion |
|------|-----------|
| `good` | LM slope $p \leq 0.05$ and $|F| \geq \mathrm{MDF}$ |
| `below_mdf` | $|F| < \mathrm{MDF}$ (flux real but below detection) |
| `below_detection` | LM slope $p > 0.05$ (slope not distinguishable from zero) |

## 9. Implementation

pyFlux is implemented in Python using `scipy.stats.linregress` (LM) and `scipy.optimize.curve_fit` with Levenberg–Marquardt (HM). The model selection and scoring logic follows the R package goFlux [2]. No R dependency is required.

## References

[1] Hutchinson, G. L., & Mosier, A. R. (1981). Improved soil cover method for field measurement of nitrous oxide fluxes. *Soil Science Society of America Journal*, *45*(2), 311–316. https://doi.org/10.2136/sssaj1981.03615995004500020017x

[2] Rheault, M., Defriez, D.-C., Perryman, C. R., Virkkala, A.-M., & Nuria, M. (2024). goFlux: A user-friendly way to calculate GHG fluxes yourself, regardless of user experience. *Journal of Open Source Software*, *9*(96), 6393. https://doi.org/10.21105/joss.06393

[3] Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model selection in small samples. *Biometrika*, *76*(2), 297–307. https://doi.org/10.1093/biomet/76.2.297

[4] Livingston, G. P., & Hutchinson, G. L. (1995). Enclosure-based measurement of trace gas exchange: Applications and sources of error. In P. A. Matson & R. C. Harriss (Eds.), *Biogenic trace gases: Measuring emissions from soil and water* (pp. 14–51). Blackwell Science.
