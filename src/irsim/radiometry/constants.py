"""Physical constants for IR radiometry.

Never define these inline elsewhere. If you need one that is not here, add it here with a source.

Unit convention (see the ir-radiometry skill):
  wavelength      always micrometres
  spectral radiance   W m^-2 sr^-1 um^-1
  band radiance       W m^-2 sr^-1
  temperature         kelvin

Sources: CODATA 2018 exact SI definitions for h, c, k_B, q.
docs/physics-model.md §16.3
"""

from typing import Final

# --- SI base ---------------------------------------------------------------
H_PLANCK: Final[float] = 6.62607015e-34  # J s      (exact)
C_LIGHT: Final[float] = 2.99792458e8  # m s^-1   (exact)
K_BOLTZMANN: Final[float] = 1.380649e-23  # J K^-1   (exact)
Q_E: Final[float] = 1.602176634e-19  # C        (exact)

# --- Derived, in the project's practical units ------------------------------
# Planck first radiation constant for spectral RADIANCE (energy form):
#   L(lam, T) = C1L / (lam**5 * (exp(C2 / (lam * T)) - 1))
# units: W um^4 m^-2 sr^-1   with lam in um  ->  L in W m^-2 sr^-1 um^-1
C1L: Final[float] = 1.1910429723971884e8

# Photon form: L_q(lam, T) = C1Q / (lam**4 * (exp(C2 / (lam * T)) - 1))
# units: photons s^-1 um^3 m^-2 sr^-1  ->  L_q in ph s^-1 m^-2 sr^-1 um^-1
C1Q: Final[float] = 5.995849160000001e26

# Second radiation constant, h c / k_B, in um K
C2: Final[float] = 1.4387768775039337e4

# Stefan-Boltzmann, W m^-2 K^-4
SIGMA_SB: Final[float] = 5.670374419e-8

# Wien displacement constant, um K
WIEN_B: Final[float] = 2897.771955

# Apéry's constant zeta(3) = sum 1/n^3: the photon-form analogue of pi^4/15.
ZETA_3: Final[float] = 1.2020569031595942

# Photon-flux Stefan-Boltzmann constant, photons s^-1 m^-2 K^-3:
#   M_q = SIGMA_Q * T^3  (hemispherical photon exitance of a blackbody)
# Derivation: integrate L_q = (2c/lam^4) / (exp(hc/(lam k T)) - 1) over lam, substituting
# t = hc/(lam k T):
#   int L_q dlam = 2c (kT/hc)^3 int_0^inf t^2/(e^t - 1) dt = 2c (kT/hc)^3 * 2 zeta(3).
# Exitance is pi times radiance, so SIGMA_Q = 4 pi zeta(3) k^3 / (h^3 c^2) = 1.5205e15.
SIGMA_Q: Final[float] = (
    4.0 * 3.141592653589793 * ZETA_3 * K_BOLTZMANN**3 / (H_PLANCK**3 * C_LIGHT**2)
)

# --- Atmosphere ------------------------------------------------------------
# Specific gas constant of water vapour, J kg^-1 K^-1 (R / M_water = 8.314462618 / 0.018015):
# absolute humidity rho_v = e / (R_v T); the §7.3 factor 216.7 g m^-3 hPa^-1 K is 1e5 / R_v.
R_V_WATER: Final[float] = 461.5
# Koschmieder constant: meteorological optical range V is the path over which the contrast of a
# black target against the horizon sky falls to the 2 % threshold, V = ln(1/0.02) / gamma_vis
# (Koschmieder 1924; WMO Guide to Meteorological Instruments, MOR definition). Kept as the
# exact ln 50 so tau(V) = 0.02 holds to round-off; the conventional rounded value is 3.912.
KOSCHMIEDER: Final[float] = 3.912023005428146  # ln(50)

# --- Solar --------------------------------------------------------------------
# Total solar irradiance at 1 AU, W m^-2 (Kopp & Lean 2011, SORCE/TIM: 1360.8 +- 0.5). Upper bound
# for any surface irradiance in a weather file; the solar-path transmittance (M11) scales it.
SOLAR_CONSTANT_W_M2: Final[float] = 1361.0

# --- Semiconductor band gaps for the Arrhenius dark-current model (docs/physics-model.md §9.1) ----
# i_dark ∝ T^1.5 exp(-E_g / 2 k_B T). Values at the detectors' operating temperatures:
#   InSb   0.23 eV at 77 K   (Littler & Seiler 1985; Vurgaftman et al. 2001 -- 0.235 eV at 0 K)
#   InGaAs 0.75 eV at 300 K  (In0.53Ga0.47As lattice-matched to InP; Vurgaftman et al. 2001)
BAND_GAP_INSB_EV: Final[float] = 0.23
BAND_GAP_INGAAS_EV: Final[float] = 0.75
EV_PER_K: Final[float] = 8.617333262e-5  # k_B in eV/K (CODATA 2018)

# --- Temperature encoding and LUT grid --------------------------------------
# One definition, shared by the config layer, the LUT builder, the Warp/SPG kernels and the
# Isaac AOV adapter. The G-buffer carries c = (T - T_ENCODE_REF_K) / T_ENCODE_SPAN_K in float32,
# never raw kelvin through fp16 (docs/physics-model.md §13.3, §3.2 precision trap, CLAUDE.md #2).
T_ENCODE_REF_K: Final[float] = 200.0
T_ENCODE_SPAN_K: Final[float] = 800.0

# Band LUT grid (docs/physics-model.md §3.2 b, §13.5): T in [200, 1000] K at 0.05 K, 16001 entries.
LUT_T_MIN_K: Final[float] = 200.0
LUT_T_MAX_K: Final[float] = 1000.0
LUT_DT_K: Final[float] = 0.05
LUT_N: Final[int] = 16001  # round((LUT_T_MAX_K - LUT_T_MIN_K) / LUT_DT_K) + 1

# --- Numerical guards -------------------------------------------------------
# exp() overflows for x beyond ~709 in float64. Cold scenes at short wavelengths
# reach this easily (200 K at 0.8 um gives x ~ 90, but 100 K at 0.4 um gives ~360,
# and users will try worse). Clip and return ~0 radiance rather than inf/nan.
EXP_ARG_MAX: Final[float] = 700.0

# Plausibility bounds asserted at API boundaries.
WAVELENGTH_MIN_UM: Final[float] = 0.1
WAVELENGTH_MAX_UM: Final[float] = 1000.0
TEMPERATURE_MIN_K: Final[float] = 1.0
TEMPERATURE_MAX_K: Final[float] = 6000.0
