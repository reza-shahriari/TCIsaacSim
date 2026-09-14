#!/usr/bin/env python3
"""Generate the solar spectral irradiance files M11.3 needs (docs/decisions/0064).

    python scripts/generate_solar_spectra.py --out data/spectra/solar

Writes two files, both **modelled, not measured**, and both named after the model rather than
after a standard they are not:

  toa_planck_5778k.csv       extraterrestrial spectral irradiance at 1 AU, W m-2 um-1
  direct_normal_am1p5.csv    clear-sky direct-normal irradiance at AM1.5,  W m-2 um-1

Why modelled: this machine has no network, and ASTM E-490 / G-173 are tabulated standards, not
formulas. Inventing digits and labelling them "E-490" would be worse than a model that says what
it is. ADR 0064 records the provenance, the error, and exactly what to do when the real tables
are available -- the loader and every consumer are unchanged by the swap, because both files are
just (wavelength, irradiance) tables.

The extraterrestrial shape is a 5778 K Planck normalised to the 1361 W m-2 solar constant. That
is the standard effective-temperature model and it is a *good* one above ~0.7 um, which is where
every band this simulator configures lives; it is badly wrong in the ultraviolet (it puts ~12 %
of the constant below 0.4 um against the real ~8 %), so the loader refuses a band that reaches
below TRUSTED_MIN_UM.

The ground-level shape is a clear-sky direct-beam band model: Rayleigh scattering in Bird &
Riordan's analytic form, Angstrom aerosol, the Chappuis ozone band, the six major water-vapour
bands and the O2 A-band and CO2 bands as Gaussians in ln(lambda). Band positions are physics;
band strengths are calibrated so the AM1.5 integral is the 900.1 W m-2 that defines G-173's
direct-normal reference. The calibration is why the *integral* is not an independent check --
the checks that are independent are the ones on band shape (1.38 um and 1.87 um must be opaque).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from irsim.radiometry.constants import C1L, C2, SOLAR_CONSTANT_W_M2  # noqa: E402

SOLAR_EFFECTIVE_TEMPERATURE_K = 5778.0
#: G-173's direct + circumsolar reference total at AM1.5 (W m-2).
AM15_DIRECT_W_M2 = 900.1
AM15_AIRMASS = 1.5

LAMBDA_MIN_UM, LAMBDA_MAX_UM, LAMBDA_STEP_UM = 0.30, 20.0, 0.002

# Angstrom turbidity for the G-173 reference atmosphere: AOD 0.084 at 500 nm, exponent 1.14.
ANGSTROM_BETA = 0.084 * 0.5**1.14
ANGSTROM_ALPHA = 1.14
OZONE_CM = 0.34  # atm-cm, the G-173 reference column

#: (centre um, ln-width, strength) for each absorber. Centres are spectroscopy; strengths are the
#: calibrated quantity (ADR 0064). Water strengths carry a single common multiplier, fitted below.
WATER_BANDS = (
    (0.720, 0.012, 0.6),
    (0.822, 0.013, 0.5),
    (0.940, 0.020, 2.2),
    (1.135, 0.026, 3.4),
    (1.380, 0.036, 26.0),
    (1.870, 0.044, 34.0),
    (2.700, 0.060, 60.0),
    (6.300, 0.090, 90.0),
)
UNIFORM_BANDS = (
    (0.762, 0.005, 1.1),  # O2 A-band
    (1.270, 0.006, 0.15),  # O2
    (2.010, 0.015, 2.0),  # CO2
    (2.060, 0.012, 1.6),  # CO2
    (4.300, 0.040, 120.0),  # CO2, opaque
    (15.00, 0.120, 200.0),  # CO2, opaque
)


def planck_exitance(lam_um: NDArray[np.float64], t_k: float) -> NDArray[np.float64]:
    """pi B_lambda(T), W m-2 um-1."""
    return np.asarray(np.pi * C1L / lam_um**5 / np.expm1(C2 / (lam_um * t_k)))


def toa_irradiance(lam_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extraterrestrial spectral irradiance, normalised so the full integral is the constant."""
    shape = planck_exitance(lam_um, SOLAR_EFFECTIVE_TEMPERATURE_K)
    return np.asarray(shape * (SOLAR_CONSTANT_W_M2 / float(np.trapezoid(shape, lam_um))))


def _gaussian_depth(
    lam_um: NDArray[np.float64], bands: tuple[tuple[float, float, float], ...], scale: float
) -> NDArray[np.float64]:
    """Sum of Gaussians in ln(lambda): an optical depth, so always >= 0."""
    ln_lam = np.log(lam_um)
    total = np.zeros_like(lam_um)
    for centre, width, strength in bands:
        total += scale * strength * np.exp(-0.5 * ((ln_lam - np.log(centre)) / width) ** 2)
    return total


def rayleigh_depth(lam_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bird & Riordan (1984) analytic Rayleigh optical depth at one air mass."""
    return np.asarray(1.0 / (lam_um**4 * (115.6406 - 1.3366 / lam_um**2)))


def aerosol_depth(lam_um: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(ANGSTROM_BETA * lam_um**-ANGSTROM_ALPHA)


def ozone_depth(lam_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Chappuis band only -- the Hartley/Huggins bands sit below the trusted range anyway."""
    return np.asarray(OZONE_CM * 0.05 * np.exp(-0.5 * ((lam_um - 0.60) / 0.07) ** 2))


def transmittance(
    lam_um: NDArray[np.float64], airmass: float, water_scale: float
) -> NDArray[np.float64]:
    depth = (
        rayleigh_depth(lam_um)
        + aerosol_depth(lam_um)
        + ozone_depth(lam_um)
        + _gaussian_depth(lam_um, WATER_BANDS, water_scale)
        + _gaussian_depth(lam_um, UNIFORM_BANDS, 1.0)
    )
    return np.asarray(np.exp(-depth * airmass))


def fit_water_scale(lam_um: NDArray[np.float64], toa: NDArray[np.float64]) -> float:
    """Bisect the common water strength so the AM1.5 direct integral is AM15_DIRECT_W_M2."""

    def total(scale: float) -> float:
        return float(np.trapezoid(toa * transmittance(lam_um, AM15_AIRMASS, scale), lam_um))

    lo, hi = 1e-4, 100.0
    if total(lo) < AM15_DIRECT_W_M2:
        raise RuntimeError(f"even a dry atmosphere passes only {total(lo):.1f} W/m2")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) > AM15_DIRECT_W_M2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


HEADER_TOA = """\
# Extraterrestrial solar spectral irradiance at 1 AU. Columns: wavelength in MICROMETRES,
# spectral irradiance in W m-2 um-1.
#
# STATUS: MODELLED, NOT MEASURED. This is a {t:.0f} K Planck exitance normalised so that its
# integral over {lo}-{hi} um is the {e:.0f} W m-2 solar constant -- the standard
# effective-temperature model of the sun, NOT the ASTM E-490 tabulated standard. Do not cite it
# as E-490. ADR 0064 records why (no network on the authoring machine; inventing digits under a
# standard's name is worse than a model that says what it is) and how to swap the real table in:
# drop an E-490 CSV here in the same two columns, and nothing else in the repository changes.
#
# Expected error: a few percent above ~0.7 um, which is where every band this simulator
# configures lives, and badly wrong in the ultraviolet -- this model puts {uv:.1f} % of the
# constant below 0.4 um against a real ~8 %. irsim.radiometry.solar therefore REFUSES a spectral
# response reaching below 0.7 um rather than quietly integrating the wrong region.
#
# Model band fractions (recorded so that swapping in E-490 shows exactly what moved):
{fractions}
# Generated by scripts/generate_solar_spectra.py on 2026-09-15 (roadmap M11.3).
wavelength_um,irradiance_w_m2_um
"""

HEADER_GROUND = """\
# Clear-sky DIRECT-NORMAL solar spectral irradiance at the ground, air mass {am}. Columns:
# wavelength in MICROMETRES, spectral irradiance in W m-2 um-1.
#
# STATUS: MODELLED, NOT MEASURED. Not ASTM G-173. The extraterrestrial input is the companion
# toa_planck_5778k.csv; the atmosphere is a clear-sky direct-beam band model --
#   * Rayleigh scattering, Bird & Riordan (1984) analytic optical depth
#   * Angstrom aerosol, beta = {beta:.5f}, alpha = {alpha:.2f} (AOD 0.084 at 500 nm)
#   * ozone Chappuis band, {o3} atm-cm
#   * the major water-vapour bands and the O2 A-band / CO2 bands as Gaussians in ln(lambda)
# Band CENTRES are spectroscopy. Band STRENGTHS carry one common water multiplier, fitted here to
# {w:.6f} so that the AM{am} integral is the {g:.1f} W m-2 that defines G-173's direct-normal
# reference. The integral is therefore NOT an independent check of this file; the checks that are
# independent are on shape -- 1.38 um and 1.87 um come out opaque, which is why a SWIR camera is
# blind there, and that was not fitted.
#
# Total: {total:.2f} W m-2. Transmittance at 1.38 um {t138:.4f}, at 1.87 um {t187:.4f},
# at 1.06 um {t106:.4f}, at 1.55 um {t155:.4f}.
# Generated by scripts/generate_solar_spectra.py on 2026-09-15 (roadmap M11.3).
wavelength_um,irradiance_w_m2_um
"""


def _rows(lam: NDArray[np.float64], value: NDArray[np.float64]) -> str:
    return "\n".join(f"{a:.4f},{b:.6e}" for a, b in zip(lam, value, strict=True)) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/spectra/solar")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lam = np.arange(LAMBDA_MIN_UM, LAMBDA_MAX_UM + 0.5 * LAMBDA_STEP_UM, LAMBDA_STEP_UM)
    toa = toa_irradiance(lam)

    def frac(a: float, b: float, y: NDArray[np.float64]) -> float:
        m = (lam >= a) & (lam <= b)
        return float(np.trapezoid(y[m], lam[m]))

    bands = (
        (0.3, 0.4, "UV  0.30-0.40"),
        (0.4, 0.7, "VIS 0.40-0.70"),
        (0.75, 1.0, "NIR 0.75-1.00"),
        (0.9, 1.7, "SWIR 0.90-1.70"),
        (3.0, 5.0, "MWIR 3.0-5.0 "),
        (7.5, 13.5, "LWIR 7.5-13.5"),
    )
    lines = "".join(
        f"#   {name}: {frac(a, b, toa):8.2f} W m-2 "
        f"({frac(a, b, toa) / SOLAR_CONSTANT_W_M2 * 100:5.2f} %)\n"
        for a, b, name in bands
    )
    (out / "toa_planck_5778k.csv").write_text(
        HEADER_TOA.format(
            t=SOLAR_EFFECTIVE_TEMPERATURE_K,
            lo=LAMBDA_MIN_UM,
            hi=LAMBDA_MAX_UM,
            e=SOLAR_CONSTANT_W_M2,
            uv=frac(0.3, 0.4, toa) / SOLAR_CONSTANT_W_M2 * 100,
            fractions=lines,
        )
        + _rows(lam, toa),
        encoding="utf-8",
    )

    scale = fit_water_scale(lam, toa)
    tau = transmittance(lam, AM15_AIRMASS, scale)
    ground = toa * tau
    (out / "direct_normal_am1p5.csv").write_text(
        HEADER_GROUND.format(
            am=AM15_AIRMASS,
            beta=ANGSTROM_BETA,
            alpha=ANGSTROM_ALPHA,
            o3=OZONE_CM,
            w=scale,
            g=AM15_DIRECT_W_M2,
            total=float(np.trapezoid(ground, lam)),
            t138=float(np.interp(1.38, lam, tau)),
            t187=float(np.interp(1.87, lam, tau)),
            t106=float(np.interp(1.06, lam, tau)),
            t155=float(np.interp(1.55, lam, tau)),
        )
        + _rows(lam, ground),
        encoding="utf-8",
    )

    print(
        f"TOA    total {float(np.trapezoid(toa, lam)):.2f} W/m2 -> {out / 'toa_planck_5778k.csv'}"
    )
    print(f"AM1.5  total {float(np.trapezoid(ground, lam)):.2f} W/m2, water scale {scale:.6f}")
    for a, b, name in bands:
        print(f"   {name}: TOA {frac(a, b, toa):8.2f}  ground {frac(a, b, ground):8.2f} W/m2")
    for q in (0.76, 0.94, 1.06, 1.135, 1.38, 1.55, 1.87, 2.2):
        print(f"   tau({q:.3f} um) = {float(np.interp(q, lam, tau)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
