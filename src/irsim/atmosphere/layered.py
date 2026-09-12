"""Layered slant-path atmosphere with an exponential-sum band model (§7.1, §5.3; ADR 0071).

One grey γ_B per band is exact only for spectrally flat extinction; real bands mix opaque lines,
moderate lines and a transparent window, and the band transmittance is a *sum of exponentials*
(M8.7's curve of growth). Per band the model is

    τ_B(path) = Σ_k w_k exp(−∫ γ_k(h) ds),      γ_k(h) = γ_k,0 e^{−h/H_k} + γ_aer,B e^{−h/H_aer}

with one term per **spectral class** k -- a sub-band of the response (window, water-vapour lines,
band-edge lines, an opaque CO₂ core) whose weight w_k is its Planck-weighted share of the band
(300 K for the emissive bands, 5800 K for the reflective ones) and whose surface extinction is
a multiplier g_k times the preset's molecular γ_mol,B(w) ("water" classes, scale height H_w)
or an absolute value ("air" classes: CO₂, scale height H_air). The non-opaque classes are
rescaled by one factor s so that the *horizontal 200 m* transmittance equals the grey preset's
(M8.3's table anchor) at the current weather; opaque classes sit on top with fixed extinction
(ADR 0048 read the table for the transparent part of the band). Along a ray at elevation θ
(flat earth, h = s sin θ) the column integrals of the exponential profiles are analytic, so
transmittance is closed-form; path radiance L_path = Σ_k w_k ∫ γ_k(h) τ_k(s) L_B(T(h)) ds with
T(h) = T_air − Γ min(h, h_tropopause) is a quadrature, and the **sky radiance is the column
emission** L_sky,B(θ) = L_path,B(∞, θ) (space contributes nothing). At θ = 0 the model reduces
to horizontal Beer–Lambert per term and, with one class, to M8.1 exactly.

Calibration (ADR 0071): the window multipliers are set so the clear dry LWIR sky reads
−40 °C at 15° elevation and the MWIR sky warmer than +10 °C, the two R13 anchors (Tucson,
clear, low humidity). Everything else is spectroscopy (class edges) or the preset.

docs/physics-model.md §7.1 [R13], §7.4, §5.3, Appendix A #2
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from irsim.atmosphere.extinction import gamma_aerosol_visible
from irsim.atmosphere.humidity import gamma_molecular
from irsim.config.atmosphere import AtmospherePreset
from irsim.config.bands import NOMINAL_RANGES_UM, BandId
from irsim.radiometry.band_integration import quadrature_grid, simpson
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.radiometry.planck import spectral_radiance
from irsim.radiometry.spectral_response import SpectralResponse
from irsim.thermal.weather import WeatherSeries

__all__ = [
    "ANCHOR_DISTANCE_M",
    "SpectralClass",
    "BAND_CLASSES",
    "WEIGHT_T_REF_K",
    "class_weights",
    "ExponentialSum",
    "column_length",
    "LayeredAtmosphere",
    "fit_exponential_sum",
    "exponential_sum_from_piecewise",
    "T_SPACE_K",
    "VISIBLE_RANGE_UM",
]

ANCHOR_DISTANCE_M = 200.0  # the §7.2 table column the grey presets were fitted at
T_SPACE_K = 0.0  # the space term: L_B(2.7 K) is zero at every LUT precision
Kind = Literal["water", "air"]


@dataclass(frozen=True)
class SpectralClass:
    """A sub-band of the response with one extinction behaviour.

    ``multiplier`` is g_k (× γ_mol,B, then × the anchor scale s) for non-opaque water classes,
    an absolute γ (m⁻¹, × s) for non-opaque air classes, and an absolute γ for opaque classes
    (never rescaled, excluded from the 200 m anchor).
    """

    name: str
    edges_um: tuple[tuple[float, float], ...]
    kind: Kind
    multiplier: float
    opaque: bool = False


# Class edges: H2O bands at 0.94, 1.1, 1.4, 2.7, 6.3 um and the rotational band beyond 13 um;
# CO2 nu3 at 4.3 um (opaque within metres), the CO2/N2O complex 4.45-4.6 and 4.85-5.0 um.
# Multipliers ESTIMATED; the window values are the R13 calibration (ADR 0071).
BAND_CLASSES: dict[str, tuple[SpectralClass, ...]] = {
    "lwir": (
        SpectralClass("edges", ((7.0, 7.8), (13.2, 14.5)), "water", 40.0),
        SpectralClass("lines", ((7.8, 8.3), (12.5, 13.2)), "water", 5.0),
        SpectralClass("window", ((8.3, 12.5),), "water", 0.3),
    ),
    "mwir": (
        SpectralClass("h2o_2p7", ((2.55, 2.95),), "water", 0.05, opaque=True),
        SpectralClass("co2_4p3", ((4.17, 4.45),), "air", 0.5, opaque=True),
        SpectralClass("h2o_wing", ((2.95, 3.35), (5.0, 5.6)), "water", 8.0),
        SpectralClass("co2_n2o", ((4.45, 4.6), (4.85, 5.0)), "air", 2.0e-3),
        SpectralClass("window", ((2.0, 2.55), (3.35, 4.17), (4.6, 4.85)), "water", 0.3),
    ),
    "swir": (
        SpectralClass("h2o_1p4", ((1.33, 1.48),), "water", 0.05, opaque=True),
        SpectralClass("h2o_1p1", ((1.10, 1.17),), "water", 10.0),
        SpectralClass("window", ((0.85, 1.10), (1.17, 1.33), (1.48, 1.75)), "water", 0.5),
    ),
    "nir": (
        SpectralClass("h2o_0p94", ((0.90, 0.98),), "water", 10.0),
        SpectralClass("window", ((0.7, 0.90), (0.98, 1.05)), "water", 0.5),
    ),
    "visible": (SpectralClass("window", ((0.35, 0.80),), "water", 1.0),),
}
WEIGHT_T_REF_K: dict[str, float] = {
    "lwir": 300.0,
    "mwir": 300.0,
    "swir": 5800.0,
    "nir": 5800.0,
    "visible": 5800.0,
}


VISIBLE_RANGE_UM = (0.4, 0.7)


def _nominal_response(band: str) -> SpectralResponse:
    lo, hi = VISIBLE_RANGE_UM if band == "visible" else NOMINAL_RANGES_UM[cast(BandId, band)]
    return SpectralResponse(np.array([lo, hi]), np.array([1.0, 1.0]), f"<top-hat {band}>", "")


def class_weights(
    band: str, response: SpectralResponse | None = None, t_ref_k: float | None = None
) -> NDArray[np.float64]:
    """Planck-weighted share of the band per spectral class; sums to 1; every wavelength of the
    response must fall in some class (the class edges cover the nominal bands with margin)."""
    classes = BAND_CLASSES[band]
    resp = response if response is not None else _nominal_response(band)
    t_ref = WEIGHT_T_REF_K[band] if t_ref_k is None else t_ref_k
    grid = quadrature_grid(resp)
    dl = float(grid[1] - grid[0])
    weight = resp.resampled(grid) * spectral_radiance(grid, np.asarray(t_ref, dtype=np.float64))
    total = float(simpson(weight, dl))
    out = np.zeros(len(classes))
    covered = np.zeros(grid.shape, dtype=bool)
    for i, c in enumerate(classes):
        mask = np.zeros(grid.shape, dtype=bool)
        for lo, hi in c.edges_um:
            mask |= (grid >= lo) & (grid < hi)
        covered |= mask
        out[i] = float(simpson(np.where(mask, weight, 0.0), dl)) / total
    if np.any(weight[~covered] > 1e-6 * weight.max()):
        lo, hi = grid[~covered][0], grid[~covered][-1]
        raise ValueError(f"band {band!r}: response has weight outside its classes ({lo}-{hi} um)")
    return np.asarray(out / out.sum(), dtype=np.float64)


def column_length(distance_m: Any, elevation_rad: float, scale_height_m: float) -> Any:
    """∫₀^d e^{−s sinθ/H} ds on a flat-earth ray: d at θ = 0, else H/sinθ (1 − e^{−d sinθ/H})."""
    d = np.asarray(distance_m, dtype=np.float64)
    st = math.sin(elevation_rad)
    if st <= 0.0:
        return d
    with np.errstate(over="ignore"):
        return scale_height_m / st * (1.0 - np.exp(-d * st / scale_height_m))


def _scaled(gamma: float, column: NDArray[np.float64]) -> NDArray[np.float64]:
    if gamma == 0.0:
        return np.zeros_like(column)
    return np.asarray(gamma * column, dtype=np.float64)


@dataclass(frozen=True)
class ExponentialSum:
    """τ_B(path) = Σ_k w_k exp(−γ_k,0 C_k(path) − γ_aer C_aer(path)) for one band and weather."""

    weights: NDArray[np.float64]
    gamma_0: NDArray[np.float64]  # surface extinction per class, m^-1 (aerosol excluded)
    scale_heights_m: NDArray[np.float64]  # per class
    gamma_aerosol: float
    aerosol_scale_height_m: float
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if abs(float(self.weights.sum()) - 1.0) > 1e-9 or np.any(self.weights < 0.0):
            raise ValueError("class weights must be non-negative and sum to 1")
        if np.any(self.gamma_0 < 0.0) or self.gamma_aerosol < 0.0:
            raise ValueError("extinctions must be non-negative")

    @property
    def n_terms(self) -> int:
        return int(self.weights.size)

    def optical_depths(self, distance_m: Any, elevation_rad: float) -> NDArray[np.float64]:
        """Per class optical depth along the ray, shape (n_terms, *distance.shape)."""
        d = np.asarray(distance_m, dtype=np.float64)
        col_aer = np.asarray(
            column_length(d, elevation_rad, self.aerosol_scale_height_m), dtype=np.float64
        )
        # 0 * inf (a zero extinction on an infinite horizontal path) is 0, not NaN
        aer = _scaled(self.gamma_aerosol, col_aer)
        out = np.empty((self.n_terms, *d.shape))
        for k in range(self.n_terms):
            col = column_length(d, elevation_rad, float(self.scale_heights_m[k]))
            out[k] = _scaled(float(self.gamma_0[k]), np.asarray(col, dtype=np.float64)) + aer
        return out

    def transmittance(self, distance_m: Any, elevation_rad: float = 0.0) -> NDArray[np.float64]:
        od = self.optical_depths(distance_m, elevation_rad)
        with np.errstate(over="ignore", invalid="ignore"):
            tau = np.tensordot(self.weights, np.exp(-od), axes=1)
        d = np.asarray(distance_m, dtype=np.float64)
        tau = np.where(np.isinf(d) & (math.sin(elevation_rad) <= 0.0), 0.0, tau)
        return np.asarray(tau, dtype=np.float64)

    def gamma_at(self, height_m: Any) -> NDArray[np.float64]:
        """Per class extinction at height h (aerosol included), shape (n_terms, *h.shape)."""
        h = np.asarray(height_m, dtype=np.float64)
        aer = self.gamma_aerosol * np.exp(-h / self.aerosol_scale_height_m)
        return np.stack(
            [
                self.gamma_0[k] * np.exp(-h / self.scale_heights_m[k]) + aer
                for k in range(self.n_terms)
            ]
        )

    def effective_gamma(self) -> float:
        """d→0 slope Σ w_k (γ_k,0 + γ_aer): what a very short horizontal path sees."""
        return float(np.dot(self.weights, self.gamma_0 + self.gamma_aerosol))

    def path_radiance_per_class(
        self,
        distance_m: float,
        elevation_rad: float,
        lb_of_height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_steps: int = 4000,
        u_max: float = 40.0,
        s_max_m: float = 300e3,
    ) -> NDArray[np.float64]:
        """Per class ∫₀^d γ_k(h) τ_k(s) L_B(T(h)) ds (unweighted), shape (n_terms,).

        Each class is integrated in its own optical-depth coordinate u = od_k(s):
        ∫ L_B(T(h(s(u)))) e^{−u} du on a uniform u grid with s(u) from the analytic, monotone
        od_k(s) -- so an opaque class whose emission comes from the first metres is resolved as
        well as a window whose emission comes from kilometres up. Beyond u_max = 40 nothing is
        left (e^{−40}). At θ = 0 the ray stays at h = 0 and the result is closed-form.
        """
        st = math.sin(elevation_rad)
        lb0 = float(lb_of_height(np.zeros(1))[0])
        out = np.zeros(self.n_terms)
        if st <= 0.0:
            if math.isinf(distance_m):
                out[self.gamma_0 + self.gamma_aerosol > 0.0] = lb0
                return out
            tau_k = np.exp(-self.optical_depths(distance_m, 0.0))
            return np.asarray((1.0 - tau_k) * lb0, dtype=np.float64)
        upper = s_max_m if math.isinf(distance_m) else min(float(distance_m), s_max_m)
        if upper <= 0.0:
            return out
        s_dense = np.concatenate([[0.0], np.logspace(-3, math.log10(upper), 20000)])
        od_dense = self.optical_depths(s_dense, elevation_rad)  # (K, n)
        n = n_steps if n_steps % 2 == 0 else n_steps + 1
        for k in range(self.n_terms):
            od_end = float(od_dense[k, -1])
            if od_end <= 0.0:
                continue
            u_top = min(od_end, u_max)
            u = np.linspace(0.0, u_top, n + 1)
            s_of_u = np.interp(u, od_dense[k], s_dense)
            lb = lb_of_height(s_of_u * st)
            out[k] = float(simpson(lb * np.exp(-u), float(u[1] - u[0])))
        return out

    def path_radiance(
        self,
        distance_m: float,
        elevation_rad: float,
        lb_of_height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_steps: int = 4000,
        u_max: float = 40.0,
        s_max_m: float = 300e3,
    ) -> float:
        """Σ_k w_k ∫₀^d γ_k(h) τ_k(s) L_B(T(h)) ds  (d = ∞ → the column emission)."""
        per_class = self.path_radiance_per_class(
            distance_m, elevation_rad, lb_of_height, n_steps, u_max, s_max_m
        )
        return float(np.dot(self.weights, per_class))


def fit_exponential_sum(
    distances_m: NDArray[np.float64], tau: NDArray[np.float64], n_terms: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Least-squares (w_k, γ_k) with w ≥ 0, Σ w = 1, γ ≥ 0 to sampled τ(d): the tool for
    fitting a band's curve of growth from a spectral calculation (M8.7) or a MODTRAN run."""
    d = np.asarray(distances_m, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    if n_terms < 1 or d.shape != t.shape or np.any(d < 0.0) or np.any((t <= 0.0) | (t > 1.0)):
        raise ValueError("need matching distances >= 0 and 0 < tau <= 1")
    if n_terms == 1:
        g_single = float(-np.sum(d * np.log(t)) / np.sum(d * d))
        return np.array([1.0]), np.array([g_single])
    g0 = float(-np.log(t[-1]) / d[-1])
    x0 = np.concatenate([np.zeros(n_terms - 1), np.log(g0 * np.logspace(-1, 1, n_terms))])

    def unpack(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        logits = np.concatenate([[0.0], x[: n_terms - 1]])
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        return w, np.exp(x[n_terms - 1 :])

    def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        w, g = unpack(x)
        model = np.exp(-np.outer(g, d)).T @ w
        return np.asarray(np.log(model) - np.log(t), dtype=np.float64)

    best = None
    for trial in range(6):
        start = x0 + (0.0 if trial == 0 else np.random.default_rng(trial).normal(0.0, 1.0, x0.size))
        res = least_squares(residual, start, method="lm" if d.size >= x0.size else "trf")
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    w, g = unpack(best.x)
    order = np.argsort(g)
    return w[order], g[order]


def exponential_sum_from_piecewise(
    response: SpectralResponse,
    gamma_of_lambda: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    breakpoints_um: tuple[float, ...],
    t_ref_k: float = 300.0,
) -> ExponentialSum:
    """The exact exponential sum of a piecewise-constant γ(λ): one term per interval between
    breakpoints, weight = the interval's Planck-weighted share, γ = γ(λ) at its midpoint. Used
    to check the model against M8.7's spectral quadrature (horizontal paths, no profile)."""
    lo, hi = response.support_um
    edges = sorted({lo, hi, *[b for b in breakpoints_um if lo < b < hi]})
    grid = quadrature_grid(response)
    dl = float(grid[1] - grid[0])
    weight = response.resampled(grid) * spectral_radiance(
        grid, np.asarray(t_ref_k, dtype=np.float64)
    )
    total = float(simpson(weight, dl))
    w = []
    g = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        mask = (grid >= a) & (grid < b) if b < hi else (grid >= a) & (grid <= b)
        w.append(float(simpson(np.where(mask, weight, 0.0), dl)) / total)
        g.append(float(gamma_of_lambda(np.array([0.5 * (a + b)]))[0]))
    w_arr = np.asarray(w) / sum(w)
    return ExponentialSum(
        weights=w_arr,
        gamma_0=np.asarray(g),
        scale_heights_m=np.full(len(g), 1e30),
        gamma_aerosol=0.0,
        aerosol_scale_height_m=1e30,
        names=tuple(f"{a}-{b}um" for a, b in zip(edges[:-1], edges[1:], strict=True)),
    )


class LayeredAtmosphere:
    """The MS.1 model bound to a preset and the shared WeatherSeries (CLAUDE.md #6)."""

    def __init__(
        self,
        preset: AtmospherePreset,
        weather: WeatherSeries,
        luts: Mapping[str, BandLUT] | None = None,
        responses: Mapping[str, SpectralResponse] | None = None,
    ) -> None:
        if not isinstance(weather, WeatherSeries):
            raise TypeError("LayeredAtmosphere takes the WeatherSeries object, never a path (#6)")
        self._preset = preset
        self._weather = weather
        self._luts = dict(luts or {})
        self._responses = dict(responses or {})
        self._weights: dict[str, NDArray[np.float64]] = {}

    @property
    def preset(self) -> AtmospherePreset:
        return self._preset

    @property
    def weather(self) -> WeatherSeries:
        return self._weather

    def weights(self, band: str) -> NDArray[np.float64]:
        if band not in self._weights:
            self._weights[band] = class_weights(band, self._responses.get(band))
        return self._weights[band]

    # -- the per-band exponential sum at time t ---------------------------------------
    def exponential_sum(self, band: str, t_s: float) -> ExponentialSum:
        sample = self._weather.at(t_s)
        w_h2o = sample.absolute_humidity_g_m3
        coeffs = self._preset.bands[band]
        gamma_mol = gamma_molecular(w_h2o, coeffs.gamma0_per_m, coeffs.beta_per_m_per_g_m3)
        vis = self._preset.bands["visible"]
        gamma_mol_vis = gamma_molecular(w_h2o, vis.gamma0_per_m, vis.beta_per_m_per_g_m3)
        gamma_aer = coeffs.aerosol_ratio_to_visible * gamma_aerosol_visible(
            sample.visibility_m, gamma_mol_vis
        )
        gamma_grey = gamma_mol + gamma_aer
        classes = BAND_CLASSES[band]
        weights = self.weights(band)
        profile = self._preset.profile
        heights = np.array(
            [
                profile.water_vapour_scale_height_m
                if c.kind == "water"
                else profile.air_scale_height_m
                for c in classes
            ]
        )
        base = np.array(
            [
                (
                    c.multiplier
                    if c.opaque
                    else c.multiplier * (gamma_mol if c.kind == "water" else 1.0)
                )
                for c in classes
            ]
        )
        opaque = np.array([c.opaque for c in classes])
        free = ~opaque
        target = math.exp(-gamma_grey * ANCHOR_DISTANCE_M) * float(weights[free].sum())

        def anchored(scale: float) -> float:
            g = np.where(free, scale * base, base)
            return float(np.sum(weights[free] * np.exp(-(g[free] + gamma_aer) * ANCHOR_DISTANCE_M)))

        if not free.any() or base[free].max() <= 0.0:
            scale = 1.0
        else:
            hi = 1.0
            while anchored(hi) > target and hi < 1e8:
                hi *= 10.0
            if anchored(0.0) < target - 1e-12:
                raise ValueError(
                    f"band {band!r}: the aerosol alone makes tau(200 m) < the grey preset's; "
                    "class table and preset disagree"
                )
            scale = (
                0.0 if anchored(0.0) <= target else brentq(lambda x: anchored(x) - target, 0.0, hi)
            )
        gamma_0 = np.where(free, scale * base, base)
        return ExponentialSum(
            weights=weights,
            gamma_0=np.asarray(gamma_0, dtype=np.float64),
            scale_heights_m=heights,
            gamma_aerosol=gamma_aer,
            aerosol_scale_height_m=profile.aerosol_scale_height_m,
            names=tuple(c.name for c in classes),
        )

    # -- temperatures and radiances along the column ------------------------------------
    def air_temperature_at(self, t_s: float, height_m: Any) -> NDArray[np.float64]:
        t0 = self._weather.at(t_s).t_air_k
        p = self._preset.profile
        h = np.minimum(np.asarray(height_m, dtype=np.float64), p.tropopause_m)
        return np.asarray(t0 - p.lapse_rate_k_per_m * h, dtype=np.float64)

    def _lb_of_height(
        self, band: str, t_s: float, quantity: Quantity
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        if band not in self._luts:
            raise KeyError(f"no LUT for band {band!r}; have {sorted(self._luts)}")
        lut = self._luts[band]

        def lb(height_m: NDArray[np.float64]) -> NDArray[np.float64]:
            t = self.air_temperature_at(t_s, height_m)
            return np.asarray(lut.lookup(t, quantity), dtype=np.float64)

        return lb

    def air_radiance(self, band: str, t_s: float, quantity: Quantity = "lb") -> float:
        """L_B(T_air) at the surface, from this model's own LUT: the level a horizontal ray's
        path radiance tends to, and what a fast path must use to stay bit-comparable (§7.1)."""
        return float(self._lb_of_height(band, t_s, quantity)(np.zeros(1))[0])

    def transmittance(
        self, band: str, t_s: float, distance_m: Any, elevation_rad: float = 0.0
    ) -> NDArray[np.float64]:
        return self.exponential_sum(band, t_s).transmittance(distance_m, elevation_rad)

    def path_radiance(
        self,
        band: str,
        t_s: float,
        distance_m: float,
        elevation_rad: float = 0.0,
        quantity: Quantity = "lb",
    ) -> float:
        es = self.exponential_sum(band, t_s)
        return es.path_radiance(distance_m, elevation_rad, self._lb_of_height(band, t_s, quantity))

    def class_transmittances(
        self, band: str, t_s: float, distance_m: float, elevation_rad: float = 0.0
    ) -> NDArray[np.float64]:
        """τ_k(d, θ) per spectral class (the band τ is Σ w_k τ_k)."""
        es = self.exponential_sum(band, t_s)
        return np.asarray(
            np.exp(-es.optical_depths(float(distance_m), elevation_rad)), dtype=np.float64
        )

    def sky_beyond_per_class(
        self,
        band: str,
        t_s: float,
        distance_m: float,
        elevation_rad: float,
        quantity: Quantity = "lb",
    ) -> NDArray[np.float64]:
        """Per class, the column emission beyond range R along the ray as seen *from R*:
        L_beyond,k = (L_sky,k − L_path,k(R)) / τ_k(R). A target at R occults exactly this."""
        es = self.exponential_sum(band, t_s)
        lb = self._lb_of_height(band, t_s, quantity)
        sky_k = es.path_radiance_per_class(math.inf, elevation_rad, lb)
        path_k = es.path_radiance_per_class(float(distance_m), elevation_rad, lb)
        tau_k = self.class_transmittances(band, t_s, distance_m, elevation_rad)
        with np.errstate(divide="ignore", invalid="ignore"):
            beyond = np.where(tau_k > 1e-300, (sky_k - path_k) / tau_k, 0.0)
        return np.asarray(np.maximum(beyond, 0.0), dtype=np.float64)

    def sky_beyond(
        self,
        band: str,
        t_s: float,
        distance_m: float,
        elevation_rad: float,
        quantity: Quantity = "lb",
    ) -> float:
        """The τ_k-weighted effective radiance beyond R (a target at it has zero excess)."""
        es = self.exponential_sum(band, t_s)
        tau_k = self.class_transmittances(band, t_s, distance_m, elevation_rad)
        beyond = self.sky_beyond_per_class(band, t_s, distance_m, elevation_rad, quantity)
        wt = es.weights * tau_k
        return float(np.dot(wt, beyond) / wt.sum()) if wt.sum() > 0.0 else 0.0

    def sky_radiance(
        self, band: str, t_s: float, elevation_rad: float, quantity: Quantity = "lb"
    ) -> float:
        """L_sky,B(θ) = L_path,B(∞, θ): the column's own emission (space adds nothing)."""
        return self.path_radiance(band, t_s, math.inf, elevation_rad, quantity)

    def apparent_sky_temperature_k(self, band: str, t_s: float, elevation_rad: float) -> float:
        lut = self._luts[band]
        return float(
            lut.apparent_temperature(np.asarray(self.sky_radiance(band, t_s, elevation_rad)))[()]
        )

    def apply(
        self,
        band: str,
        t_s: float,
        l_band: Any,
        distance_m: Any,
        elevation_rad: float = 0.0,
        quantity: Quantity = "lb",
    ) -> NDArray[np.floating]:
        """L' = τ L + L_path per pixel; horizontal paths use the closed form (any shape), a
        slant path with a scalar distance uses the quadrature."""
        lb = np.asarray(l_band)
        if lb.dtype == np.float16:
            raise TypeError("l_band is float16 (non-negotiable #2)")
        es = self.exponential_sum(band, t_s)
        d = np.asarray(distance_m, dtype=np.float64)
        tau = es.transmittance(d, elevation_rad)
        path: NDArray[np.float64] | float
        if math.sin(elevation_rad) <= 0.0:
            lb_air = self.air_radiance(band, t_s, quantity)
            tau_k = np.exp(-es.optical_depths(d, 0.0))
            path = np.tensordot(es.weights, 1.0 - tau_k, axes=1) * lb_air
        else:
            if d.shape != ():
                raise ValueError(
                    "slant-path apply takes a scalar distance (per-pixel slant paths: MS.8)"
                )
            path = es.path_radiance(
                float(d), elevation_rad, self._lb_of_height(band, t_s, quantity)
            )
        out = tau * lb.astype(np.float64) + path
        return np.asarray(
            out, dtype=lb.dtype if np.issubdtype(lb.dtype, np.floating) else np.float64
        )
