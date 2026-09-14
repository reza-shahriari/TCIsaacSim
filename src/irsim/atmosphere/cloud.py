"""Cloud clutter: LCL base from the shared weather, ε = 1 − τ, 1/f^β structure (MS.3, ADR 0070).

No cloud section exists in the spec; §5.3(a) only says T_sky → T_air under overcast. This model:

* **Base height** from the lifting condensation level of the same WeatherSeries the atmosphere
  and the thermal solver read: z_LCL ≈ 125 m K⁻¹ (T_air − T_dew) (Espy's rule; Lawrence 2005,
  BAMS 86:225, accurate to ~2 % for RH > 50 %), clamped to the preset's bounds; saturated air
  (RH = 1) puts the base at the surface, so a thick overcast reads T_air (§5.3 a).
* **Base temperature** by the environmental lapse rate of the atmosphere preset,
  T_base = T_air − Γ z_LCL (the cloud is in equilibrium with its surroundings).
* **Radiance** of a cloud pixel: ε_cloud L_B(T_base) + τ_cloud L_clear(θ), with τ_cloud authored
  in the environment preset and ε_cloud = 1 − τ_cloud derived (thick → ε = 1; the clear column
  emission stays behind thin cloud). Per band through the LUT.
* **Spatial structure** from a seeded Gaussian 1/f^β field (power spectral density ∝ f^{−β}) on
  the image plane, thresholded at the (1 − c) quantile so exactly the weather's cloud fraction c
  is covered. The generator is a fixture for clutter statistics, not a cloud simulation; its PSD
  slope is verified by a self-test and the ME.5 display-domain bands are the only physical bound
  (deferred until the evaluation lane lands them).

docs/physics-model.md §5.3(a); ADR 0070
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.humidity import dew_point_k

__all__ = [
    "ESPY_M_PER_K",
    "SkyFixedCloud",
    "generate_sky_cloud",
    "lifting_condensation_level_m",
    "cloud_base_temperature_k",
    "CloudField",
    "generate_cloud_field",
    "psd_slope",
    "cloud_radiance",
]

ESPY_M_PER_K = 125.0  # z_LCL / (T - T_dew), Espy's rule (Lawrence 2005)


def lifting_condensation_level_m(
    t_air_k: float, rh_fraction: float, min_m: float = 0.0, max_m: float = 8000.0
) -> float:
    """z_LCL = 125 (T − T_d) m, clamped to [min_m, max_m]; RH = 1 gives 0 (base at the surface)."""
    if not 0.0 < rh_fraction <= 1.0:
        raise ValueError("RH must be a fraction in (0, 1]")
    z = ESPY_M_PER_K * (t_air_k - dew_point_k(t_air_k, rh_fraction))
    return float(min(max(z, min_m), max_m))


def cloud_base_temperature_k(t_air_k: float, base_m: float, lapse_rate_k_per_m: float) -> float:
    """T_base = T_air − Γ_env z_base."""
    if base_m < 0.0 or lapse_rate_k_per_m < 0.0:
        raise ValueError("base height and lapse rate must be non-negative")
    return float(t_air_k - lapse_rate_k_per_m * base_m)


@dataclass(frozen=True)
class CloudField:
    """A unit-variance 1/f^β field and the coverage mask at the requested fraction."""

    field: NDArray[np.float64]
    coverage: NDArray[np.bool_]
    beta: float
    fraction: float
    seed: int


def generate_cloud_field(
    shape: tuple[int, int], beta: float, cloud_fraction: float, seed: int
) -> CloudField:
    """Seeded Gaussian field with PSD ∝ f^{−β} (FFT synthesis), thresholded at the (1 − c)
    quantile so exactly round(c N) pixels are covered. β = 0 is white noise."""
    h, w = int(shape[0]), int(shape[1])
    if h < 2 or w < 2:
        raise ValueError("cloud field needs at least 2x2 pixels")
    if not 0.0 <= cloud_fraction <= 1.0:
        raise ValueError("cloud_fraction must lie in [0, 1]")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    amplitude = np.zeros_like(f)
    nonzero = f > 0.0
    amplitude[nonzero] = f[nonzero] ** (-beta / 2.0)
    spectrum = np.fft.fft2(white) * amplitude
    field = np.real(np.fft.ifft2(spectrum))
    field = (field - field.mean()) / (field.std() if field.std() > 0 else 1.0)
    n_cover = int(round(cloud_fraction * h * w))
    coverage = np.zeros((h, w), dtype=bool)
    if n_cover >= h * w:
        coverage[:] = True
    elif n_cover > 0:
        order = np.argsort(field, axis=None)[::-1][:n_cover]
        coverage.flat[order] = True
    return CloudField(
        field=field,
        coverage=coverage,
        beta=float(beta),
        fraction=float(cloud_fraction),
        seed=int(seed),
    )


def psd_slope(field: NDArray[np.floating], f_min: float = 0.02, f_max: float = 0.3) -> float:
    """Slope −β of the radially averaged power spectral density in log–log over [f_min, f_max]
    cycles per pixel (the generator's self-test; also usable on a real sky/cloud patch)."""
    x = np.asarray(field, dtype=np.float64)
    x = x - x.mean()
    power = np.abs(np.fft.fft2(x)) ** 2
    fy = np.fft.fftfreq(x.shape[0])[:, None]
    fx = np.fft.fftfreq(x.shape[1])[None, :]
    f = np.sqrt(fx * fx + fy * fy).ravel()
    p = power.ravel()
    sel = (f >= f_min) & (f <= f_max)
    bins = np.logspace(np.log10(f_min), np.log10(f_max), 16)
    idx = np.digitize(f[sel], bins)
    centres = []
    means = []
    for b in range(1, bins.size):
        m = idx == b
        if m.sum() >= 4:
            centres.append(np.sqrt(bins[b - 1] * bins[b]))
            means.append(p[sel][m].mean())
    slope, _ = np.polyfit(np.log(centres), np.log(means), 1)
    return float(slope)


def cloud_radiance(
    l_clear: Any, l_base: float, tau_cloud: float, coverage: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Per pixel: covered → ε L_B(T_base) + τ L_clear (ε = 1 − τ), else L_clear."""
    if not 0.0 <= tau_cloud < 1.0:
        raise ValueError("tau_cloud must lie in [0, 1)")
    clear = np.asarray(l_clear, dtype=np.float64)
    cov = np.asarray(coverage, dtype=bool)
    if cov.shape != clear.shape:
        raise ValueError("coverage mask and clear radiance must have the same shape")
    cloudy = (1.0 - tau_cloud) * l_base + tau_cloud * clear
    return np.asarray(np.where(cov, cloudy, clear), dtype=np.float64)


@dataclass(frozen=True)
class SkyFixedCloud:
    """Cloud coverage attached to the **sky**, sampled per ray, not painted on the image plane.

    Which frame the field lives in is the whole of the physics here, and only one of the three
    obvious choices behaves:

    * A field regenerated per frame **flickers** -- every frame is a different sky.
    * A field fixed to the **image plane** is stable, and moves with the sensor: a slewing mount
      carries its clouds along with it, so a tracked target never passes in front of one and the
      background never changes. Exactly backwards.
    * A field fixed to the **sky** (this one) is stable *and* stationary in the world, so slewing
      the mount sweeps the camera across it and a target crosses cloud edges. That is the geometry
      that makes cloud a clutter source rather than a texture.

    The grid is equirectangular in (elevation, azimuth), which stretches structure azimuthally as
    the zenith is approached -- an ``n_azimuth``-wide row spans 360 degrees at every elevation.
    For a sky-target sensor working at low to moderate elevation the distortion is small; looking
    near the zenith it is not, and a proper treatment would generate on the sphere.

    The field does not move. Wind advection is a rotation of the azimuth axis over time and is not
    modelled: over the seconds a flypast lasts, cloud drift is far below a pixel.
    """

    coverage: NDArray[np.bool_]
    beta: float
    fraction: float
    seed: int

    def sample(self, elevation_rad: Any, azimuth_rad: Any) -> NDArray[np.bool_]:
        """Coverage along each ray. Elevation is clamped to the hemisphere; azimuth wraps."""
        n_el, n_az = self.coverage.shape
        el = np.asarray(elevation_rad, dtype=np.float64)
        az = np.asarray(azimuth_rad, dtype=np.float64)
        if el.shape != az.shape:
            raise ValueError(f"elevation {el.shape} and azimuth {az.shape} must match")
        row = np.clip((el / (0.5 * math.pi) * n_el).astype(np.int64), 0, n_el - 1)
        col = np.mod((az / (2.0 * math.pi) * n_az).astype(np.int64), n_az)
        return np.asarray(self.coverage[row, col])


def generate_sky_cloud(
    beta: float,
    cloud_fraction: float,
    seed: int,
    *,
    n_elevation: int = 180,
    n_azimuth: int = 720,
) -> SkyFixedCloud:
    """A :class:`SkyFixedCloud` over the whole visible hemisphere at half-degree resolution.

    The coverage fraction is exact over the *grid*, which is the sky, not over any one frame --
    a camera pointed at a gap sees no cloud and one pointed at a bank sees only cloud, which is
    what a real sensor does and what makes cloud a false-alarm source worth simulating.
    """
    field = generate_cloud_field((n_elevation, n_azimuth), beta, cloud_fraction, seed)
    return SkyFixedCloud(
        coverage=field.coverage, beta=field.beta, fraction=field.fraction, seed=field.seed
    )
