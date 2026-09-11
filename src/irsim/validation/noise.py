"""Noise analysers: NVESD 3-D decomposition and PSD diagnostics (docs/physics-model.md §10.2,
§10.3, §15 Tier 2 / Tier 4(c)). Pure NumPy, no synthesis: the sole home of ``decompose_3d``,
``spatial_psd`` and ``temporal_psd`` (ADR 0023).

**3-D decomposition.** A cube n[t, v, h] is modelled as the random-effects sum
S + T + V + H + TV + TH + VH + TVH (§10.2). The directional-mean operators of D'Agostino & Webb
give the sums of squares of each effect; their expected mean squares contain **leakage** from
the higher-order effects divided by the averaging counts (e.g. E[MS_V] = σ_V²·N_t N_h +
σ_TV²·N_h + σ_VH²·N_t + σ_TVH²), so the naive std of a directional mean overstates small
components. The unbiased estimates solve that system:

    σ²_TVH = MS_TVH
    σ²_TV  = (MS_TV − MS_TVH) / N_h
    σ²_TH  = (MS_TH − MS_TVH) / N_v
    σ²_VH  = (MS_VH − MS_TVH) / N_t
    σ²_T   = (MS_T − MS_TV − MS_TH + MS_TVH) / (N_v N_h)   (and cyclically for V and H)

Sampling noise can make a small estimate negative; it is clipped to zero and the raw value kept
in ``raw_variances`` so the floor is visible. The standard deviation of each variance estimate is
``estimate_floors``: std(σ̂²_Y) ≈ √(2/df_Y) · Σ_{X ⊇ Y} σ_X² / Π_{a ∈ X∖Y} N_a -- the fixed row and
column terms of a 64-wide cube have only 64 samples (±18 % on the variance), and a small term
inherits the sampling noise of the larger mean squares subtracted from it. Tests and reports
state tolerances from these floors (ADR 0023).

**PSD.** ``spatial_psd`` is the frame-averaged, mean-removed 2-D periodogram normalised so that
its sum equals the spatial variance (Parseval), with the radial average and the k_v = 0 / k_h = 0
lines where column and row noise concentrate; ``temporal_psd`` is the pixel-averaged
periodogram along t. ``compare_psd`` returns the maximum ratio of two radial profiles above DC,
the Tier 4(c) "within a factor of 2" statistic.

docs/physics-model.md §10.2 [R28], §10.3, §15
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Decomposition3D",
    "decompose_3d",
    "estimate_floors",
    "SpatialPSD",
    "spatial_psd",
    "temporal_psd",
    "compare_psd",
    "RATIO_ORDER",
]

RATIO_ORDER: tuple[str, ...] = ("t", "v", "h", "tv", "th", "vh", "tvh")


@dataclass(frozen=True)
class Decomposition3D:
    """Component standard deviations in the cube's unit (order ``RATIO_ORDER``)."""

    t: float
    v: float
    h: float
    tv: float
    th: float
    vh: float
    tvh: float
    raw_variances: dict[str, float]
    mean: float

    def as_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, k)) for k in RATIO_ORDER)

    @property
    def total(self) -> float:
        """√Σσ² of the (clipped) estimates."""
        return float(np.sqrt(sum(s * s for s in self.as_vector())))

    def ratios(self) -> tuple[float, ...]:
        """Each component relative to σ_TVH (the §10.2 convention)."""
        if self.tvh <= 0.0:
            raise ValueError("sigma_TVH is zero; ratios are undefined")
        return tuple(s / self.tvh for s in self.as_vector())


def _as_cube(cube: object) -> NDArray[np.float64]:
    x = np.asarray(cube)
    if x.ndim != 3:
        raise ValueError(f"cube must be (T, V, H), got shape {x.shape}")
    if x.dtype == np.float16:
        raise TypeError("cube is float16; promote to float32 or better before analysis")
    if min(x.shape) < 2:
        raise ValueError("every axis needs at least two samples")
    return x.astype(np.float64)


def decompose_3d(cube: object) -> Decomposition3D:
    """Leakage-corrected NVESD decomposition of a (T, V, H) cube (uint16 input is promoted)."""
    n = _as_cube(cube)
    nt, nv, nh = n.shape
    s = float(n.mean())
    m_t = n.mean(axis=(1, 2))  # (T,)
    m_v = n.mean(axis=(0, 2))  # (V,)
    m_h = n.mean(axis=(0, 1))  # (H,)
    m_tv = n.mean(axis=2)  # (T, V)
    m_th = n.mean(axis=1)  # (T, H)
    m_vh = n.mean(axis=0)  # (V, H)

    ss_t = nv * nh * float(np.sum((m_t - s) ** 2))
    ss_v = nt * nh * float(np.sum((m_v - s) ** 2))
    ss_h = nt * nv * float(np.sum((m_h - s) ** 2))
    ss_tv = nh * float(np.sum((m_tv - m_t[:, None] - m_v[None, :] + s) ** 2))
    ss_th = nv * float(np.sum((m_th - m_t[:, None] - m_h[None, :] + s) ** 2))
    ss_vh = nt * float(np.sum((m_vh - m_v[:, None] - m_h[None, :] + s) ** 2))
    resid = (
        n
        - m_tv[:, :, None]
        - m_th[:, None, :]
        - m_vh[None, :, :]
        + m_t[:, None, None]
        + m_v[None, :, None]
        + m_h[None, None, :]
        - s
    )
    ss_tvh = float(np.sum(resid**2))

    ms_t = ss_t / (nt - 1)
    ms_v = ss_v / (nv - 1)
    ms_h = ss_h / (nh - 1)
    ms_tv = ss_tv / ((nt - 1) * (nv - 1))
    ms_th = ss_th / ((nt - 1) * (nh - 1))
    ms_vh = ss_vh / ((nv - 1) * (nh - 1))
    ms_tvh = ss_tvh / ((nt - 1) * (nv - 1) * (nh - 1))

    raw = {
        "tvh": ms_tvh,
        "tv": (ms_tv - ms_tvh) / nh,
        "th": (ms_th - ms_tvh) / nv,
        "vh": (ms_vh - ms_tvh) / nt,
        "t": (ms_t - ms_tv - ms_th + ms_tvh) / (nv * nh),
        "v": (ms_v - ms_tv - ms_vh + ms_tvh) / (nt * nh),
        "h": (ms_h - ms_th - ms_vh + ms_tvh) / (nt * nv),
    }
    sig = {k: float(np.sqrt(max(v, 0.0))) for k, v in raw.items()}
    return Decomposition3D(
        t=sig["t"],
        v=sig["v"],
        h=sig["h"],
        tv=sig["tv"],
        th=sig["th"],
        vh=sig["vh"],
        tvh=sig["tvh"],
        raw_variances=raw,
        mean=s,
    )


@dataclass(frozen=True)
class SpatialPSD:
    """Frame-averaged 2-D power spectrum (sum = spatial variance), radial profile and axis lines."""

    psd: NDArray[np.float64]  # (V, H), fftshifted, DC at the centre
    radial_bins: NDArray[np.float64]  # bin centres in cycles/pixel
    radial: NDArray[np.float64]  # mean power per radial bin (DC excluded)
    kv0_line: NDArray[np.float64]  # power along k_v = 0 (column-noise line), fftshifted in k_h
    kh0_line: NDArray[np.float64]  # power along k_h = 0 (row-noise line), fftshifted in k_v
    total_power: float

    def fraction_on_kv0(self) -> float:
        """Share of total power (DC excluded) sitting on the k_v = 0 line."""
        return float(self._line_power(self.kv0_line) / self.total_power)

    def fraction_on_kh0(self) -> float:
        return float(self._line_power(self.kh0_line) / self.total_power)

    @staticmethod
    def _line_power(line: NDArray[np.float64]) -> float:
        centre = line.size // 2
        return float(line.sum() - line[centre])


def spatial_psd(frames: object, n_radial_bins: int = 10) -> SpatialPSD:
    """2-D periodogram of one frame (V, H) or a cube (T, V, H), mean removed per frame and
    averaged over frames. Normalised so Σ psd = spatial variance (Parseval)."""
    x = np.asarray(frames)
    if x.dtype == np.float16:
        raise TypeError("frames are float16; promote before analysis")
    x = x.astype(np.float64)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError("frames must be (V, H) or (T, V, H)")
    nt, nv, nh = x.shape
    x = x - x.mean(axis=(1, 2), keepdims=True)
    spec = np.fft.fft2(x, axes=(1, 2))
    psd = (np.abs(spec) ** 2).mean(axis=0) / (nv * nh) ** 2  # Parseval: Σ psd = variance
    psd = np.fft.fftshift(psd)
    kv = np.fft.fftshift(np.fft.fftfreq(nv))
    kh = np.fft.fftshift(np.fft.fftfreq(nh))
    kr = np.sqrt(kv[:, None] ** 2 + kh[None, :] ** 2)
    cv, ch = nv // 2, nh // 2
    total = float(psd.sum() - psd[cv, ch])
    edges = np.linspace(0.0, 0.5, n_radial_bins + 1)
    radial = np.zeros(n_radial_bins)
    for i in range(n_radial_bins):
        mask = (kr > edges[i]) & (kr <= edges[i + 1])
        radial[i] = psd[mask].mean() if mask.any() else np.nan
    return SpatialPSD(
        psd=psd,
        radial_bins=0.5 * (edges[:-1] + edges[1:]),
        radial=radial,
        kv0_line=psd[cv, :].copy(),
        kh0_line=psd[:, ch].copy(),
        total_power=total,
    )


def temporal_psd(cube: object) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pixel-averaged one-sided periodogram along t of a (T, V, H) cube, mean removed per pixel.
    Returns (frequencies in cycles/frame, power) with Σ power ≈ temporal variance."""
    n = _as_cube(cube)
    nt = n.shape[0]
    n = n - n.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(n, axis=0)
    power = (np.abs(spec) ** 2).mean(axis=(1, 2)) / nt**2
    # one-sided: double the non-DC, non-Nyquist bins so the sum is the variance
    scale = np.full(power.shape, 2.0)
    scale[0] = 1.0
    if nt % 2 == 0:
        scale[-1] = 1.0
    return (
        np.asarray(np.fft.rfftfreq(nt), dtype=np.float64),
        np.asarray(power * scale, dtype=np.float64),
    )


def compare_psd(a: SpatialPSD, b: SpatialPSD) -> float:
    """max over radial bins of max(a/b, b/a): 1.0 for identical shapes, the Tier 4(c) statistic."""
    if a.radial.shape != b.radial.shape:
        raise ValueError("radial profiles must share the binning")
    ratio = np.maximum(a.radial / b.radial, b.radial / a.radial)
    return float(np.nanmax(ratio))


_AXES: dict[str, frozenset[str]] = {k: frozenset(k) for k in RATIO_ORDER}


def estimate_floors(shape: tuple[int, int, int], sigmas: tuple[float, ...]) -> dict[str, float]:
    """Standard deviation of each unbiased variance estimate σ̂²_Y for a cube of ``shape`` with
    true component sigmas (order ``RATIO_ORDER``): √(2/df_Y) · Σ_{X ⊇ Y} σ_X² / Π_{a ∈ X∖Y} N_a.
    Use 3× this as the tolerance on |σ̂²_Y − σ_Y²|."""
    nt, nv, nh = shape
    n = {"t": nt, "v": nv, "h": nh}
    sig2 = dict(zip(RATIO_ORDER, (float(s) ** 2 for s in sigmas), strict=True))
    out: dict[str, float] = {}
    for y in RATIO_ORDER:
        df = 1.0
        for a in _AXES[y]:
            df *= n[a] - 1
        expected_ms_over_d = 0.0
        for x in RATIO_ORDER:
            if _AXES[x] >= _AXES[y]:
                div = 1.0
                for a in _AXES[x] - _AXES[y]:
                    div *= n[a]
                expected_ms_over_d += sig2[x] / div
        out[y] = float(np.sqrt(2.0 / df) * expected_ms_over_d)
    return out
