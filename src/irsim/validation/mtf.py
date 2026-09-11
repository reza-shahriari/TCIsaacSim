"""Slant-edge MTF estimator (ISO 12233-style) for the Tier 2 MTF bench (§15, §8.3 [R37]).

Given an image containing a nearly vertical edge tilted by a few degrees: fit the edge line from
the per-row sub-pixel crossings, project every pixel onto the perpendicular distance to that
line, bin the values into an edge-spread function oversampled ``oversample`` times, differentiate
to the line-spread function, apply a Hann window centred on its peak, and take |FFT| normalised
at DC. Returns frequencies in cycles per pixel (and per mm when the pitch is given) up to the
oversampled Nyquist, so aliased energy above the native Nyquist is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SlantEdgeResult", "slant_edge_mtf"]


@dataclass(frozen=True)
class SlantEdgeResult:
    freq_cyc_per_px: NDArray[np.float64]
    mtf: NDArray[np.float64]
    edge_angle_deg: float
    esf: NDArray[np.float64]
    esf_distance_px: NDArray[np.float64]
    pitch_um: float | None = None

    @property
    def freq_cyc_per_mm(self) -> NDArray[np.float64]:
        if self.pitch_um is None:
            raise ValueError("pitch_um was not given")
        return self.freq_cyc_per_px / (self.pitch_um * 1e-3)

    def at(self, freq_cyc_per_px: float) -> float:
        return float(np.interp(freq_cyc_per_px, self.freq_cyc_per_px, self.mtf))


def slant_edge_mtf(
    image: object,
    oversample: int = 4,
    pitch_um: float | None = None,
    window_px: int = 16,
    locate_px: int = 8,
) -> SlantEdgeResult:
    """Estimate the MTF from a slanted vertical edge (bright and dark sides in either order).
    ``window_px`` is the half-width of the Hann window around the edge in native pixels;
    ``locate_px`` the half-width of the per-row edge search around the global edge column."""
    img = np.asarray(image)
    if img.dtype == np.float16:
        raise TypeError("image is float16 (non-negotiable #2)")
    x = img.astype(np.float64)
    if x.ndim != 2 or min(x.shape) < 8:
        raise ValueError("need a 2-D image of at least 8x8 pixels")
    h, w = x.shape
    # per-row sub-pixel edge position in two passes: the mean |row derivative| over all rows
    # locates the edge column globally, then each row's centroid of the squared derivative is
    # taken within +/- locate_px of it (a full-row centroid is swamped by pixel noise)
    d = np.abs(np.diff(x, axis=1))
    cols = np.arange(w - 1, dtype=np.float64) + 0.5
    # pass 1: integer edge column per row from a 3-px smoothed |derivative| (robust to noise,
    # follows the tilt); pass 2: centroid of the squared derivative within +/- locate_px of it
    smooth = d + np.roll(d, 1, axis=1) + np.roll(d, -1, axis=1)
    coarse = np.argmax(smooth, axis=1)
    offsets = np.arange(-locate_px, locate_px + 1)
    idx = np.clip(coarse[:, None] + offsets[None, :], 0, w - 2)
    d2 = np.take_along_axis(d, idx, axis=1) ** 2
    weights = d2.sum(axis=1)
    if np.any(weights <= 0.0):
        raise ValueError("a row has no edge")
    x_edge = (d2 * cols[idx]).sum(axis=1) / weights
    rows = np.arange(h, dtype=np.float64)
    slope, intercept = np.polyfit(rows, x_edge, 1)
    angle = float(np.degrees(np.arctan(slope)))
    # perpendicular distance of every pixel centre to the fitted line, in pixels
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float64), rows, indexing="xy")
    dist = (jj - (slope * ii + intercept)) * np.cos(np.arctan(slope))
    # oversampled ESF by binning
    bin_w = 1.0 / oversample
    edges = np.arange(np.floor(dist.min()), np.ceil(dist.max()) + bin_w, bin_w)
    idx = np.clip(np.digitize(dist.ravel(), edges) - 1, 0, edges.size - 2)
    counts = np.bincount(idx, minlength=edges.size - 1).astype(np.float64)
    sums = np.bincount(idx, weights=x.ravel(), minlength=edges.size - 1)
    valid = counts > 0
    esf = np.interp(np.arange(edges.size - 1), np.flatnonzero(valid), sums[valid] / counts[valid])
    centres = 0.5 * (edges[:-1] + edges[1:])
    # LSF by the ISO 12233 two-point difference; its own transfer |sinc(f bin_w)| is divided out
    lsf = np.diff(esf) / bin_w
    # Hann window of +/- window_px native pixels centred on the fitted edge (distance 0), not on
    # the LSF maximum (a noise spike would win): wide enough that the window's own spectral
    # leakage stays small, narrow enough that pixel noise across the flat parts of the ESF does
    # not flood the high-frequency estimate (that floor grows with the square root of the count)
    peak = int(np.argmin(np.abs(centres[:-1] + 0.5 * bin_w)))
    half = window_px * oversample
    lo, hi = max(0, peak - half), min(lsf.size, peak + half + 1)
    seg = lsf[lo:hi] * np.hanning(hi - lo)
    n_fft = 8192  # fixed grid so results of different images share the same frequencies
    spec = np.abs(np.fft.rfft(seg, n=n_fft))
    if spec[0] <= 0.0:
        raise ValueError("no edge contrast")
    freq = np.fft.rfftfreq(n_fft, d=bin_w)  # cycles per native pixel
    derivative_transfer = np.abs(np.sinc(freq * bin_w))
    mtf = spec / spec[0] / np.where(derivative_transfer > 1e-3, derivative_transfer, 1.0)
    keep = freq <= oversample / 2.0
    return SlantEdgeResult(
        np.asarray(freq[keep], dtype=np.float64),
        np.asarray(mtf[keep], dtype=np.float64),
        angle,
        np.asarray(esf, dtype=np.float64),
        np.asarray(centres, dtype=np.float64),
        pitch_um,
    )
