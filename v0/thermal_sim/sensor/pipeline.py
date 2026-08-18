"""
Full radiometric pipeline for thermal IR simulation.
Implements all stages from thermal_camera_model.md §1-§11.

Each method corresponds to one numbered section so output images
can be compared directly against the document.
"""

import numpy as np
import scipy.ndimage
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# §11 – Palettes
# ---------------------------------------------------------------------------
def _ironbow_lut() -> np.ndarray:
    """Build a 256×3 uint8 Ironbow-style false-colour LUT."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Cold (dark blue) → warm (orange) → hot (white)
        r = np.clip(int(255 * min(1.0, max(0.0, (t - 0.5) * 2.5))), 0, 255)
        g = np.clip(int(255 * min(1.0, max(0.0, t * 2.0 - 0.2))), 0, 255)
        b = np.clip(int(255 * min(1.0, max(0.0, 1.0 - abs(t - 0.2) * 3.0))), 0, 255)
        lut[i] = [r, g, b]
    return lut


_IRONBOW_LUT = _ironbow_lut()


class RadiometricPipeline:
    """
    Full point-wise radiometric IR camera pipeline.
    Implements thermal_camera_model.md §1–§11.
    """

    def __init__(self, band_min_um: float = 8.0, band_max_um: float = 14.0):
        # Physical constants
        self.h   = 6.626e-34   # J·s
        self.c   = 2.998e8     # m/s
        self.k_B = 1.381e-23   # J/K

        self.band_min = band_min_um * 1e-6
        self.band_max = band_max_um * 1e-6

        # §3 Atmospheric defaults (clear, dry LWIR)
        self.gamma_eff = 0.1 / 1000.0   # km⁻¹ → m⁻¹
        self.t_atm     = 290.0           # K

        # §4 Optics defaults (fast Ge lens)
        self.tau_optics = 0.85
        self.f_number   = 1.0
        self.pixel_pitch_m = 17e-6       # 17 µm (typical LWIR bolometer)

        # §5 Detector (uncooled bolometer)
        self.k_resp  = 1.0               # responsivity constant (arbitrary units)
        self.offset0 = 0.0

        # §7 FPN — persistent across frames, reset on NUC
        self._fpn_gain   : Optional[np.ndarray] = None
        self._fpn_offset : Optional[np.ndarray] = None
        self._bad_pixels : Optional[np.ndarray] = None

        # §8 1/f noise state (AR-1 random walk per frame)
        self._row_noise_prev : Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # §1 – Surface Emission (Planck's Law)
    # -----------------------------------------------------------------------
    def planck_radiance(self, temp_k: np.ndarray, wavelength_m: float) -> np.ndarray:
        """Spectral radiance L_bb(λ,T) in W·m⁻²·sr⁻¹·m⁻¹."""
        temp_k = np.clip(temp_k, 1e-3, None)
        term1  = (2.0 * self.h * self.c**2) / (wavelength_m**5)
        expt   = np.exp((self.h * self.c) / (wavelength_m * self.k_B * temp_k)) - 1.0
        expt   = np.where(expt == 0, 1e-300, expt)
        return term1 / expt

    def band_integrated_radiance(self, temp_k: np.ndarray, samples: int = 8) -> np.ndarray:
        """
        §1: ∫ L_bb(λ,T) dλ over the sensor band.
        Uses midpoint rule with `samples` quadrature points.
        """
        wavelengths = np.linspace(self.band_min, self.band_max, samples)
        dw = (self.band_max - self.band_min) / samples
        radiance = np.zeros_like(temp_k, dtype=np.float64)
        for wl in wavelengths:
            radiance += self.planck_radiance(temp_k, wl) * dw
        return radiance.astype(np.float32)

    # -----------------------------------------------------------------------
    # §3 – Atmospheric Propagation (Beer-Lambert)
    # -----------------------------------------------------------------------
    def apply_atmosphere(
        self,
        l_point: np.ndarray,
        distance_map: np.ndarray
    ) -> np.ndarray:
        """
        §3: L_received = τ(R)·L_point + (1−τ(R))·L_bb(T_atm)
        τ(R) = exp(−γ·R)
        """
        tau_r         = np.exp(-self.gamma_eff * distance_map).astype(np.float32)
        l_path        = self.band_integrated_radiance(
            np.full_like(l_point, self.t_atm))
        return (tau_r * l_point + (1.0 - tau_r) * l_path).astype(np.float32)

    # -----------------------------------------------------------------------
    # §4 – Optics: Radiance → Irradiance
    # -----------------------------------------------------------------------
    def apply_optics(
        self,
        l_received: np.ndarray,
        theta_field: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        §4: E = L · τ_optics · π/(4N²) · cos⁴(θ_field)
        If theta_field is None, a cos⁴ vignette is synthesised from pixel coords.
        """
        h, w = l_received.shape[:2]
        if theta_field is None:
            # Approximate: θ_field from pixel offset → cos(θ) ≈ f/√(f²+r²)
            # We use normalised radius as a proxy for off-axis angle
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r_norm = np.sqrt(((xx - w/2)/(w/2))**2 + ((yy - h/2)/(h/2))**2)
            cos4   = (1.0 / np.sqrt(1.0 + r_norm**2))**4
        else:
            cos4 = np.cos(theta_field)**4

        scale = self.tau_optics * (np.pi / (4.0 * self.f_number**2))
        return (l_received * scale * cos4).astype(np.float32)

    # -----------------------------------------------------------------------
    # §6 – Blur: Diffraction MTF + Depth-of-Field + Thermal Blooming
    # -----------------------------------------------------------------------
    def apply_mtf_blur(
        self,
        irradiance: np.ndarray,
        distance_map: Optional[np.ndarray] = None,
        bloom_sigma: float = 2.0
    ) -> np.ndarray:
        """
        §6 Diffraction-limited MTF and depth-of-field blur.

        - Diffraction kernel: σ_diff ≈ λ·N / (2·d_pixel)
          For LWIR peak λ≈10µm, N=1.0, d=17µm → σ≈0.3 px (very sharp)
        - DoF blur: circle of confusion = |depth – focus_depth| / f_number
          Approximated as depth-adaptive Gaussian.
        - Thermal blooming: cross-talk from saturated hot pixels.
        """
        # 1. Diffraction PSF (Gaussian approximation)
        lambda_peak = (self.band_min + self.band_max) / 2.0
        sigma_diff  = (lambda_peak * self.f_number) / (2.0 * self.pixel_pitch_m)
        blurred     = scipy.ndimage.gaussian_filter(irradiance, sigma=max(sigma_diff, 0.3))

        # 2. Depth-of-Field blur (depth-dependent)
        if distance_map is not None:
            focus_depth = float(np.median(distance_map))
            # Circle of confusion in pixels
            coc = np.abs(distance_map - focus_depth) / (focus_depth * self.f_number + 1e-6)
            coc = np.clip(coc, 0, 5.0)
            # Blend: near-focus = unblurred, far from focus = extra Gaussian
            dof_blur = scipy.ndimage.gaussian_filter(blurred, sigma=2.0)
            alpha    = np.clip(coc / 5.0, 0, 1)
            blurred  = (1.0 - alpha) * blurred + alpha * dof_blur

        # 3. Thermal blooming (cross-talk from saturating pixels)
        sat_thresh  = np.percentile(blurred, 99.0)
        bloom_mask  = blurred > sat_thresh
        if np.any(bloom_mask):
            bloom = scipy.ndimage.gaussian_filter(blurred * bloom_mask, sigma=bloom_sigma)
            blurred = blurred + 0.08 * bloom

        return blurred.astype(np.float32)

    # -----------------------------------------------------------------------
    # §7 – Fixed-Pattern Noise (FPN) + NUC
    # -----------------------------------------------------------------------
    def init_fpn(self, shape: Tuple[int, int]) -> None:
        """Initialise FPN maps (call once, or again after NUC)."""
        h, w = shape
        # Gain non-uniformity: Gaussian(μ=1, σ≈2%)
        self._fpn_gain   = np.random.normal(1.0, 0.02, (h, w)).astype(np.float32)
        # Offset FPN: per-pixel + per-column components
        self._fpn_offset = (
            np.random.normal(0, 0.015, (h, w)) +
            np.random.normal(0, 0.005, (1, w))
        ).astype(np.float32)
        # Bad pixels: random ~0.3% fraction stuck at extreme values
        bad = np.random.random((h, w)) < 0.003
        stuck_hot  = (np.random.random((h, w)) < 0.5) & bad
        stuck_cold = bad & ~stuck_hot
        self._bad_pixels = np.zeros((h, w), dtype=np.float32)
        self._bad_pixels[stuck_hot]  =  1.0   # will be scaled later
        self._bad_pixels[stuck_cold] = -1.0

    def apply_fpn(self, signal: np.ndarray, nuc_frozen: bool = False) -> np.ndarray:
        """
        §7: Apply fixed-pattern noise.
        S_measured = G(u,v)·S_raw + O(u,v)  [+ bad pixels]
        """
        if self._fpn_gain is None or self._fpn_gain.shape != signal.shape:
            self.init_fpn(signal.shape)

        if nuc_frozen:
            # NUC in progress: camera sees flat reference — return near-zero field
            return np.full_like(signal, np.mean(signal) * 0.01)

        s = signal * self._fpn_gain + self._fpn_offset * np.mean(np.abs(signal))

        # Bad pixels
        sig_range = float(np.percentile(signal, 99) - np.percentile(signal, 1)) + 1e-10
        s += self._bad_pixels * sig_range * 0.5

        return s.astype(np.float32)

    def reset_fpn(self) -> None:
        """Simulate NUC correction: re-zero FPN offsets (gain stays)."""
        if self._fpn_offset is not None:
            self._fpn_offset *= 0.05   # residual after NUC

    # -----------------------------------------------------------------------
    # §8 – Temporal + Row/Column Noise (3-D Noise Model)
    # -----------------------------------------------------------------------
    def apply_temporal_noise(
        self,
        signal: np.ndarray,
        netd_sigma: float = 0.015
    ) -> np.ndarray:
        """
        §8: 3-D noise model.
        - Temporal (spatially white): σ_t = NETD · responsivity
        - Row noise: one shared RV per row (readout electronics)
        - Column noise: one shared RV per column
        - 1/f / AR(1) row drift across frames
        """
        h, w = signal.shape
        sig_mean = float(np.mean(np.abs(signal))) + 1e-30

        # Temporal white noise
        temporal = np.random.normal(0, netd_sigma * sig_mean, (h, w))

        # Row noise (AR-1 drift — correlated frame-to-frame)
        if self._row_noise_prev is None:
            self._row_noise_prev = np.random.normal(0, netd_sigma * sig_mean * 0.5, (h, 1))
        row_noise_new = (
            0.95 * self._row_noise_prev +
            np.random.normal(0, netd_sigma * sig_mean * 0.2, (h, 1))
        )
        self._row_noise_prev = row_noise_new
        row_noise = np.tile(row_noise_new, (1, w))

        # Column noise (independent per frame)
        col_noise = np.tile(
            np.random.normal(0, netd_sigma * sig_mean * 0.3, (1, w)),
            (h, 1)
        )

        return (signal + temporal + row_noise + col_noise).astype(np.float32)

    # -----------------------------------------------------------------------
    # §9 – ADC Quantization
    # -----------------------------------------------------------------------
    def quantize(
        self,
        signal: np.ndarray,
        bits: int = 14
    ) -> np.ndarray:
        """
        §9: DN_raw = round(clamp((signal − V_min)/(V_max − V_min) · (2^bits − 1), 0, 2^bits−1))
        Returns float32 signal mapped back to original range (for downstream use).
        """
        levels = 2**bits - 1
        s_min  = float(np.percentile(signal, 0.5))
        s_max  = float(np.percentile(signal, 99.5))
        if s_max <= s_min:
            return signal
        dn = np.round(np.clip((signal - s_min) / (s_max - s_min) * levels, 0, levels))
        # Return back-converted to float for subsequent processing
        return (dn / levels * (s_max - s_min) + s_min).astype(np.float32)

    # -----------------------------------------------------------------------
    # §10 – AGC (Automatic Gain Control)
    # -----------------------------------------------------------------------
    def agc_linear_percentile(
        self,
        signal: np.ndarray,
        lo_pct: float = 1.0,
        hi_pct: float = 99.0
    ) -> np.ndarray:
        """§10 Linear percentile-clipped AGC → [0,1]."""
        lo = float(np.percentile(signal, lo_pct))
        hi = float(np.percentile(signal, hi_pct))
        if hi <= lo:
            return np.zeros_like(signal, dtype=np.float32)
        return np.clip((signal - lo) / (hi - lo), 0, 1).astype(np.float32)

    def agc_histogram_eq(self, signal: np.ndarray, bins: int = 512) -> np.ndarray:
        """§10 Histogram equalisation AGC → [0,1]."""
        flat   = signal.ravel()
        counts, edges = np.histogram(flat, bins=bins)
        cdf    = np.cumsum(counts).astype(np.float64)
        cdf   /= cdf[-1]
        centres = 0.5 * (edges[:-1] + edges[1:])
        out    = np.interp(signal.ravel(), centres, cdf).reshape(signal.shape)
        return out.astype(np.float32)

    def agc_plateau_eq(
        self,
        signal: np.ndarray,
        bins: int = 512,
        clip_limit: float = 0.03
    ) -> np.ndarray:
        """
        §10 Plateau (CLAHE-style) equalisation: caps histogram bins so one
        hot spot can't dominate the whole stretch.
        """
        flat = signal.ravel()
        counts, edges = np.histogram(flat, bins=bins)
        # Clip and redistribute excess uniformly
        excess     = np.maximum(counts - int(clip_limit * flat.size), 0)
        redistrib  = excess.sum() // bins
        counts_c   = np.minimum(counts, int(clip_limit * flat.size)) + redistrib
        cdf        = np.cumsum(counts_c).astype(np.float64)
        cdf       /= cdf[-1]
        centres    = 0.5 * (edges[:-1] + edges[1:])
        out        = np.interp(signal.ravel(), centres, cdf).reshape(signal.shape)
        return out.astype(np.float32)

    def agc_log(self, signal: np.ndarray) -> np.ndarray:
        """Log-scale per-frame AGC → [0,1]  (good for static snapshots)."""
        signal = np.clip(signal, 1e-30, None)
        log_s  = np.log(signal)
        lo, hi = log_s.min(), log_s.max()
        if hi <= lo:
            return np.zeros_like(signal, dtype=np.float32)
        return np.clip((log_s - lo) / (hi - lo), 0, 1).astype(np.float32)

    # -----------------------------------------------------------------------
    # §11 – Palette Mapping
    # -----------------------------------------------------------------------
    def palette_white_hot(self, norm: np.ndarray) -> np.ndarray:
        """§11 White-Hot: cold=black, hot=white."""
        return (np.clip(norm, 0, 1) * 255).astype(np.uint8)

    def palette_black_hot(self, norm: np.ndarray) -> np.ndarray:
        """§11 Black-Hot: cold=white, hot=black."""
        return (np.clip(1.0 - norm, 0, 1) * 255).astype(np.uint8)

    def palette_ironbow(self, norm: np.ndarray) -> np.ndarray:
        """§11 Ironbow false-colour: cold=dark-blue → orange → white."""
        idx = np.clip((norm * 255).astype(np.int32), 0, 255)
        return _IRONBOW_LUT[idx]   # returns H×W×3 uint8

    # -----------------------------------------------------------------------
    # Convenience: full pipeline in one call (keeps backwards compat)
    # -----------------------------------------------------------------------
    def process_frame(
        self,
        temp_map: np.ndarray,
        distance_map: np.ndarray,
        emissivity_map: Optional[np.ndarray] = None,
        view_angle_cos_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """§1–§4: emission → atmosphere → optics → irradiance."""
        if emissivity_map is None:
            emissivity_map = np.ones_like(temp_map, dtype=np.float32) * 0.95

        # §5 Directional emissivity (Fresnel drop at grazing angles)
        if view_angle_cos_map is not None:
            fresnel = 1.0 - (1.0 - np.clip(view_angle_cos_map, 0, 1))**5
            e_eff   = emissivity_map * (0.3 + 0.7 * fresnel)
        else:
            e_eff   = emissivity_map

        # §1 Surface emission + reflected environment
        l_self  = self.band_integrated_radiance(temp_map)
        l_env   = self.band_integrated_radiance(np.full_like(temp_map, self.t_atm))
        l_point = e_eff * l_self + (1.0 - e_eff) * l_env

        # §3 Atmospheric propagation
        l_recv  = self.apply_atmosphere(l_point, distance_map)

        # §4 Optics → irradiance
        irr     = self.apply_optics(l_recv)
        return irr

    def apply_sensor_artifacts(
        self,
        irradiance: np.ndarray,
        distance_map: np.ndarray,
        nuc_frozen: bool = False
    ) -> np.ndarray:
        """§6–§8 combined: MTF blur → FPN → temporal noise."""
        s = self.apply_mtf_blur(irradiance, distance_map)
        s = self.apply_fpn(s, nuc_frozen=nuc_frozen)
        s = self.apply_temporal_noise(s)
        return np.clip(s, 0, None)

    def apply_agc_and_palette(self, signal: np.ndarray) -> np.ndarray:
        """Legacy helper: log AGC → white-hot 8-bit."""
        return self.palette_white_hot(self.agc_log(signal))
