"""A spinning rotor as a time-averaged veil: what the camera records is the disc, not the blade.

ADR 0074 left propellers off the quadrotor because a solid disc of the right diameter hides the
motors underneath it -- which are the subject -- and a blade frozen at one azimuth is a picture no
thermal camera takes. A 28-inch prop at flight rpm turns 50 times a second; a 60 Hz bolometer
integrates for the whole 16.7 ms frame, so each blade crosses every azimuth of the disc more than
once *while the pixel is charging*. The detector reports the time average, and the time average of
an intermittent opaque occluder is a **partially covering veil**:

    L = alpha L_blade + (1 - alpha) L_behind                            (in radiance, always)

``alpha`` is not a transmittance and the blade is not semi-transparent. It is the fraction of the
integration window during which blade material stood between that pixel and the background, and the
blend is linear in radiance because radiance is what the detector integrates. Blending apparent
*temperatures* would be a different number and a wrong one (§3.2's L(T) is convex, so the mean of
the radiances is not the radiance of the mean).

**One formula covers both regimes, and they are the ADR 0077 split again.** The coverage at a given
azimuth is the running mean of the blade-passage pulse train over a window of length ``T = omega
t_window`` -- the angle swept during the integration. A bolometer has no shutter, so T is a whole
frame period and the window is long compared with the blade spacing: the ripple washes out and the
picture is a smooth annulus. A cooled photon detector integrates for a couple of milliseconds, T
falls below the blade spacing, and the same expression resolves into a few smeared arcs. Nothing
selects between them; the window length does.

**The mean is conserved for every window length** -- a running mean cannot change the mean of a
periodic function -- so the disc's total flux contribution is independent of the shutter even though
its appearance is not. That is the invariant this module is tested on.

**Viewing tilt is carried by the projected area of a pitched plate, not by a fudge.** A blade is a
plate whose normal sits ``pitch`` off the disc axis, so as it spins its projected area towards the
camera varies, and the azimuthal mean of ``|d . n|`` is available in closed form
(:func:`projection_factor`). Two consequences are worth stating because they are visible:

* A **zero-pitch** blade is invisible edge-on and a coarse-pitch one is not, because the edge-on
  factor is ``(2/pi) sin(pitch)``.
* Coverage per unit image area is **exactly tilt-invariant** while ``tilt < 90 - pitch``: the
  blade's projected area and the ellipse's area shrink by the same cosine, so a banking rotor
  changes shape without changing brightness. Only within ``pitch`` degrees of edge-on does the
  annulus densify, as the same blade area is packed into a vanishing ellipse. (This module was
  first written asserting that coverage rises with tilt throughout, and the rasteriser
  contradicted it -- the peak at 0 and at 45 degrees was the same number.)

The aerial demo stage looks 75 degrees off the disc axis against 18 degrees of blade pitch, so it
sits just past that crossover and the term is doing real work, not decorating.

docs/physics-model.md §8.3, §9.2 (the within-integration average), §6.6; ADR 0074, ADR 0077
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RotorDisc",
    "swept_angle_rad",
    "projection_factor",
    "azimuthal_duty",
    "coverage_map",
    "veil_radiance",
]


@dataclass(frozen=True)
class RotorDisc:
    """A rotor by its real dimensions. Defaults are a 28-inch heavy-lift multirotor prop.

    ``chord_root_m`` / ``chord_tip_m`` taper linearly between ``root_m`` and ``radius_m``.
    ``pitch_deg`` is the angle between the blade plate's normal and the disc axis -- the geometric
    pitch angle, quoted at 75 % radius for a real prop and treated as constant here.
    """

    radius_m: float = 0.3556  # 28 inch diameter
    root_m: float = 0.045  # blades start outboard of the hub
    blades: int = 2
    chord_root_m: float = 0.038
    chord_tip_m: float = 0.022
    pitch_deg: float = 18.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if not 0.0 <= self.root_m < self.radius_m:
            raise ValueError("root_m must lie in [0, radius_m)")
        if self.blades < 1:
            raise ValueError("a rotor needs at least one blade")
        if self.chord_root_m <= 0.0 or self.chord_tip_m <= 0.0:
            raise ValueError("chords must be positive")
        if not 0.0 <= self.pitch_deg < 90.0:
            raise ValueError("pitch_deg must lie in [0, 90)")
        # The angular width of one blade at the root must leave room for the others, or the
        # blades overlap and the geometry is not a rotor.
        if self.root_m > 0.0 and self.chord_root_m / self.root_m >= 2.0 * math.pi / self.blades:
            raise ValueError("blades overlap at the root: chord too wide for this blade count")

    def chord_m(self, radius_m: Any) -> NDArray[np.float64]:
        """Blade chord at radius r, linear between root and tip; zero inboard of the cutout."""
        r = np.asarray(radius_m, dtype=np.float64)
        frac = (r - self.root_m) / (self.radius_m - self.root_m)
        chord = self.chord_root_m + frac * (self.chord_tip_m - self.chord_root_m)
        inside = (r >= self.root_m) & (r <= self.radius_m)
        return np.asarray(np.where(inside, chord, 0.0))

    def blade_area_m2(self) -> float:
        """Planform area of **one** blade, the integral of the chord over the span."""
        return 0.5 * (self.chord_root_m + self.chord_tip_m) * (self.radius_m - self.root_m)

    def local_solidity(self, radius_m: Any) -> NDArray[np.float64]:
        """Fraction of the annulus at radius r that is blade, ``N c(r) / (2 pi r)``.

        This is the coverage a face-on, zero-pitch rotor shows once the window is long enough to
        wash the ripple out. It is small: a 28-inch two-blade prop is under 5 % at three-quarter
        radius, which is why ADR 0074 predicted "a faint, low-contrast annulus" and why a solid
        disc would have been so wrong.
        """
        r = np.asarray(radius_m, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = self.blades * self.chord_m(r) / (2.0 * np.pi * r)
        return np.asarray(np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0))

    def blade_width_rad(self, radius_m: Any) -> NDArray[np.float64]:
        """Angular extent of one blade at radius r, ``c(r) / r``."""
        r = np.asarray(radius_m, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = self.chord_m(r) / r
        return np.asarray(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0))


def swept_angle_rad(rpm: float, window_s: float) -> float:
    """Angle the rotor turns through during one integration window, radians.

    This single number decides whether the picture is an annulus or a set of arcs: compare it with
    the blade spacing ``2 pi / N``. For a 3000 rpm prop it is 300 degrees in a 60 Hz bolometer
    frame (1.7 blade spacings, a smooth annulus with visible ripple) and 36 degrees in a 2 ms
    cooled-MWIR integration (a fifth of a spacing -- two distinct arcs).
    """
    if window_s < 0.0:
        raise ValueError("window_s must be non-negative")
    return 2.0 * math.pi * (abs(rpm) / 60.0) * float(window_s)


def projection_factor(tilt_rad: Any, pitch_rad: Any) -> NDArray[np.float64]:
    """Azimuthal mean of ``|d . n|`` for a plate pitched ``pitch`` off the axis, viewed at ``tilt``.

    A blade at azimuth phi has normal ``n = cos(pitch) z - sin(pitch) t(phi)``; a view direction
    ``tilt`` off the axis gives ``d . n = A + B sin(phi)`` with ``A = cos(pitch) cos(tilt)`` and
    ``B = sin(pitch) sin(tilt)``. Averaging the modulus over a turn is closed-form:

        |A| >= |B|:  k = |A|                       (the sign never changes)
        |A| <  |B|:  k = (2/pi) [sqrt(B^2 - A^2) + A asin(A/B)]

    Face-on this is ``cos(pitch)`` and edge-on ``(2/pi) sin(pitch)``, so a feathered blade vanishes
    edge-on and a pitched one does not. That limit is the whole reason pitch is a parameter here.
    """
    a = np.cos(np.asarray(pitch_rad, dtype=np.float64)) * np.cos(
        np.asarray(tilt_rad, dtype=np.float64)
    )
    b = np.sin(np.asarray(pitch_rad, dtype=np.float64)) * np.sin(
        np.asarray(tilt_rad, dtype=np.float64)
    )
    a, b = np.abs(a), np.abs(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = (2.0 / np.pi) * (
            np.sqrt(np.maximum(b * b - a * a, 0.0)) + a * np.arcsin(np.clip(a / b, -1.0, 1.0))
        )
    return np.asarray(np.where(a >= b, a, np.nan_to_num(crossing, nan=0.0)))


def _pulse_integral(x: Any, period: float, width: Any) -> NDArray[np.float64]:
    """Antiderivative of a ``period``-periodic train of unit pulses of angular ``width``.

    Pulses are centred on multiples of the period, so the train is the blade-passage indicator seen
    at one azimuth. Only differences of this are used, so the constant is free. ``width`` may be an
    array: the blade narrows in angle with radius, so every annulus has its own pulse train.
    """
    y = np.asarray(x, dtype=np.float64) + 0.5 * np.asarray(width, dtype=np.float64)
    whole, part = np.divmod(y, period)
    return np.asarray(whole * width + np.clip(part, 0.0, width))


def azimuthal_duty(
    azimuth_rad: Any, swept_rad: float, width_rad: Any, blades: int
) -> NDArray[np.float64]:
    """Fraction of the integration window that blade material covered each azimuth.

    The running mean of the blade-passage pulse train over the swept angle. Long windows flatten it
    to the local solidity; short ones leave the arcs. **The azimuthal mean is the solidity for every
    window length**, because a running mean cannot move the mean of a periodic function -- so the
    shutter changes where the disc's flux lands, never how much there is.

    The ripple that survives at intermediate windows is real, not numerical: at 1.7 blade spacings
    some azimuths are crossed twice and some once, which is the banding a real prop shows on video.

    ``width_rad`` is the blade's **planform** angular extent ``c/r``; the pitch foreshortening is
    carried entirely by :func:`projection_factor`, so the arcs here are wider than the projected
    blade by a factor ``1/cos(pitch)`` -- 5 % at 18 degrees. The mean is unaffected.
    """
    if blades < 1:
        raise ValueError("a rotor needs at least one blade")
    period = 2.0 * math.pi / blades
    w = np.clip(np.asarray(width_rad, dtype=np.float64), 0.0, period)
    u = np.asarray(azimuth_rad, dtype=np.float64)
    t = float(swept_rad)
    if t <= 0.0:
        # A shutter this short freezes the blade: the indicator itself, no averaging.
        _, part = np.divmod(u + 0.5 * w, period)
        return np.asarray((part <= w).astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        duty = (_pulse_integral(u, period, w) - _pulse_integral(u - t, period, w)) / t
    return np.asarray(np.clip(np.nan_to_num(duty, nan=0.0), 0.0, 1.0))


def coverage_map(
    disc: RotorDisc,
    shape: tuple[int, int],
    centre_px: tuple[float, float],
    semi_major_px: float,
    semi_minor_px: float,
    swept_rad: float,
    *,
    rotation_deg: float = 0.0,
    phase_rad: float = 0.0,
) -> NDArray[np.float64]:
    """Time-averaged blade coverage of each pixel, in [0, 1], over the projected disc.

    The disc projects to an ellipse: ``semi_major_px`` is the un-foreshortened radius and
    ``semi_minor_px = semi_major_px cos(tilt)``, so the **viewing tilt is read off the axes** rather
    than passed separately -- there is then no way for the drawn ellipse and the projection physics
    to disagree about it. ``rotation_deg`` turns the major axis in the image, measured from +x
    towards +y (rows down, per the G-buffer contract); ``phase_rad`` is where a blade sits at the
    *end* of the window and only matters when the arcs are resolved.

    Coverage is ``k / cos(tilt)`` times the azimuthal duty: ``k`` is the projected area of a pitched
    blade and the ``1/cos(tilt)`` is the ellipse shrinking under the same blade area. The two
    cancel exactly to ``cos(pitch)`` until ``tilt`` passes ``90 - pitch``; past that the product
    grows and is clipped at 1, because at grazing incidence the blades genuinely stack up along the
    ray and the swept annulus becomes opaque.

    Generate this on the **supersampled** grid and let the existing box downsample resolve the edge;
    a disc seen nearly edge-on is a few pixels thick and a hard in/out test on the native grid
    aliases into a dashed line.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape must be positive")
    if semi_major_px <= 0.0 or semi_minor_px < 0.0:
        raise ValueError("semi_major_px must be positive and semi_minor_px non-negative")
    if semi_minor_px > semi_major_px:
        raise ValueError("semi_minor_px is the foreshortened axis and cannot exceed semi_major_px")
    alpha = np.zeros((height, width), dtype=np.float64)
    if semi_minor_px == 0.0:
        return alpha  # exactly edge-on: the ellipse has no area to cover

    rows, cols = np.indices((height, width), dtype=np.float64)
    theta = math.radians(rotation_deg)
    dx, dy = cols + 0.5 - centre_px[0], rows + 0.5 - centre_px[1]
    # Into the ellipse's own frame, then undo the foreshortening to get disc coordinates.
    p = (dx * math.cos(theta) + dy * math.sin(theta)) / semi_major_px
    q = (-dx * math.sin(theta) + dy * math.cos(theta)) / semi_minor_px

    unit_radius = np.hypot(p, q)
    radius_m = unit_radius * disc.radius_m
    on_disc = (radius_m >= disc.root_m) & (radius_m <= disc.radius_m)
    if not bool(on_disc.any()):
        return alpha

    azimuth = np.arctan2(q, p) - phase_rad
    duty = azimuthal_duty(azimuth, swept_rad, disc.blade_width_rad(radius_m), disc.blades)
    cos_tilt = semi_minor_px / semi_major_px
    k = projection_factor(math.acos(min(1.0, cos_tilt)), math.radians(disc.pitch_deg))
    alpha = np.where(on_disc, np.clip(duty * float(k) / cos_tilt, 0.0, 1.0), 0.0)
    return np.asarray(alpha, dtype=np.float64)


def veil_radiance(
    background: Any,
    alpha: Any,
    blade_radiance: Any,
    *,
    occluded: Any = None,
) -> NDArray[np.float64]:
    """Composite the veil over the scene **in radiance** (CLAUDE.md #3's reasoning, §3.2).

    ``blade_radiance`` is the band radiance the blade would show if it filled the pixel -- already
    ``eps L_B(T) + (1 - eps) L_env`` for the blade material, which for a prop is near air
    temperature and therefore near the airframe. The lift over a cold sky is small and that is the
    point: the disc is a faint annulus, and a model that made it bright would be filming something
    else.

    ``occluded`` marks pixels where something opaque stands **in front of** the disc plane -- the
    near half of the airframe, seen from below -- and zeroes the veil there. Without it a blade
    would be painted over the motor bell it is mounted on.
    """
    bg = np.asarray(background, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    if a.shape != bg.shape:
        raise ValueError(f"alpha {a.shape} does not match background {bg.shape}")
    if np.any(a < 0.0) or np.any(a > 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    if occluded is not None:
        mask = np.asarray(occluded, dtype=bool)
        if mask.shape != bg.shape:
            raise ValueError(f"occluded {mask.shape} does not match background {bg.shape}")
        a = np.where(mask, 0.0, a)
    blade = np.asarray(blade_radiance, dtype=np.float64)
    return np.asarray(a * blade + (1.0 - a) * bg)
