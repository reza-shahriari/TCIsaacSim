"""The cold shield: how much warm structure a cooled pixel sees, and what it costs (§9.1).

§9.1 states the effect and gives no formula: "A cooled system's cold stop limits the detector's
view of warm surroundings. Effective f-number for background flux is set by the cold shield, not
the lens: if they are mismatched ('cold shield inefficiency'), background flux rises and NETD
degrades. Expose a ``cold_shield_efficiency ∈ [0, 1]`` parameter; it is a real trade-study knob."
This module supplies the formula (spec issue S31, ADR 0066).

The geometry. A pixel behind a perfectly matched cold stop sees the scene through the lens cone,
projected solid angle Ω_lens = π/(4F² + 1), and **cold metal everywhere else** -- a cold shield at
the detector's own temperature radiates nothing in band, which is the entire reason a dewar has
one. A mismatched shield opens a wider cone than the lens fills, and the difference is filled by
the warm structure inside the dewar and around the lens barrel. So

    Ω_admit = min(π, Ω_lens / η_cs)          (π: a pixel cannot see more than its own hemisphere)
    Φ_bg    = A_d (Ω_admit − Ω_lens) L_B(T_surround)

with η_cs = 1 giving **exactly** zero and η_cs → 0 opening the full hemisphere.

Two things this term is **not**:

* It is not the optics' self-emission (§8.2, ``irsim.optics.self_emission``). That is
  A_d Ω_lens (1 − τ) L_B(T_housing) -- the lens glowing *inside* the scene cone. This is the warm
  structure *outside* it. They add; neither contains the other.
* It does not pass through the optics, so τ_opt does **not** multiply it. The whole point is that
  this flux reaches the pixel without going through the lens.

What it costs. The background is a DC offset, and NUC removes offsets -- so the damage is not the
level, it is the **shot noise** the level carries. In the shot-limited case

    NETD(η_cs) / NETD(1) = √((N_signal + N_bg) / N_signal)

which is the analytic factor `netd_degradation_factor` returns and the one a trade study wants.

docs/physics-model.md §9.1, §8.1, §2; ADR 0066, spec issue S31
"""

from __future__ import annotations

import math

from irsim.detector.params import PhotonParams
from irsim.optics.aperture import aperture_factor

__all__ = [
    "HEMISPHERE_PROJECTED_SR",
    "admitted_solid_angle",
    "excess_solid_angle",
    "background_power",
    "background_electrons",
    "netd_degradation_factor",
]

#: Projected solid angle of a full hemisphere, ∫cos θ dΩ over 2π sr. The ceiling on Ω_admit: a
#: pixel on a plane cannot receive from behind itself however badly the shield is matched.
HEMISPHERE_PROJECTED_SR = math.pi


def _check_efficiency(cold_shield_efficiency: float) -> float:
    eta = float(cold_shield_efficiency)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"cold_shield_efficiency must lie in [0, 1], got {eta}")
    return eta


def admitted_solid_angle(f_number: float, cold_shield_efficiency: float) -> float:
    """Ω_admit = min(π, Ω_lens / η_cs) in sr; π at η_cs = 0, Ω_lens at η_cs = 1."""
    eta = _check_efficiency(cold_shield_efficiency)
    omega_lens = aperture_factor(f_number)
    if eta == 0.0:
        return HEMISPHERE_PROJECTED_SR
    return min(HEMISPHERE_PROJECTED_SR, omega_lens / eta)


def excess_solid_angle(f_number: float, cold_shield_efficiency: float) -> float:
    """Ω_admit − Ω_lens: the part of the pixel's cone filled by warm structure. ≥ 0 always."""
    return max(
        0.0, admitted_solid_angle(f_number, cold_shield_efficiency) - aperture_factor(f_number)
    )


def background_power(
    lb_surround: float,
    f_number: float,
    cold_shield_efficiency: float,
    active_area_m2: float,
) -> float:
    """Φ_bg = A_d (Ω_admit − Ω_lens) L_B(T_surround). Units follow ``lb_surround``.

    Exactly 0.0 at η_cs = 1 -- not 1e-17, which would still put a spurious Poisson term into the
    noise budget of every perfectly-shielded camera.
    """
    if not active_area_m2 > 0.0:
        raise ValueError(f"active_area_m2 must be positive, got {active_area_m2}")
    if lb_surround < 0.0:
        raise ValueError("surround radiance cannot be negative")
    return active_area_m2 * excess_solid_angle(f_number, cold_shield_efficiency) * lb_surround


def background_electrons(
    lb_q_surround: float,
    params: PhotonParams,
    f_number: float,
    cold_shield_efficiency: float,
) -> float:
    """N_bg = η · t_int · Φ_bg, the mean Poisson count the shield mismatch adds per integration.

    η is applied here for the same reason it is in :func:`irsim.detector.photon.photoelectrons`:
    once, at the detector, because R(λ) is a peak-1 shape (ADR 0009).
    """
    phi = background_power(lb_q_surround, f_number, cold_shield_efficiency, params.active_area_m2)
    return params.quantum_efficiency * params.integration_time_s * phi


def netd_degradation_factor(signal_electrons: float, background_electrons_: float) -> float:
    """√((N_signal + N_bg) / N_signal): what the mismatch costs a shot-limited camera.

    1.0 when there is no background, and rising without bound as the shield opens. Read noise and
    dark current are deliberately absent: this is the *shot* term's ratio, which is the part the
    cold shield controls, and mixing the others in would hide how much of the degradation the
    shield is responsible for.
    """
    if signal_electrons <= 0.0:
        raise ValueError("signal_electrons must be positive")
    if background_electrons_ < 0.0:
        raise ValueError("background_electrons cannot be negative")
    return math.sqrt((signal_electrons + background_electrons_) / signal_electrons)
