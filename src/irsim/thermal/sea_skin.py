"""Sea skin temperature: the cool skin and the diurnal warm layer (MM.4, ADR 0080).

A maritime scenario knows its **bulk** sea surface temperature -- that is what a buoy, a ship's
intake or a satellite SST product reports, and it is an authored scenario input here (§12.2
``ground.bulk_sst_k``). An infrared camera does not see it. It sees the **skin**, the top fraction
of a millimetre, and the skin is not the bulk:

.. math::  T_{skin} = T_{bulk} - \\Delta T_{cool}(U, Q_{net}) + \\Delta T_{warm}(Q_{sw}, U)

* the **cool skin** is always there and always cools. Heat leaves the ocean through a conductive
  sublayer that turbulence cannot penetrate, so a gradient must exist to carry it, and the skin
  sits 0.1-0.6 K below the water a millimetre beneath it whenever the surface is losing heat.
* the **diurnal warm layer** appears on sunny, calm days, when absorbed shortwave accumulates in a
  shallow surface layer faster than the wind can mix it down. It is worth up to about 3 K, and it
  is *zero* at night and zero once the wind has anything to mix with.

Both are small next to the 10-20 K sea-sky contrast that dominates a maritime frame, and neither
is negligible next to a 50 mK NETD: 0.3 K is six times the noise. The reason to model them rather
than hand-author a skin temperature is that they move with the scene's own weather -- the same
wind that roughens the surface for Cox-Munk (MM.2) sets the sublayer thickness here, and the same
solar irradiance that lights the visible frame drives the warm layer. Authoring a skin temperature
beside the bulk one would let a scene run a calm-sea skin under a 15 m/s wind.

**What this model does not include.** ``Q_net`` here is the *net longwave* the scene's own sky
model implies (M6.5). The real net heat loss also carries sensible and latent turbulent fluxes,
and at sea the latent term is usually the largest of the three; a full-flux ``Q_net`` runs roughly
twice the longwave-only value, and the deficit scales linearly with it. So the deficit computed
from the scene's longwave alone is a **lower bound** on the true one, by about a factor of two.
That is recorded rather than hidden because the function takes ``Q_net`` as an argument: a caller
that has the turbulent fluxes can pass their sum and the model is unchanged.

docs/physics-model.md §6.1, §6.5; roadmap MM.4; ADR 0080.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import (
    DENSITY_AIR_SEA_LEVEL,
    DENSITY_SEAWATER,
    EMISSIVITY_SEAWATER_BROADBAND,
    KINEMATIC_VISCOSITY_SEAWATER,
    SIGMA_SB,
    THERMAL_CONDUCTIVITY_SEAWATER,
)

__all__ = [
    "SeaSkinParams",
    "DEFAULT_SEA_SKIN",
    "friction_velocity_water_m_s",
    "cool_skin_thickness_m",
    "cool_skin_deficit_k",
    "warm_layer_k",
    "skin_temperature_k",
    "net_longwave_up_w_m2",
]


@dataclass(frozen=True)
class SeaSkinParams:
    """The parameterisation choices, separated from the thermophysical constants.

    Everything in :mod:`irsim.radiometry.constants` is a property of seawater or air. Everything
    here is a modelling decision with a defensible range and no single right value, which is why
    the two live apart: a reader can see at a glance which numbers are measurements of the world
    and which are this model's opinions.
    """

    #: Saunders' proportionality constant, sublayer thickness = lambda * nu / u*. Saunders (1967)
    #: proposed 5-10; Fairall et al. (1996) make it a function of surface buoyancy flux, running
    #: 6 in windy conditions and rising under free convection. 6 is the windy-limit value and the
    #: one most often quoted; ADR 0080 records the buoyancy-dependent form as the upgrade path.
    saunders_lambda: float = 6.0
    #: Upper bound on the conductive sublayer, blended in smoothly rather than clipped (see
    #: :func:`cool_skin_thickness_m`). Saunders' form diverges as the wind drops -- u* -> 0 gives
    #: an infinitely thick sublayer and an unbounded deficit -- while the observed cool skin stays
    #: within a few millimetres because free convection takes over the transport. 2 mm is the top
    #: of the observed range.
    max_thickness_m: float = 2.0e-3
    #: Warm-layer amplitude at full noon insolation and no wind. The observed diurnal amplitude in
    #: calm, clear, low-latitude conditions reaches 3 K; 1 K is more typical and the form below
    #: scales linearly in irradiance, so the everyday value falls out rather than being authored.
    warm_layer_max_k: float = 3.0
    #: Reference absorbed shortwave the amplitude is quoted at: clear-sky noon global horizontal.
    warm_layer_reference_w_m2: float = 1000.0
    #: Wind above which no warm layer survives. Observations put the cutoff at 5-7 m/s: beyond it
    #: wind stirring mixes the absorbed heat through a layer deep enough that the surface rise is
    #: unmeasurable.
    warm_layer_cutoff_m_s: float = 6.0


DEFAULT_SEA_SKIN = SeaSkinParams()


def friction_velocity_water_m_s(
    wind_speed_m_s: Any, drag_coefficient: float = 1.3e-3
) -> NDArray[np.float64]:
    """Water-side friction velocity u*_w = U sqrt(rho_air C_D / rho_water).

    The wind stress is continuous across the surface, so the same tau that the air exerts is what
    stirs the water: tau = rho_a C_D U^2 = rho_w u*_w^2. The neutral 10 m drag coefficient is taken
    constant at 1.3e-3, the mid-range value for 4-15 m/s; it rises toward 2e-3 in a gale and the
    deficit is already small there, so the constant costs little where it matters.
    """
    u = np.asarray(wind_speed_m_s, dtype=np.float64)
    if np.any(u < 0.0):
        raise ValueError("wind_speed_m_s cannot be negative")
    if not drag_coefficient > 0.0:
        raise ValueError("drag_coefficient must be positive")
    return np.asarray(u * math.sqrt(DENSITY_AIR_SEA_LEVEL * drag_coefficient / DENSITY_SEAWATER))


def cool_skin_thickness_m(
    wind_speed_m_s: Any, params: SeaSkinParams = DEFAULT_SEA_SKIN
) -> NDArray[np.float64]:
    """Thickness of the conductive sublayer, Saunders (1967), bounded at low wind.

    ``delta = lambda nu / u*`` is the Saunders form and it is right where the surface is
    wind-stirred: it gives about 1 mm at 5 m/s, which is what the measurements show. It also
    diverges as the wind drops, because it describes a sublayer under a shear that is vanishing,
    and free convection -- which it does not model -- is what actually limits the transport there.

    The bound is applied as ``delta_max tanh(delta_saunders / delta_max)`` rather than as a
    ``min``. The two agree to third order wherever the Saunders term is small, so the wind-stirred
    regime is untouched; what the smooth form buys is that the deficit stays **strictly**
    decreasing in wind instead of acquiring a flat shelf and a corner at the crossover. (Below
    about 0.14 m/s the tanh saturates in float64 and the result is flat -- dead calm, where this
    model claims nothing anyway.)
    """
    u_star = friction_velocity_water_m_s(wind_speed_m_s)
    with np.errstate(divide="ignore"):
        saunders = params.saunders_lambda * KINEMATIC_VISCOSITY_SEAWATER / u_star
    limit = params.max_thickness_m
    return np.asarray(limit * np.tanh(np.where(np.isfinite(saunders), saunders, np.inf) / limit))


def cool_skin_deficit_k(
    q_net_up_w_m2: Any, wind_speed_m_s: Any, params: SeaSkinParams = DEFAULT_SEA_SKIN
) -> NDArray[np.float64]:
    """dT_cool = Q_net delta / k_w, the gradient needed to conduct Q_net across the sublayer.

    ``q_net_up_w_m2`` is the net heat flux **leaving** the ocean, positive upward. Under net
    warming it is negative and the deficit is clamped to zero: the cool skin cannot invert into a
    warm one, because the mechanism is conduction against an outgoing flux. A surface that is
    genuinely warmer than the water beneath it is the *warm layer*, a different mechanism with a
    different depth scale, and it is :func:`warm_layer_k`'s job. Returning a negative "deficit"
    here would double-count it.
    """
    q = np.asarray(q_net_up_w_m2, dtype=np.float64)
    delta = cool_skin_thickness_m(wind_speed_m_s, params)
    return np.asarray(np.maximum(0.0, q) * delta / THERMAL_CONDUCTIVITY_SEAWATER)


def warm_layer_k(
    q_solar_absorbed_w_m2: Any,
    wind_speed_m_s: Any,
    params: SeaSkinParams = DEFAULT_SEA_SKIN,
) -> NDArray[np.float64]:
    """dT_warm: absorbed shortwave trapped in a shallow layer the wind has not mixed away.

    **This is an empirical form, not a derivation**, and it is labelled as such wherever it
    surfaces. It carries the three properties the observations agree on -- proportional to the
    absorbed irradiance, vanishing at night, and suppressed by wind above a threshold -- and
    nothing else:

    .. math::  \\Delta T_{warm} = \\Delta T_{max}\\,\\min(1, Q_{sw}/Q_{ref})\\,\\max(0, 1-(U/U_c)^2)

    What it does **not** carry is the warm layer's memory. The real thing integrates the day's
    heating, so it peaks in the mid-afternoon and decays for an hour or two after sunset; this
    form responds instantaneously and is exactly zero the moment the sun sets. A lagged form would
    be more faithful in the afternoon and would violate "zero at night", which is the property a
    night-time maritime scene actually depends on. ADR 0080 records the trade and the prognostic
    upgrade path.
    """
    q = np.asarray(q_solar_absorbed_w_m2, dtype=np.float64)
    if np.any(q < 0.0):
        raise ValueError("absorbed solar irradiance cannot be negative")
    u = np.asarray(wind_speed_m_s, dtype=np.float64)
    if np.any(u < 0.0):
        raise ValueError("wind_speed_m_s cannot be negative")
    if not params.warm_layer_cutoff_m_s > 0.0:
        raise ValueError("warm_layer_cutoff_m_s must be positive")
    drive = np.minimum(1.0, q / params.warm_layer_reference_w_m2)
    mixing = np.maximum(0.0, 1.0 - (u / params.warm_layer_cutoff_m_s) ** 2)
    return np.asarray(params.warm_layer_max_k * drive * mixing)


def net_longwave_up_w_m2(
    t_surface_k: Any,
    q_longwave_down_w_m2: Any,
    emissivity: float = EMISSIVITY_SEAWATER_BROADBAND,
) -> NDArray[np.float64]:
    """Net longwave leaving the surface: eps sigma T^4 - eps Q_LW_down (W m^-2, positive up).

    The absorbed share of the downwelling is ``eps Q_down``, not ``Q_down``: the rest is reflected
    and never enters the water, so it cannot drive a gradient through the sublayer. Both terms
    carry the same eps, which is why a net flux is far less sensitive to its value than either
    term alone.
    """
    t = np.asarray(t_surface_k, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("t_surface_k must be positive kelvin")
    down = np.asarray(q_longwave_down_w_m2, dtype=np.float64)
    if not 0.0 < emissivity <= 1.0:
        raise ValueError("emissivity must lie in (0, 1]")
    return np.asarray(emissivity * (SIGMA_SB * t**4 - down))


def skin_temperature_k(
    bulk_sst_k: Any,
    q_net_up_w_m2: Any,
    q_solar_absorbed_w_m2: Any,
    wind_speed_m_s: Any,
    params: SeaSkinParams = DEFAULT_SEA_SKIN,
) -> NDArray[np.float64]:
    """T_skin = T_bulk - dT_cool + dT_warm, the temperature the camera actually sees.

    The two corrections are computed from the same wind, so they cannot disagree about the
    weather, and they oppose each other: a calm sunny afternoon has both a warm layer and a thin
    conductive sublayer, and a windy night has neither worth speaking of.
    """
    bulk = np.asarray(bulk_sst_k, dtype=np.float64)
    if np.any(bulk <= 0.0):
        raise ValueError("bulk_sst_k must be positive kelvin")
    cool = cool_skin_deficit_k(q_net_up_w_m2, wind_speed_m_s, params)
    warm = warm_layer_k(q_solar_absorbed_w_m2, wind_speed_m_s, params)
    return np.asarray(bulk - cool + warm)
