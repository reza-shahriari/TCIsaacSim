"""Detector models: FPA parameters, photon and bolometer transfer, NETD.

docs/physics-model.md §9
"""

from irsim.detector.bolometer import (
    BolometerTransfer,
    absorbed_power_w,
    membrane_delta_t_k,
    static_responsivity_v_per_w,
)
from irsim.detector.params import BolometerParams, FpaParams, PhotonParams, fpa_params_from_config
from irsim.detector.photon import electrons_to_dn, electrons_to_signal_dn, photoelectrons
from irsim.detector.quantise import dn_max_for_bits, quantise

__all__ = [
    "BolometerParams",
    "FpaParams",
    "PhotonParams",
    "fpa_params_from_config",
    "BolometerTransfer",
    "absorbed_power_w",
    "membrane_delta_t_k",
    "static_responsivity_v_per_w",
    "quantise",
    "dn_max_for_bits",
    "photoelectrons",
    "electrons_to_signal_dn",
    "electrons_to_dn",
]
