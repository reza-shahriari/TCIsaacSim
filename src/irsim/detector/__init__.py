"""Detector models: FPA parameters, photon and bolometer transfer, NETD.

docs/physics-model.md §9
"""

from irsim.detector.anchor import anchor_noise
from irsim.detector.bolometer import (
    BolometerTransfer,
    MicrobolometerDetector,
    absorbed_power_w,
    membrane_delta_t_k,
    static_responsivity_v_per_w,
)
from irsim.detector.dark_current import dark_current_a, dark_electrons
from irsim.detector.fpa_thermal import FpaTempMode, FpaThermalModel, gain_of_t, offset_of_t
from irsim.detector.lowpass import (
    BolometerLowPass,
    alpha_for,
    responsivity_rolloff,
    trailing_decay_length_px,
)
from irsim.detector.netd import (
    NoiseBudget,
    bolometer_floors,
    predict_netd_k,
    shot_variance,
    sigma_total,
    signal_derivative_per_k,
)
from irsim.detector.params import BolometerParams, FpaParams, PhotonParams, fpa_params_from_config
from irsim.detector.photon import (
    PhotonDetector,
    electrons_to_dn,
    electrons_to_signal_dn,
    photoelectrons,
)
from irsim.detector.quantise import dn_max_for_bits, quantise
from irsim.detector.response import Detector, DetectorFrame

__all__ = [
    "BolometerParams",
    "FpaParams",
    "PhotonParams",
    "fpa_params_from_config",
    "BolometerTransfer",
    "BolometerLowPass",
    "FpaTempMode",
    "FpaThermalModel",
    "gain_of_t",
    "offset_of_t",
    "alpha_for",
    "responsivity_rolloff",
    "trailing_decay_length_px",
    "absorbed_power_w",
    "membrane_delta_t_k",
    "static_responsivity_v_per_w",
    "quantise",
    "dn_max_for_bits",
    "photoelectrons",
    "electrons_to_signal_dn",
    "electrons_to_dn",
    "NoiseBudget",
    "anchor_noise",
    "bolometer_floors",
    "predict_netd_k",
    "shot_variance",
    "sigma_total",
    "signal_derivative_per_k",
    "MicrobolometerDetector",
    "PhotonDetector",
    "Detector",
    "DetectorFrame",
    "dark_current_a",
    "dark_electrons",
]
