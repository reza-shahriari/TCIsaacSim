"""Detector models: FPA parameters, photon and bolometer transfer, NETD.

docs/physics-model.md §9
"""

from irsim.detector.params import BolometerParams, FpaParams, PhotonParams, fpa_params_from_config

__all__ = ["BolometerParams", "FpaParams", "PhotonParams", "fpa_params_from_config"]
