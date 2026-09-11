"""ISP: NUC, AGC, DDE, palette, and the radiometric branch. docs/physics-model.md §11."""

from irsim.isp.agc import agc_linear, agc_plateau
from irsim.isp.dde import dde
from irsim.isp.palette import PALETTES, palette_table, quantise_display, to_display8
from irsim.isp.radiometric import (
    RadiometricCalibration,
    apparent_temperature,
    apparent_temperature_from_dn,
)

__all__ = [
    "RadiometricCalibration",
    "apparent_temperature",
    "apparent_temperature_from_dn",
    "agc_linear",
    "agc_plateau",
    "dde",
    "PALETTES",
    "palette_table",
    "quantise_display",
    "to_display8",
]
