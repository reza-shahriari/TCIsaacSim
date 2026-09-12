"""ISP: NUC, AGC, DDE, palette, and the radiometric branch. docs/physics-model.md §11."""

from irsim.isp.agc import agc_linear, agc_plateau
from irsim.isp.bad_pixel import MAX_PASSES, replace_bad_pixels
from irsim.isp.dde import dde
from irsim.isp.display import DisplayOutputs, agc_none, isp_config_hash, run_display_branch
from irsim.isp.nuc import TwoPointNuc
from irsim.isp.palette import PALETTES, palette_table, quantise_display, to_display8
from irsim.isp.radiometric import (
    RadiometricCalibration,
    apparent_temperature,
    apparent_temperature_from_dn,
)

__all__ = [
    "replace_bad_pixels",
    "MAX_PASSES",
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
    "DisplayOutputs",
    "agc_none",
    "isp_config_hash",
    "run_display_branch",
    "TwoPointNuc",
]
