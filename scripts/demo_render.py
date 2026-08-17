"""End-to-end demo: renders a synthetic scene through the full v0 pipeline and
saves a PNG. Not part of the test suite -- a visual sanity check that the
pipeline (thermal_physics/pipeline.py) produces a believable thermal image,
and a runnable example of how to call render_frame() directly.

Deliberately reuses only thermal_physics/ (no Isaac Sim) -- this is exactly
the Phase 1 "synthetic inputs" workflow from docs/isaacsim_checklist.md.

Run with: python scripts/demo_render.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_physics.pipeline import render_frame, SensorParams
from thermal_physics.dynamics.lumped_capacitance import LumpedThermalNode
from thermal_physics.dynamics.ice import EnthalpyThermalNode, WATER_LATENT_HEAT_FUSION_J_PER_KG


def build_scene(h: int, w: int):
    """A ground plane with: sunlit vs. shaded regions, a warm-engine hot spot,
    a cold ice patch, and a range gradient toward the horizon -- i.e. point-wise
    temperature variation on one "ground" mesh, not one value per object (SS0)."""
    y, x = np.mgrid[0:h, 0:w]

    # base ground: shaded (left) to sunlit (right), 285-305 K
    temperature_k = 285.0 + 20.0 * (x / w)

    # range increases toward the top of frame (the "horizon")
    range_m = 5.0 + 500.0 * (y / h) ** 1.5

    emissivity = np.full((h, w), 0.95)

    # a warm engine block, partway through Phase 3's warm-up curve
    engine = LumpedThermalNode(thermal_capacitance=800.0, thermal_conductance=8.0,
                                ambient_temp_k=290.0)
    for _ in range(400):  # a few minutes of simulated warm-up at dt=0.5s
        engine.step(dt_s=0.5, heat_input_w=600.0)
    ey, ex, radius = int(h * 0.6), int(w * 0.25), min(h, w) // 10
    engine_mask = (y - ey) ** 2 + (x - ex) ** 2 < radius**2
    temperature_k[engine_mask] = engine.temperature_k
    emissivity[engine_mask] = 0.3  # bare/polished metal -- low emissivity

    # a melting ice patch, partway through its plateau (SS4)
    ice = EnthalpyThermalNode(mass_kg=2.0, c_solid=2100.0, c_liquid=4186.0,
                               latent_heat_j_per_kg=WATER_LATENT_HEAT_FUSION_J_PER_KG,
                               t_melt_k=273.15, initial_temp_k=273.15)
    ice_t_k, _melt_fraction = ice.step(dt_s=1.0, heat_input_w=40.0)
    iy, ix = int(h * 0.35), int(w * 0.7)
    ice_mask = (y - iy) ** 2 + (x - ix) ** 2 < radius**2
    temperature_k[ice_mask] = ice_t_k
    emissivity[ice_mask] = 0.97

    return temperature_k, emissivity, range_m, engine.temperature_k, ice_t_k


def main():
    h, w = 240, 320
    temperature_k, emissivity, range_m, engine_t, ice_t = build_scene(h, w)

    params = SensorParams(seed=7)
    img = render_frame(temperature_k, emissivity, range_m, params)

    out_dir = Path(__file__).resolve().parent.parent / "captures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demo_frame.png"
    plt.imsave(out_path, img)

    print(f"engine block temperature after warm-up: {engine_t:.1f} K")
    print(f"ice patch temperature (mid-melt plateau): {ice_t:.2f} K")
    print(f"scene temperature range: {temperature_k.min():.1f}-{temperature_k.max():.1f} K")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
