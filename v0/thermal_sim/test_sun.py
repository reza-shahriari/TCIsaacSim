import os
import sys
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.usd
from PIL import Image
import numpy as np

# Add project root to sys.path to resolve internal modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from thermal_utils.materials import make_thermal_material
from thermal_sensors.camera import MultiSpectralCamera
from pxr import UsdGeom, Gf

def main():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    with rep.new_layer():
        camera = rep.create.camera(position=(0, -8, 2), look_at=(0, 0, 1))
        render_product = rep.create.render_product(camera, (1024, 1024))
        ms_camera = MultiSpectralCamera(render_product)

        # Thermal Sun
        sun = rep.create.light(light_type="distant", intensity=10000.0, color=(1.0, 1.0, 1.0))
        
        # Object
        texture_board = rep.create.cube(position=(0, 0, 0.5), scale=2.0)
        with texture_board:
            rep.randomizer.materials([make_thermal_material(
                base_temp_k=300.0,
                emissivity_map_path="dummy_emissivity_map.png"
            )])

        rep.settings.set_render_pathtraced(64)
        
        print("Warming up render...")
        for _ in range(30):
            rep.orchestrator.step()
            
        sun_path = sun.get_node_targets()[0]
        sun_prim = stage.GetPrimAtPath(sun_path)
        xform = UsdGeom.Xformable(sun_prim)

        angles = [0, 90, 180]
        for angle in angles:
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3d(angle, 0, 0))
            
            for _ in range(20):
                rep.orchestrator.step()
                
            data = ms_camera.get_data()
            lwir = ms_camera.process_lwir(data)
            if lwir is not None:
                Image.fromarray(lwir, mode="L").save(f"test_sun_{angle}.png")
                print(f"Mean pixel value at {angle}: {np.mean(lwir)}")

    simulation_app.close()

if __name__ == "__main__":
    main()
