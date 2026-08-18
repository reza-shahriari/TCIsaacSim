import os
import sys

# Must initialize SimulationApp before importing Omniverse modules
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.usd
from PIL import Image

# Add project root to sys.path to resolve internal modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from thermal_utils.materials import make_thermal_material
from thermal_sensors.camera import MultiSpectralCamera

def main():
    omni.usd.get_context().new_stage()

    with rep.new_layer():
        # Setup Camera
        camera = rep.create.camera(position=(0, -8, 2), look_at=(0, 0, 1))
        render_product = rep.create.render_product(camera, (1024, 1024))
        
        # We don't need a physical light anymore for thermal heating!
        # The custom MDL shader handles heating via its `sun_direction` parameter.
        dome_light = rep.create.light(light_type="dome", intensity=1000.0)
        
        # Initialize the Multi-Spectral Camera Framework
        ms_camera = MultiSpectralCamera(render_product)

        # =========================================================
        # SCENE OBJECTS & PHYSICAL ASSIGNMENT (TEMP + EMISSIVITY)
        # =========================================================

        # Ground Plane
        ground = rep.create.plane(scale=50.0, position=(0, 0, -0.5))
        with ground:
            rep.randomizer.materials([make_thermal_material(base_temp_k=285.0)])

        # Background Wall
        bg_wall = rep.create.cube(position=(0, 10, 5), scale=(50.0, 1.0, 20.0))
        with bg_wall:
            rep.randomizer.materials([make_thermal_material(base_temp_k=280.0)])

        # Building Structure
        building = rep.create.cube(position=(0, 0, 1), scale=3.0)
        with building:
            rep.randomizer.materials([make_thermal_material(base_temp_k=295.0)])

        # Door
        door = rep.create.cube(position=(0, -1.5, 0.3), scale=(0.8, 0.1, 1.6))
        with door:
            rep.randomizer.materials([make_thermal_material(base_temp_k=300.0)])

        # Window
        window = rep.create.cube(position=(0.8, -1.5, 1.5), scale=(0.8, 0.1, 0.8))
        with window:
            rep.randomizer.materials([make_thermal_material(base_temp_k=360.0)])

        # Generator
        generator = rep.create.cube(position=(-1.5, -2.5, 0.0), scale=0.5)
        with generator:
            rep.randomizer.materials([make_thermal_material(base_temp_k=390.0)])

        # Dynamic Texture-based Object (Base 300K, Emissivity dictates solar absorption)
        texture_board = rep.create.cube(position=(1.5, -2.5, 0.5), scale=1.0)
        with texture_board:
            rep.randomizer.materials([make_thermal_material(
                base_temp_k=300.0,
                emissivity_map_path="dummy_emissivity_map.png"
            )])

        # Use Path-Traced mode for accurate emission and lighting bounces
        rep.settings.set_render_pathtraced(64)
        
        print("Warming up render...")
        for _ in range(30):
            rep.orchestrator.step()
            
        # =========================================================
        # ANIMATE SUN & TIME ON CUSTOM MDL SHADER
        # =========================================================
        
        # We will simulate time passing and the sun moving
        time_steps = [0.0, 2.0, 4.0]
        sun_directions = [(0.7, 0.0, 0.7), (0.0, 0.0, 1.0), (-0.7, 0.0, 0.7)]
        
        from pxr import UsdShade
        stage = omni.usd.get_context().get_stage()
        
        for i in range(3):
            current_time = time_steps[i]
            current_sun = sun_directions[i]
            
            # Find all our custom thermal materials and update them
            for prim in stage.Traverse():
                if prim.GetName() == "Shader" and prim.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(prim)
                    if shader.GetIdAttr().Get() == "dynamic_thermal":
                        shader.GetInput("time").Set(current_time)
                        shader.GetInput("sun_direction").Set(current_sun)
            
            # Let the path tracer accumulate the new values
            for _ in range(20):
                rep.orchestrator.step()
                
            data = ms_camera.get_data()
            
            lwir_img_data = ms_camera.process_lwir(data)
            if lwir_img_data is not None:
                Image.fromarray(lwir_img_data, mode="L").save(f"capture_dynamic_t{int(current_time)}.png")
                print(f"Saved capture_dynamic_t{int(current_time)}.png")

    simulation_app.close()

if __name__ == "__main__":
    main()
