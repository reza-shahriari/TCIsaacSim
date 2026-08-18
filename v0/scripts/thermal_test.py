import os
import argparse
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageFilter
import numpy as np
import carb

def main():
    omni.usd.get_context().new_stage()

    # 1. Create a gradient texture for the building (Simulates left side being warmer than right)
    # A 256x256 image with a horizontal gradient
    grad = np.linspace(255, 50, 256)
    grad_img = np.tile(grad, (256, 1)).astype(np.uint8)
    grad_path = os.path.abspath("heat_grad.png")
    Image.fromarray(grad_img).save(grad_path)

    with rep.new_layer():
        # Add a dome light for general ambient lighting
        dome_light = rep.create.light(light_type="dome", intensity=2000.0)
        distance_light = rep.create.light(light_type="distant", intensity=3000.0, rotation=(315, 0, 0))

        camera = rep.create.camera(position=(0, -8, 2), look_at=(0, 0, 1))
        render_product = rep.create.render_product(camera, (1024, 1024))

        # Create Ground Plane
        ground = rep.create.plane(scale=50.0, position=(0, 0, -0.5))
        mat_ground = rep.create.material_omnipbr(
            diffuse=(0.2, 0.2, 0.2),
            emissive_color=(1.0, 1.0, 1.0),
            emissive_intensity=2.0 # Cold ground
        )
        with ground:
            rep.randomizer.materials([mat_ground])

        # Create Background Wall
        bg_wall = rep.create.cube(position=(0, 10, 5), scale=(50.0, 1.0, 20.0))
        mat_bg = rep.create.material_omnipbr(
            diffuse=(0.1, 0.1, 0.1),
            emissive_color=(1.0, 1.0, 1.0),
            emissive_intensity=1.0 # Very cold background
        )
        with bg_wall:
            rep.randomizer.materials([mat_bg])

        # Create building (Main body)
        # Position: y spans -1.5 to 1.5, z spans -0.5 to 2.5
        building = rep.create.cube(position=(0, 0, 1), scale=3.0)
        mat_building = rep.create.material_omnipbr(
            diffuse=(0.8, 0.8, 0.8),
            emissive_color=(1.0, 1.0, 1.0),
            emissive_texture=grad_path, # Apply the gradient texture!
            emissive_intensity=25.0 # Warm building body
        )
        with building:
            rep.randomizer.materials([mat_building])

        # Create Door (sticks out slightly at front y=-1.55)
        door = rep.create.cube(position=(0, -1.55, 0.3), scale=(0.8, 0.1, 1.6))
        mat_door = rep.create.material_omnipbr(
            diffuse=(0.4, 0.2, 0.1),
            emissive_color=(1.0, 1.0, 1.0),
            emissive_intensity=5.0 # Cooler door (better insulated or outside temp)
        )
        with door:
            rep.randomizer.materials([mat_door])

        # Create Window (sticks out slightly at front y=-1.55)
        window = rep.create.cube(position=(0.8, -1.55, 1.5), scale=(0.8, 0.1, 0.8))
        mat_window = rep.create.material_omnipbr(
            diffuse=(0.1, 0.5, 0.8),
            emissive_color=(1.0, 1.0, 1.0),
            emissive_intensity=40.0 # Hot window (heat escaping)
        )
        with window:
            rep.randomizer.materials([mat_window])

        rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        rgb_annotator.attach([render_product])
        
        try:
            thermal_annotator = rep.AnnotatorRegistry.get_annotator("PtSelfIllumination")
        except Exception as e:
            print(f"PtSelfIllumination not found natively: {e}")
            thermal_annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
        
        thermal_annotator.attach([render_product])

        rep.settings.set_render_pathtraced(64)
        
        for _ in range(50):
            rep.orchestrator.step()
            
        rgb_data = rgb_annotator.get_data()
        thermal_data = thermal_annotator.get_data()

        if rgb_data is not None and rgb_data.size > 0:
            rgb_img = Image.fromarray(rgb_data)
            rgb_img.save("building_rgb.png")
            print("Saved building_rgb.png")

        if thermal_data is not None and thermal_data.size > 0:
            # We assume thermal_data is (H, W, 3) or (H, W, 4)
            if len(thermal_data.shape) == 3:
                intensity = np.mean(thermal_data[:, :, :3], axis=2)
            else:
                intensity = thermal_data
                
            print(f"Max thermal value: {np.max(intensity)}")

            # 2. Add realistic thermal sensor noise (NETD)
            noise = np.random.normal(0, np.max(intensity)*0.02, intensity.shape)
            intensity = intensity + noise
            
            # Normalize to 0-255
            intensity_norm = intensity - np.min(intensity)
            max_val = np.max(intensity_norm)
            if max_val > 0:
                intensity_norm = (intensity_norm / max_val) * 255.0
            
            intensity_uint8 = intensity_norm.astype(np.uint8)
            
            # Save as Grayscale (Black and White)
            thermal_img = Image.fromarray(intensity_uint8, mode="L")
            
            # 3. Apply Gaussian Blur to simulate Thermal Bleeding/Blooming and conduction gradients
            # This blends the sharp edges between the hot door and cold wall, creating a thermal gradient
            thermal_img = thermal_img.filter(ImageFilter.GaussianBlur(radius=4))
            
            thermal_img.save("building_thermal.png")
            print("Saved building_thermal.png")

    simulation_app.close()

if __name__ == "__main__":
    main()
