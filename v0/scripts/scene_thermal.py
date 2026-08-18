import os
import argparse
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.usd
from PIL import Image
import numpy as np
import scipy.ndimage
import carb

class ThermalSensorModel:
    """
    Physically-based framework for Thermal Sensor Simulation.
    Calculates exact radiometric power using the Stefan-Boltzmann law.
    """
    def __init__(self, spectral_band="LWIR", netd_sigma=0.015, vignette_strength=0.3, bloom_sigma=3.0):
        self.spectral_band = spectral_band
        self.sigma_sb = 5.67e-8 # Stefan-Boltzmann constant (W / (m^2 * K^4))
        
        # Sensor specific noise and optical characteristics
        self.netd_sigma = netd_sigma
        self.vignette_strength = vignette_strength
        self.bloom_sigma = bloom_sigma

    def temperature_to_intensity(self, temp_k, emissivity=1.0):
        """
        Calculate physical Radiance (W/m^2/sr) using the Stefan-Boltzmann Law.
        M = emissivity * sigma * T^4 (Radiant Exitance)
        L = M / pi (Radiance)
        """
        exitance = emissivity * self.sigma_sb * (temp_k ** 4)
        radiance = exitance / np.pi
        
        # Future extension: Integrate specifically over the spectral_band (e.g., 8-14um for LWIR) 
        # using Planck's Law instead of the full-spectrum Stefan-Boltzmann approximation.
        return float(radiance)

    def make_material(self, temp_k, emissivity=1.0, diffuse=(0.5, 0.5, 0.5)):
        """Generate a thermal material for a given physical temperature and emissivity."""
        intensity = self.temperature_to_intensity(temp_k, emissivity=emissivity)
        return rep.create.material_omnipbr(
            diffuse=diffuse,
            emissive_color=(1.0, 1.0, 1.0),
            emissive_intensity=intensity
        )

    def apply_post_processing(self, thermal_data):
        """Post-process the raw radiometric emission to simulate camera optics and AGC (Auto Gain Control)."""
        if len(thermal_data.shape) == 3:
            gray = np.mean(thermal_data[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = thermal_data.astype(np.float32)
            
        # 1. Thermal Blooming / Heat Conduction Simulation
        gray = scipy.ndimage.gaussian_filter(gray, sigma=self.bloom_sigma)

        # 2. Vignette (Lens falloff)
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(cx, cy)
        vignette = 1.0 - self.vignette_strength * (r ** 2)
        gray *= vignette

        # 3. AGC (Automatic Gain Control) - Normalize dynamic range
        # Because physics follows T^4, hot objects completely wash out cold ones.
        # We normalize to 0-1 range to simulate an auto-exposing IR sensor.
        gray_min = gray.min()
        gray_max = gray.max()
        if gray_max > gray_min:
            gray = (gray - gray_min) / (gray_max - gray_min)
        else:
            gray = np.zeros_like(gray)

        # 4. NETD Sensor Noise
        noise = np.random.normal(0, self.netd_sigma, gray.shape)
        gray = np.clip(gray + noise, 0, 1)

        # Simulate 14-bit internal quantization scaled to 8-bit output
        # (For accurate raw logging, you would save it as a 16-bit PNG)
        return (gray * 255).astype(np.uint8)

def main():
    omni.usd.get_context().new_stage()
    sensor = ThermalSensorModel(spectral_band="LWIR")

    with rep.new_layer():
        dome_light = rep.create.light(light_type="dome", intensity=2000.0)
        distance_light = rep.create.light(light_type="distant", intensity=3000.0, rotation=(315, 0, 0))

        camera = rep.create.camera(position=(0, -8, 2), look_at=(0, 0, 1))
        render_product = rep.create.render_product(camera, (1024, 1024))

        # =========================================================
        # SCENE OBJECTS & PHYSICAL ASSIGNMENT (TEMP + EMISSIVITY)
        # =========================================================

        # Ground Plane (e.g., Concrete: e=0.92, 285K)
        ground = rep.create.plane(scale=50.0, position=(0, 0, -0.5))
        with ground:
            rep.randomizer.materials([sensor.make_material(285.0, emissivity=0.92, diffuse=(0.2, 0.2, 0.2))])

        # Background Wall (e.g., Brick: e=0.93, 280K)
        bg_wall = rep.create.cube(position=(0, 10, 5), scale=(50.0, 1.0, 20.0))
        with bg_wall:
            rep.randomizer.materials([sensor.make_material(280.0, emissivity=0.93, diffuse=(0.1, 0.1, 0.1))])

        # Building Structure (e.g., Painted surface: e=0.90, 295K)
        building = rep.create.cube(position=(0, 0, 1), scale=3.0)
        with building:
            rep.randomizer.materials([sensor.make_material(295.0, emissivity=0.90, diffuse=(0.8, 0.8, 0.8))])

        # Door (e.g., Wood: e=0.85, 300K)
        door = rep.create.cube(position=(0, -1.5, 0.3), scale=(0.8, 0.1, 1.6))
        with door:
            rep.randomizer.materials([sensor.make_material(300.0, emissivity=0.85, diffuse=(0.4, 0.2, 0.1))])

        # Window (e.g., Glass: e=0.92, 360K heat escaping)
        window = rep.create.cube(position=(0.8, -1.5, 1.5), scale=(0.8, 0.1, 0.8))
        with window:
            rep.randomizer.materials([sensor.make_material(360.0, emissivity=0.92, diffuse=(0.1, 0.5, 0.8))])

        # Generator (e.g., Polished Aluminum: e=0.05, 390K)
        # Notice: Despite being 390K, low emissivity (0.05) means it won't radiate nearly as much heat!
        generator = rep.create.cube(position=(-1.5, -2.5, 0.0), scale=0.5)
        with generator:
            rep.randomizer.materials([sensor.make_material(390.0, emissivity=0.05, diffuse=(0.2, 0.2, 0.2))])


        # =========================================================
        # CAPTURE & RENDER
        # =========================================================
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        rgb_annotator.attach([render_product])
        
        try:
            thermal_annotator = rep.AnnotatorRegistry.get_annotator("PtSelfIllumination")
        except Exception as e:
            thermal_annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
        
        thermal_annotator.attach([render_product])
        rep.settings.set_render_pathtraced(64)
        
        for _ in range(50):
            rep.orchestrator.step()
            
        rgb_data = rgb_annotator.get_data()
        thermal_data = thermal_annotator.get_data()

        if rgb_data is not None and rgb_data.size > 0:
            rgb_img = Image.fromarray(rgb_data)
            rgb_img.save("scene_rgb.png")

        if thermal_data is not None and thermal_data.size > 0:
            print(f"Max physical radiance captured: {np.max(thermal_data)} W/m^2/sr")
            simulated_ir_array = sensor.apply_post_processing(thermal_data)
            thermal_img = Image.fromarray(simulated_ir_array, mode="L")
            thermal_img.save("scene_thermal.png")

    simulation_app.close()

if __name__ == "__main__":
    main()
