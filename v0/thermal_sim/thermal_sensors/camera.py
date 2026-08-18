import numpy as np
import scipy.ndimage
import omni.replicator.core as rep

class MultiSpectralCamera:
    """
    A multi-spectral camera model that captures raw Isaac Sim rendering
    and processes it into specific optical bands (RGB, LWIR, SWIR).
    """
    def __init__(self, render_product):
        self.render_product = render_product
        
        # 1. Standard LDR Color for RGB
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self.rgb_annotator.attach([self.render_product])
        
        # 2. Path-Traced Self-Illumination for LWIR (Emission only)
        try:
            self.thermal_annotator = rep.AnnotatorRegistry.get_annotator("PtSelfIllumination")
        except Exception:
            self.thermal_annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
        self.thermal_annotator.attach([self.render_product])
        
        # 3. HDR Color for SWIR (Reflected light)
        self.hdr_annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
        self.hdr_annotator.attach([self.render_product])

    def get_data(self):
        """Fetch data from all annotators simultaneously."""
        return {
            "rgb": self.rgb_annotator.get_data(),
            "emission": self.thermal_annotator.get_data(),
            "hdr": self.hdr_annotator.get_data()
        }

    def process_rgb(self, data):
        """Standard RGB output."""
        return data["rgb"]

    def process_lwir(self, data, netd_sigma=0.015, vignette_strength=0.3, bloom_sigma=3.0):
        """
        Process emission data into Long-Wave Infrared (LWIR) / Thermal band.
        Uses AGC, thermal blooming, and NETD noise based on physical heat emission.
        """
        thermal_data = data["emission"]
        if thermal_data is None or thermal_data.size == 0:
            return None
            
        if len(thermal_data.shape) == 3:
            gray = np.mean(thermal_data[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = thermal_data.astype(np.float32)
            
        # Thermal blooming (Heat bleeding via lens and physical conduction)
        gray = scipy.ndimage.gaussian_filter(gray, sigma=bloom_sigma)

        # Vignette (Lens falloff)
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(cx, cy)
        vignette = 1.0 - vignette_strength * (r ** 2)
        gray *= vignette

        # Automatic Gain Control (AGC) - Normalize the T^4 radiance curve
        gray_min = gray.min()
        gray_max = gray.max()
        if gray_max > gray_min:
            gray = (gray - gray_min) / (gray_max - gray_min)
        else:
            gray = np.zeros_like(gray)

        # NETD Sensor Noise
        noise = np.random.normal(0, netd_sigma, gray.shape)
        gray = np.clip(gray + noise, 0, 1)
        
        return (gray * 255).astype(np.uint8)

    def process_swir(self, data, noise_sigma=0.04, bloom_sigma=1.0):
        """
        Process HDR scene data into Short-Wave Infrared (SWIR) band.
        SWIR captures reflected light (night-vision behavior) rather than heat emission.
        """
        hdr_data = data["hdr"]
        if hdr_data is None or hdr_data.size == 0:
            return None
            
        if len(hdr_data.shape) == 3:
            gray = np.mean(hdr_data[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = hdr_data.astype(np.float32)
            
        # Mild bloom for optical scattering in the SWIR band
        gray = scipy.ndimage.gaussian_filter(gray, sigma=bloom_sigma)
        
        # SWIR cameras usually compress high dynamic range to boost shadows (Logarithmic AGC)
        gray_min = gray.min()
        gray_max = gray.max()
        if gray_max > gray_min:
            gray = np.log1p(gray) / np.log1p(gray_max)
        else:
            gray = np.zeros_like(gray)

        # SWIR sensors (InGaAs) typically have pronounced shot noise
        noise = np.random.normal(0, noise_sigma, gray.shape)
        gray = np.clip(gray + noise, 0, 1)
        
        return (gray * 255).astype(np.uint8)
