import numpy as np
from PIL import Image
import os

def generate_maps():
    # 512x512 temperature map (0-255)
    # Let's make a hot circle in the middle
    y, x = np.ogrid[:512, :512]
    center = (256, 256)
    radius = 150
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # 0 (cold) to 255 (hot)
    temp_map = np.clip(255 - dist_from_center, 0, 255).astype(np.uint8)
    
    # Let's add some noise
    noise = np.random.randint(-20, 20, (512, 512), dtype=np.int16)
    temp_map = np.clip(temp_map.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    Image.fromarray(temp_map, mode='L').save('dummy_temp_map.png')
    
    # Emissivity map (0-255) where 255 is e=1.0
    # Let's make a vertical stripe with low emissivity (0.1 -> 25)
    emissivity_map = np.ones((512, 512), dtype=np.uint8) * 230 # e=0.9
    emissivity_map[:, 200:300] = 25 # e=0.1
    
    Image.fromarray(emissivity_map, mode='L').save('dummy_emissivity_map.png')
    print("Created dummy_temp_map.png and dummy_emissivity_map.png")

if __name__ == "__main__":
    generate_maps()
