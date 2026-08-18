"""
Thermal IR Camera Simulation — Physically-Based Pipeline
=========================================================

Architecture:
  1. Scene objects are tagged with semantic labels (class names)
  2. Each frame: semantic segmentation annotator → per-pixel object ID
  3. Thermo solver gives T(t) for each object
  4. Per-pixel temperature map built from semantics → T lookup
  5. Full Planck + Beer-Lambert + sensor noise pipeline applied in Python
  6. No hacks: temperatures live in Python, MDL only handles visible RGB

Progressive steps saved as RGB+IR pairs:
  step1_raw_temp         — temperature map from semantics (no physics)
  step2_planck           — Planck band-integrated radiance
  step3_atmosphere       — Beer-Lambert atmospheric attenuation
  step4_sensor_noise     — sensor noise, blooming, Narcissus
  step5_dynamic_gif      — 60-frame heat-up / cool-down animation
"""

import os
import sys
import numpy as np
import imageio

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.usd
from pxr import UsdShade, Sdf

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

OUTPUT_DIR = os.path.join(current_dir, "output_steps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from sensor.pipeline import RadiometricPipeline
from physics.thermo import ThermalObject, ThermalSystem

# =============================================================================
# Scene object definitions — class name → physical properties
# =============================================================================
OBJECTS = {
    "sea":       {"emissivity": 0.96, "diffuse": (0.08, 0.12, 0.22)},
    "chassis":   {"emissivity": 0.85, "diffuse": (0.28, 0.28, 0.32)},
    "engine":    {"emissivity": 0.92, "diffuse": (0.48, 0.28, 0.18)},
    "cold_pole": {"emissivity": 0.30, "diffuse": (0.65, 0.65, 0.70)},
}

# Background temperature for pixels that don't hit any object (sky)
T_SKY = 220.0  # K

# =============================================================================
# MDL material — pure visible-light diffuse (no thermal encoding tricks)
# =============================================================================
MDL_SRC = """
mdl 1.6;
import ::df::*;
import ::state::*;
import ::math::*;

export material visible_diffuse(
    uniform color diffuse_color = color(0.4, 0.4, 0.4)
) = material(
    surface: material_surface(
        scattering: df::diffuse_reflection_bsdf(tint: diffuse_color)
    )
);
"""

def write_mdl():
    mdl_path = os.path.join(current_dir, "thermal_utils", "visible_diffuse.mdl")
    with open(mdl_path, "w") as f:
        f.write(MDL_SRC)
    return mdl_path


def make_visible_material(stage, name: str, diffuse: tuple) -> str:
    """Create a plain diffuse material on the USD stage and return its path."""
    mdl_path = os.path.join(current_dir, "thermal_utils", "visible_diffuse.mdl")
    mat_path = f"/World/Looks/mat_{name}"
    from pxr import UsdShade, Sdf
    if not stage.GetPrimAtPath(mat_path):
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(mdl_path, "mdl")
        shader.SetSourceAssetSubIdentifier("visible_diffuse", "mdl")
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(diffuse)
    return mat_path


def bind_mat(stage, prim_path: str, mat_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    if prim.IsValid() and mat:
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


# =============================================================================
# IR post-processing
# =============================================================================
def whot_palette(ir_float):
    """
    White-hot palette: cold=black (0), hot=white (255).
    Input: 2D float array (already normalized 0→1).
    Returns: uint8 grayscale image.
    """
    return (np.clip(ir_float, 0, 1) * 255).astype(np.uint8)


def log_agc(signal: np.ndarray) -> np.ndarray:
    """Per-frame log AGC — use only for static screenshots, NOT for GIF."""
    signal = np.clip(signal, 0, None)
    s_min, s_max = signal.min(), signal.max()
    if s_max <= s_min:
        return np.zeros_like(signal, dtype=np.float32)
    log_s = np.log1p(signal - s_min)
    return (log_s / np.log1p(s_max - s_min)).astype(np.float32)


def global_agc(signal: np.ndarray, g_log_p1: float, g_log_p99: float) -> np.ndarray:
    """
    Percentile-clipped log-space normalization.
    Per thermal_camera_model.md §10: use 1st/99th percentile (not absolute
    min/max) so that a cold sky or a tiny saturated hot-spot doesn't crush
    all other scene contrast. Fixed across ALL frames so brightness changes
    visibly over time as the engine heats and cools.
    """
    signal = np.clip(signal, 1e-30, None)
    log_s = np.log(signal)
    if g_log_p99 <= g_log_p1:
        return np.zeros_like(signal, dtype=np.float32)
    return np.clip((log_s - g_log_p1) / (g_log_p99 - g_log_p1), 0, 1).astype(np.float32)


def save_pair(step_name, rgb_img, ir_img, t_map=None, note=""):
    rgb_path = os.path.join(OUTPUT_DIR, f"{step_name}_rgb.png")
    ir_path  = os.path.join(OUTPUT_DIR, f"{step_name}_ir.png")
    imageio.imsave(rgb_path, rgb_img)
    imageio.imsave(ir_path, ir_img)
    temp_info = f"  Temp: [{t_map.min():.1f}, {t_map.max():.1f}]K  mean={t_map.mean():.1f}K" if t_map is not None else ""
    print(f"  → Saved {step_name} | IR range [{ir_img.min()}, {ir_img.max()}]{temp_info}  {note}")


# =============================================================================
# Build temperature map from semantic labels
# =============================================================================
def build_temp_map(sem_data, id_to_temp, default_temp=T_SKY):
    """
    sem_data : 2D uint32 array of semantic IDs (from SemSeg annotator)
    id_to_temp: dict mapping semantic ID → temperature in K
    Returns float32 2D array of per-pixel temperatures.
    """
    t_map = np.full(sem_data.shape, default_temp, dtype=np.float32)
    for sid, temp in id_to_temp.items():
        t_map[sem_data == sid] = temp
    return t_map


def build_emissivity_map(sem_data, id_to_emissivity, default=0.5):
    e_map = np.full(sem_data.shape, default, dtype=np.float32)
    for sid, e in id_to_emissivity.items():
        e_map[sem_data == sid] = e
    return e_map


# =============================================================================
# Main
# =============================================================================
def main():
    write_mdl()

    # --------------------------------------------------------------------------
    # 1. Physics setup
    # --------------------------------------------------------------------------
    ambient_temp = 280.0
    sys_thermo = ThermalSystem(ambient_temp_k=ambient_temp)

    engine_phys  = ThermalObject("engine",  ambient_temp, 150.0,  450.0)
    chassis_phys = ThermalObject("chassis", ambient_temp, 500.0,  900.0)
    sea_phys     = ThermalObject("sea",     ambient_temp, 1e6,   4000.0)
    # Cold pole: no thermo, stays at ambient - 20K
    cold_temp = ambient_temp - 20.0

    sys_thermo.add_object(engine_phys)
    sys_thermo.add_object(chassis_phys)
    sys_thermo.add_object(sea_phys)
    sys_thermo.add_conduction("engine", "chassis", h_W_per_K=100.0)
    sys_thermo.add_conduction("chassis", "sea",    h_W_per_K=500.0)
    sys_thermo.set_convection("engine",  20.0)
    sys_thermo.set_convection("chassis", 50.0)

    # Warm up engine so there's visible temperature contrast at frame 0
    for _ in range(15):
        engine_phys.add_heat(30000.0)
        sys_thermo.step(dt_seconds=30.0)
    temps = sys_thermo.get_temperatures()
    print(f"Initial temps: Engine={temps['engine']:.1f}K  Chassis={temps['chassis']:.1f}K  Sea={temps['sea']:.1f}K")

    # --------------------------------------------------------------------------
    # 2. Scene setup
    # --------------------------------------------------------------------------
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # --------------------------------------------------------------------------
    # Scene setup — create objects THEN attach annotators
    # --------------------------------------------------------------------------
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Lighting for RGB
    rep.create.light(light_type="dome",    intensity=600.0)
    rep.create.light(light_type="distant", intensity=4000.0, rotation=(315, 45, 0))

    camera = rep.create.camera(position=(10, 10, 12), look_at=(0, 0, 3.5))
    rp = rep.create.render_product(camera, (640, 480))

    # Annotators
    ldr_ann  = rep.AnnotatorRegistry.get_annotator("LdrColor")
    ldr_ann.attach([rp])

    sem_ann  = rep.AnnotatorRegistry.get_annotator("SemanticSegmentation")
    sem_ann.attach([rp])

    # Depth: correct name is 'distance_to_camera'
    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    depth_ann.attach([rp])

    rep.settings.set_render_pathtraced(32)

    # Materials (pure diffuse for RGB)
    mats = {}
    for name, props in OBJECTS.items():
        mats[name] = make_visible_material(stage, name, props["diffuse"])

    # Scene objects with semantic labels
    sea_prim     = rep.create.cube(     semantics=[("class", "sea")],       position=(0, 0, -0.5), scale=(40, 40, 0.5))
    chassis_prim = rep.create.cube(     semantics=[("class", "chassis")],   position=(0, 0, 4.0),  scale=(2.5, 2.5, 0.4))
    engine_prim  = rep.create.cylinder( semantics=[("class", "engine")],    position=(0, 0, 4.5),  scale=(0.6, 0.6, 0.5))
    pole_prim    = rep.create.cylinder( semantics=[("class", "cold_pole")], position=(5, 0, 2.0),  scale=(0.3, 0.3, 2.0))

    # Bind materials
    def get_prim_paths(group):
        return [str(p) for p in group.node.get_attribute("inputs:primsIn").get()]

    all_prims = {
        "sea":       get_prim_paths(sea_prim),
        "chassis":   get_prim_paths(chassis_prim),
        "engine":    get_prim_paths(engine_prim),
        "cold_pole": get_prim_paths(pole_prim),
    }

    for name, paths in all_prims.items():
        mat = UsdShade.Material(stage.GetPrimAtPath(str(mats[name])))
        for p in paths:
            prim = stage.GetPrimAtPath(p)
            if prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).Bind(mat)
                print(f"  Bound {name} → {p}")

    # Warm up renderer
    print("Warming up renderer...")
    for _ in range(20):
        rep.orchestrator.step()

    # --------------------------------------------------------------------------
    # 3. Decode semantic labels
    # --------------------------------------------------------------------------
    # Run one frame to get semantic data
    rep.orchestrator.step()
    rep.orchestrator.step()
    sem_raw = sem_ann.get_data()

    # sem_raw may have an "info" sub-key with idToLabels
    print(f"Semantic raw type: {type(sem_raw)}")
    if isinstance(sem_raw, dict):
        id_to_labels = sem_raw.get("info", {}).get("idToLabels", {})
        sem_array = sem_raw.get("data", sem_raw)
        print(f"  idToLabels: {id_to_labels}")
    else:
        sem_array = sem_raw
        id_to_labels = {}
        print(f"  sem_array shape: {sem_array.shape}, dtype: {sem_array.dtype}, unique: {np.unique(sem_array)}")

    # Build class_name → semantic_id map
    # The SemanticSegmentation annotator in Isaac Sim 6 returns a plain ndarray.
    # We need to use the StableIdSemanticIdMap or the idToLabels from info dict.
    # Try the 'semantic_segmentation' (lowercase) annotator which provides a dict:
    class_to_sid = {}
    if isinstance(sem_raw, dict):
        id_to_labels = sem_raw.get("info", {}).get("idToLabels", {})
        for sid_str, labels in id_to_labels.items():
            class_name = labels.get("class", "") if isinstance(labels, dict) else str(labels)
            if class_name:
                class_to_sid[class_name] = int(sid_str)

    # Fallback: query USD stage semantics directly to map prim semantic IDs
    if not class_to_sid:
        from pxr import Semantics
        unique_ids = np.unique(sem_array)
        print(f"  Unique semantic IDs from annotator: {unique_ids}")
        # Walk all prims and collect their semanticIds + class labels
        prim_class_to_sid = {}
        for prim in stage.Traverse():
            if prim.HasAPI(Semantics.SemanticsAPI, "class"):
                sem_api = Semantics.SemanticsAPI(prim, "class")
                label_attr = sem_api.GetSemanticLabelAttr()
                if label_attr and label_attr.Get():
                    class_label = label_attr.Get()
                    # Find which ID in the image corresponds to this prim
                    # by checking the prim's path in our known prim path dict
                    prim_path = str(prim.GetPath())
                    for obj_name, paths in all_prims.items():
                        if any(prim_path.startswith(p) or p.startswith(prim_path) for p in paths):
                            prim_class_to_sid[obj_name] = class_label
        print(f"  USD prim class labels: {prim_class_to_sid}")

        # Use the 'semantic_segmentation' (lowercase) annotator for the dict version
        sem_ann2 = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        sem_ann2.attach([rp])
        for _ in range(3):
            rep.orchestrator.step()
        sem_raw2 = sem_ann2.get_data()
        print(f"  semantic_segmentation (lowercase) type: {type(sem_raw2)}, keys: {list(sem_raw2.keys()) if isinstance(sem_raw2, dict) else 'array'}")
        if isinstance(sem_raw2, dict):
            id_to_labels2 = sem_raw2.get("info", {}).get("idToLabels", {})
            print(f"  idToLabels: {id_to_labels2}")
            for sid_str, labels in id_to_labels2.items():
                if isinstance(labels, dict):
                    class_name = labels.get("class", "")
                else:
                    class_name = str(labels)
                if class_name and class_name in OBJECTS:
                    class_to_sid[class_name] = int(sid_str)

        if not class_to_sid:
            # Last resort: assign IDs in order of object creation (ID 0=bg, 2=sea, 3=chassis, 4=engine, 5=pole)
            # This matches typical Replicator semantic ID assignment order
            ordered_classes = ["sea", "chassis", "engine", "cold_pole"]
            non_zero_ids = sorted([i for i in unique_ids if i > 0])
            for i, cls in enumerate(ordered_classes):
                if i < len(non_zero_ids):
                    class_to_sid[cls] = non_zero_ids[i]
            print(f"  Used fallback ID assignment: {class_to_sid}")

    print(f"  Final class→semantic_id: {class_to_sid}")

    # --------------------------------------------------------------------------
    # 4. Helper: build per-pixel T map from current thermo state
    # --------------------------------------------------------------------------
    def current_temp_map(temps_dict, sem_arr, c2s):
        """Map semantic IDs to current temperatures."""
        # Default: sky temperature
        t_map = np.full(sem_arr.shape[:2], T_SKY, dtype=np.float32)

        temp_lookup = {
            "engine":    temps_dict.get("engine",  ambient_temp),
            "chassis":   temps_dict.get("chassis", ambient_temp),
            "sea":       temps_dict.get("sea",     ambient_temp),
            "cold_pole": cold_temp,
        }

        for cls_name, temp in temp_lookup.items():
            sid = c2s.get(cls_name)
            if sid is not None:
                if len(sem_arr.shape) == 3:
                    mask = sem_arr[:, :, 0] == sid
                else:
                    mask = sem_arr == sid
                t_map[mask] = temp

        return t_map

    def current_emissivity_map(sem_arr, c2s):
        e_map = np.full(sem_arr.shape[:2], 0.5, dtype=np.float32)
        for cls_name, props in OBJECTS.items():
            sid = c2s.get(cls_name)
            if sid is not None:
                if len(sem_arr.shape) == 3:
                    mask = sem_arr[:, :, 0] == sid
                else:
                    mask = sem_arr == sid
                e_map[mask] = props["emissivity"]
        return e_map

    pipeline = RadiometricPipeline()

    # --------------------------------------------------------------------------
    # 5. Progressive steps using INITIAL temperatures
    # --------------------------------------------------------------------------
    def get_rgb():
        d = ldr_ann.get_data()
        return d[:, :, :3] if d is not None else None

    def get_depth():
        d = depth_ann.get_data()
        return d.astype(np.float32) if d is not None else None

    def get_sem():
        d = sem_ann.get_data()
        if isinstance(d, dict):
            return d.get("data", d)
        return d

    print("\n=== Generating Progressive Pipeline Steps ===")
    rep.orchestrator.step()
    rep.orchestrator.step()
    rgb0 = get_rgb()
    sem0 = get_sem()
    depth0 = get_depth()
    depth_used = depth0 if depth0 is not None else np.full(sem0.shape[:2], 10.0, np.float32)

    t_map0 = current_temp_map(sys_thermo.get_temperatures(), sem0, class_to_sid)
    e_map0 = current_emissivity_map(sem0, class_to_sid)

    # Step 0: Raw Temperature (no physics, just visualization)
    save_pair("step0_raw_temp", rgb0, pipeline.palette_white_hot(log_agc(t_map0)), t_map0, "(Raw Temp)")

    # Step 1: Surface Emission (Planck's Law + Emissivity)
    l_self = pipeline.band_integrated_radiance(t_map0)
    l_env  = pipeline.band_integrated_radiance(np.full_like(t_map0, pipeline.t_atm))
    l_point = e_map0 * l_self + (1.0 - e_map0) * l_env
    save_pair("step1_surface_emission", rgb0, pipeline.palette_white_hot(log_agc(l_point)), t_map0, "(Planck + Emissivity)")

    # Step 2: Atmospheric Attenuation
    l_recv = pipeline.apply_atmosphere(l_point, depth_used)
    save_pair("step2_atmosphere", rgb0, pipeline.palette_white_hot(log_agc(l_recv)), t_map0, "(Beer-Lambert)")

    # Step 3: Optics (Irradiance)
    irr = pipeline.apply_optics(l_recv)
    save_pair("step3_optics_irradiance", rgb0, pipeline.palette_white_hot(log_agc(irr)), t_map0, "(Lens + Vignette)")

    # Step 4: MTF Blur & Blooming
    blur_sig = pipeline.apply_mtf_blur(irr, depth_used)
    save_pair("step4_mtf_blur", rgb0, pipeline.palette_white_hot(log_agc(blur_sig)), t_map0, "(Diffraction + DoF + Blooming)")

    # Step 5: Fixed-Pattern Noise (FPN)
    fpn_sig = pipeline.apply_fpn(blur_sig, nuc_frozen=False)
    save_pair("step5_fpn", rgb0, pipeline.palette_white_hot(log_agc(fpn_sig)), t_map0, "(Spatial Noise + Bad Pixels)")

    # Step 6: Temporal & Readout Noise
    noise_sig = pipeline.apply_temporal_noise(fpn_sig)
    save_pair("step6_temporal_noise", rgb0, pipeline.palette_white_hot(log_agc(noise_sig)), t_map0, "(3D Noise Model)")

    # Step 7: ADC Quantization
    quantized_sig = pipeline.quantize(noise_sig, bits=14)
    save_pair("step7_quantized", rgb0, pipeline.palette_white_hot(log_agc(quantized_sig)), t_map0, "(14-bit ADC)")
    
    # Step 8: AGC (Histogram Equalization vs Linear)
    agc_lin  = pipeline.agc_linear_percentile(quantized_sig)
    agc_hist = pipeline.agc_histogram_eq(quantized_sig)
    agc_plat = pipeline.agc_plateau_eq(quantized_sig)
    save_pair("step8_agc_linear", rgb0, pipeline.palette_white_hot(agc_lin), t_map0, "(Linear Percentile)")
    save_pair("step8_agc_hist_eq", rgb0, pipeline.palette_white_hot(agc_hist), t_map0, "(Histogram Eq)")
    save_pair("step8_agc_plateau_eq", rgb0, pipeline.palette_white_hot(agc_plat), t_map0, "(Plateau Eq / CLAHE)")

    # Step 9: Palettes (Using Plateau Eq output)
    save_pair("step9_palette_blackhot", rgb0, pipeline.palette_black_hot(agc_plat), t_map0, "(Black Hot)")
    save_pair("step9_palette_ironbow", rgb0, pipeline.palette_ironbow(agc_plat), t_map0, "(Ironbow False Color)")

    # --------------------------------------------------------------------------
    # 6. Step 5: Dynamic 60-frame GIF
    # --------------------------------------------------------------------------
    print("\n=== Step 5: Dynamic Thermodynamics (60 frames) ===")
    physics_dt = 5.0
    num_frames = 60

    # ------------------------------------------------------------------
    # 5a. PRE-SIMULATE physics to find the global radiance range
    #     This is the key fix: we must normalize ALL frames against the
    #     SAME global min/max so that brightness visibly changes over time.
    # ------------------------------------------------------------------
    print("  Pre-simulating physics to compute global radiance range...")
    import copy
    # snapshot solver state so we can replay from the same starting point
    saved_engine_temp  = engine_phys.temp_k
    saved_chassis_temp = chassis_phys.temp_k
    saved_sea_temp     = sea_phys.temp_k

    # Grab the fixed semantic + depth maps (objects don't move)
    rep.orchestrator.step()
    sem_fixed   = get_sem()
    depth_fixed = get_depth()
    if depth_fixed is None:
        depth_fixed = np.full(t_map0.shape, 10.0, np.float32)

    e_map_fixed = current_emissivity_map(sem_fixed, class_to_sid)

    # Accumulate LOG-SPACE global bounds across all frames
    g_log_min =  1e30
    g_log_max = -1e30

    # Temporary lightweight sim for range computation (pure Python, no Isaac)
    presim = ThermalSystem(ambient_temp_k=ambient_temp)
    pe = ThermalObject("engine",  saved_engine_temp,  150.0, 450.0)
    pc = ThermalObject("chassis", saved_chassis_temp, 500.0, 900.0)
    ps = ThermalObject("sea",     saved_sea_temp,       1e6, 4000.0)
    presim.add_object(pe); presim.add_object(pc); presim.add_object(ps)
    presim.add_conduction("engine", "chassis", h_W_per_K=100.0)
    presim.add_conduction("chassis", "sea",    h_W_per_K=500.0)
    presim.set_convection("engine", 20.0)
    presim.set_convection("chassis", 50.0)

    # -----------------------------------------------------------------------
    # Physics-based analytical normalization bounds.
    #
    # Pixel-statistics approaches (percentile, min/max) fail when the engine
    # is a tiny fraction of the image — the sky/sea pixels dominate the
    # histogram and the engine never shows up in any useful percentile.
    #
    # Instead: compute the irradiance that the pipeline produces for a single
    # pixel at T_cold (ambient) and T_hot (max expected engine temp), at a
    # representative distance. This gives a physically motivated, fixed scale
    # so the engine sweeps from near-black to near-white as it heats up.
    # -----------------------------------------------------------------------
    T_cold = 240.0                 # K  — anchor below ambient so sea/chassis are visible mid-gray
    T_hot  = 520.0                 # K  — upper bound (engine at full load)
    ref_dist = np.array([[8.0]])   # metres — representative camera distance
    ref_e    = np.array([[0.92]])  # representative emissivity

    irr_cold = pipeline.process_frame(np.array([[T_cold]]), ref_dist, ref_e)
    irr_hot  = pipeline.process_frame(np.array([[T_hot]]),  ref_dist, ref_e)

    g_log_p1  = float(np.log(np.clip(irr_cold, 1e-30, None)))
    g_log_p99 = float(np.log(np.clip(irr_hot,  1e-30, None)))
    print(f"  Physics-based log-irradiance range: cold({T_cold}K)={g_log_p1:.3f}  hot({T_hot}K)={g_log_p99:.3f}")

    # ------------------------------------------------------------------
    # 5b. Render frames using FIXED global scale
    # ------------------------------------------------------------------
    rgb_frames = []
    ir_frames  = []

    for frame in range(num_frames):
        engine_phys.add_heat(30000.0 if frame < 30 else 0.0)
        sys_thermo.step(dt_seconds=physics_dt)
        temps_now = sys_thermo.get_temperatures()

        rep.orchestrator.step()
        sem_f   = get_sem()
        depth_f = get_depth()
        rgb_f   = get_rgb()

        if sem_f is None or rgb_f is None:
            continue

        t_map_f = current_temp_map(temps_now, sem_f, class_to_sid)
        e_map_f = current_emissivity_map(sem_f, class_to_sid)
        depth_f = depth_f if depth_f is not None else np.full(t_map_f.shape, 10.0, np.float32)

        # Radiometric pipeline
        irr_f = pipeline.process_frame(t_map_f, depth_f, e_map_f)
        irr_f = pipeline.apply_sensor_artifacts(irr_f, depth_f, nuc_frozen=False)

        # KEY FIX: percentile-clipped global log-space normalization
        norm_f = global_agc(irr_f, g_log_p1, g_log_p99)
        ir_f   = whot_palette(norm_f)

        rgb_frames.append(rgb_f[:, :, :3])
        ir_frames.append(ir_f)

        if frame % 10 == 0:
            print(f"  Frame {frame:03d} | Engine={temps_now['engine']:.1f}K  "
                  f"Chassis={temps_now['chassis']:.1f}K  "
                  f"IR mean={ir_f.mean():.1f}  max={ir_f.max()}")

    if ir_frames:
        imageio.mimsave(os.path.join(OUTPUT_DIR, "step5_dynamic_ir.gif"),  ir_frames,  fps=10)
        imageio.mimsave(os.path.join(OUTPUT_DIR, "step5_dynamic_rgb.gif"), rgb_frames, fps=10)
        print(f"  Saved {len(ir_frames)}-frame GIFs.")

    print("\n✅ All steps complete! Output → thermal_sim/output_steps/")
    simulation_app.close()


if __name__ == "__main__":
    main()
