import omni.replicator.core as rep
import omni.kit.commands
import omni.usd
from pxr import Sdf, UsdShade
import os

def make_thermal_material(base_temp_k=300.0, emissivity=0.95, 
                          diffuse_color=(0.4, 0.4, 0.45), mat_id=None):
    """
    Generate a dynamic thermal material using custom MDL shader.
    
    Args:
        base_temp_k: Initial temperature in Kelvin
        emissivity: Surface emissivity (0-1)
        diffuse_color: RGB diffuse color for visible-light rendering
        mat_id: Optional unique ID for the material name
    """
    if mat_id is None:
        mat_id = f"{base_temp_k:.0f}_{emissivity:.2f}"
    
    mat_name = f"thermal_mat_{mat_id}"
    mat_path = f"/Replicator/Looks/{mat_name}"
    
    stage = omni.usd.get_context().get_stage()
    
    # Resolve the absolute path to the MDL file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mdl_path = os.path.join(current_dir, "dynamic_thermal.mdl")
    
    if not stage.GetPrimAtPath(mat_path):
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(mdl_path, "mdl")
        shader.SetSourceAssetSubIdentifier("dynamic_thermal", "mdl")
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        
        shader.CreateInput("base_temp_k", Sdf.ValueTypeNames.Float).Set(base_temp_k)
        shader.CreateInput("emissivity", Sdf.ValueTypeNames.Float).Set(emissivity)
        shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
            
    return mat_path
