from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd
from pxr import UsdShade, Sdf

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

mat_path = "/Replicator/Looks/dynamic_thermal"
mdl_path = "/home/reza/Vision/ThIsaac/thermal_sim/thermal_utils/dynamic_thermal.mdl"

material = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
shader.SetSourceAsset(mdl_path, "mdl")
shader.SetSourceAssetSubIdentifier("dynamic_thermal", "mdl")
material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

print("Created material:", material.GetPath())
app.close()
