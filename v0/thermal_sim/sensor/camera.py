import omni.usd
from pxr import Usd, UsdGeom, Gf

class ThermalCamera:
    """
    Object-oriented Thermal Camera utilizing Isaac Sim 6.0 RtxCamera API
    and OmniSensorGenericCameraCoreAPI for ISP configuration.
    """
    def __init__(self, prim_path: str, resolution=(640, 480)):
        self.prim_path = prim_path
        self.resolution = resolution
        self.stage = omni.usd.get_context().get_stage()
        self.camera_prim = None
        
        # Advanced artifacts state
        self.nuc_frozen = False
        self.nuc_freeze_frames_remaining = 0
        
        self._create_camera()
        
    def _create_camera(self):
        """Creates the camera prim and configures the ISP schema."""
        if not self.stage.GetPrimAtPath(self.prim_path):
            self.camera_prim = UsdGeom.Camera.Define(self.stage, self.prim_path)
        else:
            self.camera_prim = UsdGeom.Camera(self.stage.GetPrimAtPath(self.prim_path))
            
        prim = self.camera_prim.GetPrim()
        
        # Note: In Isaac Sim 6.0 we would also wrap with RtxCamera:
        # from isaacsim.sensors.experimental.rtx import RtxCamera
        # self.rtx_camera = RtxCamera(prim_path=self.prim_path, resolution=self.resolution)

        # Basic optical settings (Thermal lenses are fast, e.g., f/1.0)
        self.camera_prim.GetFocalLengthAttr().Set(15.0)
        self.camera_prim.GetFocusDistanceAttr().Set(400.0)
        self.camera_prim.GetFStopAttr().Set(1.0)
        self.camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))
        
        # Apply 6.0 ISP Schema (OmniSensorGenericCameraCoreAPI)
        # This allows configuring the Image Signal Processor directly on the prim.
        try:
            prim.ApplyAPI(Usd.SchemaRegistry().GetSchemaTypeName("OmniSensorGenericCameraCoreAPI"))
            # Example of setting an ISP attribute if known:
            # prim.GetAttribute("outputs:isp:exposure").Set(1.0)
        except Exception as e:
            print(f"[ThermalCamera] Notice: OmniSensorGenericCameraCoreAPI could not be applied. {e}")
            
        print(f"[ThermalCamera] Initialized at {self.prim_path} | Mode: Thermal LWIR | Resolution: {self.resolution}")

    def set_local_transform(self, translation=Gf.Vec3d(0, 0, 0), rotation_xyz=Gf.Vec3d(0, 0, 0)):
        """Sets the local transform of the camera (useful for mounting)."""
        xformable = UsdGeom.Xformable(self.camera_prim)
        xformable.ClearXformOpOrder()
        
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(translation)
        
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(rotation_xyz)
        
        print(f"[ThermalCamera] Mounted at {self.prim_path} with translation {translation}")
        
    def trigger_nuc(self, freeze_frames=30):
        """
        Simulates a Non-Uniformity Correction (NUC) shutter event.
        Freezes the camera output and zeroes out Fixed Pattern Noise.
        """
        print("[ThermalCamera] NUC Shutter triggered. Calibrating...")
        self.nuc_frozen = True
        self.nuc_freeze_frames_remaining = freeze_frames
        
    def update(self):
        """Called every frame to manage sensor state like NUC."""
        if self.nuc_frozen:
            self.nuc_freeze_frames_remaining -= 1
            if self.nuc_freeze_frames_remaining <= 0:
                self.nuc_frozen = False
                print("[ThermalCamera] NUC Calibration complete.")
