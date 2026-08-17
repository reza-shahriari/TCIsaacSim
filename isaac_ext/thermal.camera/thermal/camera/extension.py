"""Isaac Sim extension entry point. docs/isaacsim_implementation_plan.md SS6
(Project Structure) -- packaged the way isaacsim.sensors.rtx is packaged.
"""
import omni.ext


class ThermalCameraExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        # TODO: register the writer_prototype.ThermalWriterPrototype (Phase 2),
        # and later the SPG shader (Phase 4, ../../../spg/).
        raise NotImplementedError("TODO: extension startup")

    def on_shutdown(self) -> None:
        raise NotImplementedError("TODO: extension shutdown / cleanup")
