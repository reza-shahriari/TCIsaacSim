"""Phase 2 prototype only -- see docs/isaacsim_implementation_plan.md SS1's
"prototype in plain Python first" note. Not the final architecture (that's
the SPG shader in spg/); this exists to prove thermal_physics.pipeline
against real Isaac Sim data before any CUDA is written. Retire once Phase 4
(docs/isaacsim_checklist.md) lands.
"""
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry

# from thermal_physics.pipeline import render_frame  # TODO: uncomment once implemented


class ThermalWriterPrototype(Writer):
    def __init__(self, output_dir: str):
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),                     # stand-in for encoded T(u,v) until SPG's emission pass exists
            AnnotatorRegistry.get_annotator("distance_to_image_plane"), # R(u,v)
            AnnotatorRegistry.get_annotator("semantic_segmentation"),
        ]
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        # TODO: a semantic-label -> (temperature_k, emissivity) dict, per
        # docs/isaacsim_implementation_plan.md SS2 step 1 (Phase 2 version).

    def write(self, data: dict) -> None:
        raise NotImplementedError(
            "TODO: build temperature_k/emissivity/range_m arrays from data[...], "
            "call thermal_physics.pipeline.render_frame, write/publish the result"
        )


WriterRegistry.register(ThermalWriterPrototype)
