"""Time-varying temperature -- how T(t) is generated, not how it's sensed.

Implements docs/thermal_object_dynamics.md. Feeds thermal_physics.pipeline
its per-point temperature_k field; has no knowledge of cameras/rendering.
"""
