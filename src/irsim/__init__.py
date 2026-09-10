"""irsim — engine-free physics core for a multi-band infrared camera simulator.

Nothing in this package may import an engine module (omni, pxr, isaacsim, warp, carb),
the glue package irsim_isaac, or the ML/imaging stack (torch, cv2, PIL, ...).
Engine glue belongs in `irsim_isaac`. See CLAUDE.md, non-negotiable #1.
"""

__version__ = "0.1.0"
