"""Headless Isaac Sim launch. docs/isaacsim_checklist.md steps 2.1-2.3.

Run with: python scripts/run_standalone.py
(needs Isaac Sim's own Python environment, not this project's venv -- see
docs/isaacsim_implementation_plan.md's "Hard prerequisite" note)
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

# TODO: build/load a scene, step the simulation, capture frames.
# Start with nothing here at all (checklist 2.3's bare launch-then-close
# check) before adding scene content.

simulation_app.close()
