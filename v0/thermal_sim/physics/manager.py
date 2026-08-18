import omni.timeline
import omni.physx
from .thermo import ThermalSystem, ThermalObject

class ThermalSimulationManager:
    """
    Manages the thermodynamic simulation step in sync with Isaac Sim's physics timeline.
    """
    def __init__(self, ambient_temp_k=290.0):
        self.system = ThermalSystem(ambient_temp_k=ambient_temp_k)
        self._timeline = omni.timeline.get_timeline_interface()
        self._physx_interface = omni.physx.get_physx_interface()
        self._sub = None
        
    def start_simulation(self):
        """Subscribe to physics step events."""
        if not self._sub:
            # Hook into the physx step event to drive thermodynamics
            self._sub = self._physx_interface.subscribe_physics_step_events(self._on_physics_step)
            print("[ThermalManager] Subscribed to physics step events.")

    def stop_simulation(self):
        """Unsubscribe from physics step events."""
        if self._sub:
            self._sub.unsubscribe()
            self._sub = None
            print("[ThermalManager] Unsubscribed from physics step events.")

    def _on_physics_step(self, dt):
        """
        Callback triggered every physics frame.
        dt: delta time in seconds.
        """
        self.system.step(dt_seconds=dt)
        
        # In a full integration, we would read the updated temperatures from self.system
        # and apply them to USD Prim attributes (e.g. inputs:base_temp_k) so the 
        # OmniGraph rendering pipeline sees the new temperatures.
        
        # Example:
        # temps = self.system.get_temperatures()
        # for name, temp in temps.items():
        #     prim = self._get_prim(name)
        #     if prim.HasAttribute("outputs:thermal:temperature"):
        #         prim.GetAttribute("outputs:thermal:temperature").Set(temp)
