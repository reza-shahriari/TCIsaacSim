from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.replicator.core as rep
print([x for x in dir(rep.create) if 'material' in x])
app.close()
