from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.replicator.core as rep
if hasattr(rep.create, 'material'):
    help(rep.create.material)
else:
    print("NO rep.create.material")
app.close()
