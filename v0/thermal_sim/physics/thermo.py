import numpy as np

class ThermalObject:
    def __init__(self, name, initial_temp_k, mass_kg, specific_heat_capacity, is_wet=False):
        self.name = name
        self.temp_k = initial_temp_k
        self.mass = mass_kg
        self.c = specific_heat_capacity # J/(kg*K)
        self.active_heating_watts = 0.0
        self.is_wet = is_wet # Phase 5: Surface moisture (increases inertia, adds evap cooling)
        
    def add_heat(self, watts):
        self.active_heating_watts = watts
        
    def update_internal(self, dt_seconds):
        pass

class TerrainObject(ThermalObject):
    """
    Phase 4: 1D Terrain Heat Equation.
    Models subsurface heat transfer for realistic thermal inertia (roads staying hot at night).
    """
    def __init__(self, name, initial_temp_k, mass_kg, specific_heat_capacity, thermal_conductivity=1.5):
        super().__init__(name, initial_temp_k, mass_kg, specific_heat_capacity)
        # 3 layers: 0=Surface, 1=Shallow, 2=Deep Core (Constant)
        self.layer_temps = np.array([initial_temp_k, initial_temp_k, initial_temp_k - 5.0])
        self.k_soil = thermal_conductivity
        
    def update_internal(self, dt_seconds):
        # Heat conducts between surface and shallow layer
        # Simplified explicit solver
        dT_surface = (self.layer_temps[1] - self.layer_temps[0]) * self.k_soil * dt_seconds / (self.mass * self.c * 0.1)
        dT_shallow = (self.layer_temps[0] - self.layer_temps[1]) * self.k_soil * dt_seconds / (self.mass * self.c * 0.9)
        self.layer_temps[0] += dT_surface
        self.layer_temps[1] += dT_shallow
        self.temp_k = self.layer_temps[0] # Expose surface temp to system

class ThermalSystem:
    """
    A lumped-capacitance thermodynamic solver.
    Tracks temperature states of objects over time considering thermal mass,
    active heat generation, conduction, and convection.
    """
    def __init__(self, ambient_temp_k=290.0, wind_speed_m_s=0.0):
        self.ambient_temp_k = ambient_temp_k
        self.wind_speed = wind_speed_m_s # Phase 5: Wind driven convection
        self.objects = {}
        # Conduction paths: (obj1_name, obj2_name) -> thermal_conductivity_W_per_K (U*A)
        self.conduction_paths = {}
        # Convection paths: obj_name -> convection_coefficient_W_per_K (U*A to ambient)
        self.convection_paths = {}
        
    def add_object(self, obj):
        self.objects[obj.name] = obj
        self.convection_paths[obj.name] = 0.0 # Default no convection
        
    def add_conduction(self, name1, name2, h_W_per_K):
        """Add a conductive heat transfer path between two objects."""
        self.conduction_paths[(name1, name2)] = h_W_per_K
        
    def set_convection(self, name, h_W_per_K):
        """Set convective heat transfer to the ambient environment (e.g., air or sea)."""
        self.convection_paths[name] = h_W_per_K
        
    def step(self, dt_seconds):
        """Advance the simulation by dt_seconds, calculating all heat flows (Joules)."""
        dE = {name: obj.active_heating_watts * dt_seconds for name, obj in self.objects.items()}
        
        # Convection to ambient
        for name, h in self.convection_paths.items():
            obj = self.objects[name]
            # Phase 5: Wind-driven convection increases h linearly
            h_eff = h * (1.0 + 0.5 * self.wind_speed)
            q_conv = h_eff * (self.ambient_temp_k - obj.temp_k) * dt_seconds
            
            # Phase 5: Evaporative cooling for wet objects
            if obj.is_wet:
                # Simplified evaporative heat loss
                q_conv -= (10.0 * (1.0 + 0.2 * self.wind_speed)) * dt_seconds
                
            dE[name] += q_conv
            
        # Conduction between objects
        for (n1, n2), h in self.conduction_paths.items():
            obj1 = self.objects[n1]
            obj2 = self.objects[n2]
            q_cond = h * (obj2.temp_k - obj1.temp_k) * dt_seconds
            dE[n1] += q_cond
            dE[n2] -= q_cond
            
        # Apply energy changes to temperature based on thermal mass
        for name, energy_j in dE.items():
            obj = self.objects[name]
            # Phase 5: Wet surfaces have effectively higher thermal mass (water cp is high)
            eff_c = obj.c * 2.0 if obj.is_wet else obj.c
            dT = energy_j / (obj.mass * eff_c)
            obj.temp_k += dT
            
            # Phase 4: Step internal dynamics (e.g. 1D terrain layer propagation)
            if hasattr(obj, 'layer_temps'):
                obj.layer_temps[0] += dT
            obj.update_internal(dt_seconds)

    def get_temperatures(self):
        return {name: obj.temp_k for name, obj in self.objects.items()}
