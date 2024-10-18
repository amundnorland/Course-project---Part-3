import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

# Constants
constants = {
    "time_periods": 24,
    "energy_sources": ["Grid", "Solar", "Battery"],
    "scenarios": ["S_high", "S_avg", "S_low"],
    #####CHANGING THESE VALUES WHEN SOLVING FOR EACH SCENARIO INDIVIDUALLY#####
    "probs": {"S_high": 0.3, "S_avg": 0.5, "S_low": 0.2}
}

# Input Data (generate random grid cost for each time period)
data = {
    'Cost_grid': [5.2, 6.0, 5.5, 5.8, 6.2, 7.1, 9.3, 12.5, 14.1, 13.8, 
                12.9, 11.5, 10.4, 9.7, 8.5, 9.0, 10.1, 11.8, 12.3, 
                14.0, 13.5, 10.8, 7.9, 6.7],
    'Cost_solar': 0,  # Solar has zero marginal cost
    'Cost_battery': 10,
    'aFRR_price': 10,  # aFRR price 
    'Energy_demand': [20] * constants['time_periods'],  # Constant demand for simplicity
    
    'Solar_generation': {  # Solar generation also varies
        'S_high': [2] * constants['time_periods'],  # High solar generation
        'S_avg': [1.2] * constants['time_periods'],  # Average solar generation
        'S_low': [0.5] * constants['time_periods'],  # Low solar generation
    },
    'Battery_charge_eff': 0.9,
    'Battery_discharge_eff': 0.9,
    'Max_battery_capacity': 5,
    'Max_battery_charge': 1,
    'Max_battery_discharge': 1,
    'Grid_capacity': 20,
    'Initial_battery_storage': 2
}
#Define the model setup
def StochasticModelSetUp(data, constants):
    # Create a concrete model
    m = pyo.ConcreteModel()

    # Sets
    m.T = pyo.RangeSet(1, constants["time_periods"])  # Time periods
    m.I = pyo.Set(initialize=constants["energy_sources"])  # Energy sources
    m.S = pyo.Set(initialize=constants["scenarios"])  # Scenarios

    # Parameters
    m.C_grid = pyo.Param(m.T, initialize={t: data['Cost_grid'][t-1] for t in m.T})
    m.C_solar = pyo.Param(initialize=data['Cost_solar'])  # Solar cost is constant
    m.C_battery = pyo.Param(initialize=data['Cost_battery'])  # Battery cost is constant
    m.C_exp = pyo.Param(m.T, initialize={t: 0.9 * data['Cost_grid'][t-1] for t in m.T})
    m.P_aFRR = pyo.Param(m.T, initialize=data['aFRR_price'])  # aFRR price
    
    m.D = pyo.Param(m.T, initialize={t: data['Energy_demand'][t-1] 
    for t in range(1, constants["time_periods"] + 1)})
    
    m.G_solar = pyo.Param(m.T, m.S, initialize={(t, s): data['Solar_generation'][s][t-1] 
    for t in range(1, constants["time_periods"] + 1) for s in constants["scenarios"]})
    
    m.eta_charge = pyo.Param(initialize=data['Battery_charge_eff'])
    m.eta_discharge = pyo.Param(initialize=data['Battery_discharge_eff'])
    m.E_max = pyo.Param(initialize=data['Max_battery_capacity'])
    m.P_charge_max = pyo.Param(initialize=data['Max_battery_charge'])
    m.P_discharge_max = pyo.Param(initialize=data['Max_battery_discharge'])
    m.G_max = pyo.Param(initialize=data['Grid_capacity'])
    m.I_INIT = pyo.Param(initialize=data['Initial_battery_storage'])
    m.pi = pyo.Param(m.S, initialize=constants["probs"])  # Probability of each scenario

    # Variables
    m.x_aFRR = pyo.Var(m.T, within=pyo.NonNegativeReals)  # aFRR reserve 
    # Energy supply from sources with bounds
    m.y_supply = pyo.Var(m.T, m.S, m.I, within=pyo.NonNegativeReals, bounds=(0, m.G_max)) 
    m.z_export = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Energy exported to the grid
    m.q_charge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery charge
    m.q_discharge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery discharge
    # Battery energy storage (shared across scenarios)
    m.e_storage = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, m.E_max)) 


#Mathematical formulation 1st stage
def Obj_first_stage(m):
    return -sum(m.x_aFRR[t] * m.P_aFRR[t] for t in m.T) + m.alpha
m.obj = pyo.Objective(rule=Obj_first_stage, sense=pyo.minimize)

def EnergyBalance(m, t, s):
    return m.D[t] + m.x_aFRR[t] == sum(m.y_supply[t, s, i] for i in m.I) - m.z_export[t, s] + m.q_discharge[t, s] - m.eta_charge * m.q_charge[t, s]
m.EnergyBalance = pyo.Constraint(m.T, m.S, rule=EnergyBalance)

def ReserveMarketLimit(m, t, s):
    return m.x_aFRR[t] <= m.q_charge[t,s] + m.q_discharge[t,s]
m.ReserveMarketLimit = pyo.Constraint(m.T,m.S, rule=ReserveMarketLimit)

def ExportLimit(m, t, s):
    return m.z_export[t, s] + m.x_aFRR[t] <= m.G_max
m.ExportLimit = pyo.Constraint(m.T, m.S, rule=ExportLimit)

def CreateCuts(m,)




    # Objective Function: Expected cost over all scenarios
    def Obj(m):
        return -sum(m.x_aFRR[t] * m.P_aFRR[t] for t in m.T) + \
               sum(m.pi[s] * sum(
                   m.y_supply[t, s, 'Grid'] * m.C_grid[t] + \
                   m.y_supply[t, s, 'Battery'] * m.C_battery + \
                   m.y_supply[t, s, 'Solar'] * m.C_solar - \
                   m.z_export[t, s] * m.C_exp[t]
                   for t in m.T) for s in m.S)
    m.obj = pyo.Objective(rule=Obj, sense=pyo.minimize)


    # Constraints
    def EnergyBalance(m, t, s):
        return m.D[t] + m.x_aFRR[t] == sum(m.y_supply[t, s, i] for i in m.I) - m.z_export[t, s] + m.q_discharge[t, s] - m.eta_charge * m.q_charge[t, s]
    m.EnergyBalance = pyo.Constraint(m.T, m.S, rule=EnergyBalance)

    def ReserveMarketLimit(m, t, s):
        return m.x_aFRR[t] <= m.q_charge[t,s] + m.q_discharge[t,s]
    m.ReserveMarketLimit = pyo.Constraint(m.T,m.S, rule=ReserveMarketLimit)

    def StorageDynamics(m, t, s):
        if t == 1:
            return m.e_storage[t] == m.I_INIT
        else:
            return m.e_storage[t] == m.e_storage[t-1] + m.q_charge[t-1, s] - m.q_discharge[t-1, s] / m.eta_discharge

    m.StorageDynamics = pyo.Constraint(m.T, m.S, rule=StorageDynamics)

    def BatteryLimits(m, t):
        return m.e_storage[t] <= m.E_max
    m.BatteryLimits = pyo.Constraint(m.T, rule=BatteryLimits)

    def ChargeLimit(m, t, s):
        return m.q_charge[t, s] <= m.P_charge_max
    m.ChargeLimit = pyo.Constraint(m.T, m.S, rule=ChargeLimit)

    def DischargeLimit(m, t, s):
        return m.q_discharge[t, s] <= m.P_discharge_max
    m.DischargeLimit = pyo.Constraint(m.T, m.S, rule=DischargeLimit)

    def BatterySupplyLimit(m, t, s):
        return m.y_supply[t, s, 'Battery'] <= m.eta_discharge * m.e_storage[t]
    m.BatterySupplyLimit = pyo.Constraint(m.T, m.S, rule=BatterySupplyLimit)

    def ImportLimit(m, t, s):
        return m.y_supply[t, s, 'Grid'] <= m.G_max
    m.ImportLimit = pyo.Constraint(m.T, m.S, rule=ImportLimit)

    def SolarPowerLimit(m, t, s):
        return m.y_supply[t, s, 'Solar'] == m.G_solar[t, s]
    m.SolarPowerLimit = pyo.Constraint(m.T, m.S, rule=SolarPowerLimit)

    def ExportLimit(m, t, s):
        return m.z_export[t, s] + m.x_aFRR[t] <= m.G_max
    m.ExportLimit = pyo.Constraint(m.T, m.S, rule=ExportLimit)


# Solve the model
def SolveModel(m):
    solver = SolverFactory('gurobi')
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = solver.solve(m, tee=True)
    return results, m

# Display results
def DisplayResults(m):
    return print(m.display(), m.dual.display())

m = StochasticModelSetUp(data, constants)
SolveModel(m)
DisplayResults(m)