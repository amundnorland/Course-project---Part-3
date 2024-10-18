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

#Master problem: Define the model setup
#Mathematical formulation first stage
def First_stage_model(data, constants, Cuts):
    m = pyo.ConcreteModel()

    # Sets
    m.T = pyo.RangeSet(1, constants["time_periods"])  # Time periods
    m.Cut = pyo.Set(initialize=Cuts["Set"])  # Set for cuts

    # Parameters
    m.P_aFRR = pyo.Param(m.T, initialize=data['aFRR_price'])  # aFRR price
    m.G_max = pyo.Param(initialize=data['Grid_capacity'])  # Grid capacity limit
    # parameters for cuts:
    m.Phi = pyo.Param(m.Cut, initialize = Cuts["Phi"])
    m.Lambda = pyo.Param(m.Cut, m.T, initialize = Cuts["lambda"])
    m.x_hat = pyo.Param(m.Cut, m.T, initialize = Cuts["x_hat"])

    # Variables
    m.x_aFRR = pyo.Var(m.T, within=pyo.NonNegativeReals)  # aFRR reserve (first-stage decision)
    m.alpha = pyo.Var(bounds=(0, 1000000))  # Approximates second-stage cost
    

    # Objective: minimize first-stage cost with the approximation of the second-stage cost (alpha)
    def Obj_first_stage(m):
        return -sum(m.x_aFRR[t] * m.P_aFRR[t] for t in m.T) + m.alpha
    m.obj_first = pyo.Objective(rule=Obj_first_stage, sense=pyo.minimize)

    def ReserveMarketLimit_first_stage(m, t):
    # Constraining x_aFRR to be less than or equal to the maximum grid capacity at each time period
        return m.x_aFRR[t] <= m.G_max  
    m.ReserveMarketLimit_first_stage = pyo.Constraint(m.T, rule=ReserveMarketLimit_first_stage)



    # Create Benders Optimality cuts from first stage that is ent to second stage
    def Optimality_cut(m, c):
        a = 2
        b = "print"
        print(m.Phi[c])
        return(m.alpha >= m.Phi[c] + sum(m.Lambda[c,t]*(m.x_aFRR[t]-m.x_hat[c,t]) for t in m.T))
    m.Optimality_cut = pyo.Constraint(m.Cut, rule=Optimality_cut)

   #def Feasibility_cut(m,c):
     #  return m.alpha >= 0
   # m.Feasibility_cut = pyo.Constraint(m.Cut, rule=Feasibility_cut)

    return m

#Subproblem: Define model setup
#Mathematical formulation 2nd stage
def Second_stage_model(data, constants, Cuts):
    
    m = pyo.ConcreteModel()

    # Sets
    m.T = pyo.RangeSet(1, constants["time_periods"])  # Time periods
    m.I = pyo.Set(initialize=constants["energy_sources"])  # Energy sources
    m.S = pyo.Set(initialize=constants["scenarios"])  # Scenarios
    m.Cut = pyo.Set(initialize = Cuts["Set"]) #Set for cuts


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
    #Parameter for cuts
    m.Phi = pyo.Param(m.Cut, initialize = Cuts["Phi"])
    m.Lambda = pyo.Param(m.Cut, m.T, initialize = Cuts["lambda"])
    m.x_hat = pyo.Param(m.Cut, m.T, initialize = Cuts["x_hat"])

    # Variables 
    # Energy supply from sources with bounds
    m.y_supply = pyo.Var(m.T, m.S, m.I, within=pyo.NonNegativeReals, bounds=(0, m.G_max)) 
    m.z_export = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Energy exported to the grid
    m.q_charge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery charge
    m.q_discharge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery discharge
    # Battery energy storage (shared across scenarios)
    m.e_storage = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, m.E_max))

    def Obj_second_stage(m):
        return sum(m.pi[s] * sum(
                    m.y_supply[t, s, 'Grid'] * m.C_grid[t] + \
                    m.y_supply[t, s, 'Battery'] * m.C_battery + \
                    m.y_supply[t, s, 'Solar'] * m.C_solar - \
                    m.z_export[t, s] * m.C_exp[t]
                    for t in m.T) for s in m.S)
    m.obj_second = pyo.Objective(rule=Obj_second_stage, sense=pyo.minimize)
    

    def EnergyBalance_sec(m, t, s):
        return m.D[t] + m.x_hat[t] == sum(m.y_supply[t, s, i] for i in m.I) - m.z_export[t, s] + m.q_discharge[t, s] - m.eta_charge * m.q_charge[t, s]
    m.EnergyBalance_sec = pyo.Constraint(m.T, m.S, rule=EnergyBalance_sec)

    def ReserveMarketLimit_sec(m, t, s):
        return m.x_hat[t] <= m.q_charge[t,s] + m.q_discharge[t,s]
    m.ReserveMarketLimit_sec = pyo.Constraint(m.T,m.S, rule=ReserveMarketLimit_sec)

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

    def ExportLimit_sec(m, t, s):
        return m.z_export[t, s] + m.x_hat[t] <= m.G_max
    m.ExportLimit_sec = pyo.Constraint(m.T, m.S, rule=ExportLimit_sec)

    def Lambda_constraint(m, t):
        return m.x_aFRR[t] == m.x_hat[t]
    m.Lambda_constraint = pyo.Constraint(m.T, rule=Lambda_constraint)
    return m


# Solve the model
def SolveModel(m):
    solver = SolverFactory('gurobi')
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = solver.solve(m, tee=True)
    return results, m

# Display results
def DisplayResults(m):
    return print(m.display(), m.dual.display())

# Function for creating new linear cuts for optimization problem (Inspired by code given as example for Benders decomposition in class)
def Create_cuts(Cuts,m):
    """Add new cut to existing dictionary of cut information"""
    
    #Find cut iteration by checking number of existing cuts
    cut_it = len(Cuts["Set"])
    #Add new cut to list, since 0-index is a thing this works well
    Cuts["Set"].append(cut_it)
    
    #Find 2nd stage cost result
    Cuts["Phi"][cut_it] = pyo.value(m.obj)
    #Find lambda x_hat for each type of grain
    for t in m.T:
        Cuts["lambda"][cut_it,t] = m.dual[m.Lambda_constraint[t]]
        Cuts["x_hat"][cut_it,t] = m.X_hat[t]
    return(Cuts)

#Setup for Benders decomposition - Perform this for x-iterations
Cuts = {}
Cuts["Set"] = []
Cuts["Phi"] = {}
Cuts["lambda"] = {}
Cuts["x_hat"] = {}

#Using a for loop for iteration
for i in range(10):

    #Solve 1st stage problem
    m_first_stage = First_stage_model(data, constants, Cuts)
    SolveModel(m_first_stage)
    

    #First stage result process with x_hat value
    X_hat = {f"t{t}": m_first_stage.x_aFRR[t] for t in range(1, 25)}
  
    
    #Printing first stage results
    print("Iteration",i)
    for i in X_hat:
        print(i,X_hat[i].value)
    input()
    
    #Setup and solve 2nd stage problem
    m_second_stage = Second_stage_model(data, constants, X_hat)
    SolveModel(m_second_stage)


    #Creating cuts for the first stage model
    Cuts = Create_cuts(Cuts,m_second_stage)
    
    #Print results for second stage
    print("Objective function:",pyo.value(m_second_stage.obj_second))
    print("Cut information acquired:")
    for component in Cuts:
        if component == "lambda" or component == "x_hat":
            for t in m_second_stage.T:
                print(component,t,Cuts[component][t])
        else:
            print(component,Cuts[component])
    input()
    
    #Performing a convergence check with upper and lower bound
    print("UB:",pyo.value(m_first_stage.alpha.value),"- LB:",pyo.value(m_second_stage.obj))
    input()
    
import matplotlib.pyplot as plt
import pyomo.environ as pyo
# (rest of your imports and model code...)

# Define the plotting function
def plot_variables_at_timestep(t, m_first_stage, m_second_stage, constants):
    # Collect variables from the first stage
    x_aFRR_vals = [pyo.value(m_first_stage.x_aFRR[t]) for t in m_first_stage.T]
    
    # Collect variables from the second stage
    scenarios = constants['scenarios']
    y_supply_vals = {
        source: [pyo.value(m_second_stage.y_supply[t, s, source]) for t in m_second_stage.T for s in scenarios]
        for source in constants['energy_sources']
    }
    z_export_vals = [pyo.value(m_second_stage.z_export[t, s]) for t in m_second_stage.T for s in scenarios]
    q_charge_vals = [pyo.value(m_second_stage.q_charge[t, s]) for t in m_second_stage.T for s in scenarios]
    q_discharge_vals = [pyo.value(m_second_stage.q_discharge[t, s]) for t in m_second_stage.T for s in scenarios]
    e_storage_vals = [pyo.value(m_second_stage.e_storage[t]) for t in m_second_stage.T]
    
    # Plot the values
    plt.figure(figsize=(10, 8))

    # Plot first stage variables
    plt.subplot(3, 2, 1)
    plt.plot(range(1, constants['time_periods'] + 1), x_aFRR_vals, label='x_aFRR')
    plt.title(f'aFRR Reserve (timestep {t})')
    plt.xlabel('Time')
    plt.ylabel('x_aFRR')
    plt.grid()

    # Plot energy supply (second stage)
    plt.subplot(3, 2, 2)
    for source, values in y_supply_vals.items():
        plt.plot(range(1, constants['time_periods'] + 1), values, label=f'y_supply ({source})')
    plt.title(f'Energy Supply (timestep {t})')
    plt.xlabel('Time')
    plt.ylabel('Supply')
    plt.legend()
    plt.grid()

    # Plot energy export (second stage)
    plt.subplot(3, 2, 3)
    plt.plot(range(1, constants['time_periods'] + 1), z_export_vals, label='z_export')
    plt.title(f'Energy Export (timestep {t})')
    plt.xlabel('Time')
    plt.ylabel('z_export')
    plt.grid()

    # Plot battery charge
    plt.subplot(3, 2, 4)
    plt.plot(range(1, constants['time_periods'] + 1), q_charge_vals, label='q_charge')
    plt.title(f'Battery Charge (timestep {t})')
    plt.xlabel('Time')
    plt.ylabel('q_charge')
    plt.grid()

    # Plot battery discharge
    plt.subplot(3, 2, 5)
    plt.plot(range(1, constants['time_periods'] + 1), q_discharge_vals, label='q_discharge')
    plt.title(f'Battery Discharge (timestep {t})')
    plt.xlabel('Time')
    plt.ylabel('q_discharge')
    plt.grid()

    # Plot battery storage
    plt.subplot(3, 2, 6)
    plt.plot(range(1, constants['time_periods'] + 1), e_storage_vals, label='e_storage')
    plt.title(f'Battery Storage (timestep {t})')
    plt.xlabel('Time')
    plt.ylabel('e_storage')
    plt.grid()

    plt.tight_layout()
    plt.show()

# After solving the models, you can plot the variables as follows:
m_first_stage = First_stage_model(data, constants, Cuts)
results_first, m_first_stage = SolveModel(m_first_stage)

X_hat = {f"t{t}": m_first_stage.x_aFRR[t] for t in range(1, 25)}

m_second_stage = Second_stage_model(data, constants, X_hat)
results_second, m_second_stage = SolveModel(m_second_stage)

#Plotting for a given timestep, for example t=1
plot_variables_at_timestep(t=1, m_first_stage, m_second_stage, constants)

