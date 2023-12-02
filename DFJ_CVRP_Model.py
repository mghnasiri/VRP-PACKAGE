""" Contains the core logic of your model.
CVRP-TW
This file  include functions or classes that define your model's behavior, calculations, data processing, etc. """
# model.py
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import math

# Function to create and initialize the graph


def create_graph(data_subset, num_data_points):
    n = len(data_subset) - 1
    # directed graph with a vertex for each city
    G = nx.complete_graph(num_data_points, nx.DiGraph())
    # Add any additional logic for graph initialization
    return G


def eucl_dist(x1, y1, x2, y2):
    return (math.sqrt((x1-x2)**2 + (y1-y2)**2))


def numVehiclesNeededForCustomers(G,Q,q):
    sumDemand = 0
    for i in G.nodes:
        sumDemand += q[i]
    return math.ceil(sumDemand / Q)



def solve_MTZ_VRP_problem(G,depot, max_vehicles,q,num_data_points,Q,time_windows,service_times,dem_points,dataset_name_with_extension):
    
    customers = [*range(1, num_data_points + 1)]  
    locations = [depot] + customers   
    connections = [(i, j) for i in locations for j in locations if i != j]

    
    # Model
    m = gp.Model("DFJ_CVRP")
    
    # Decision variables
    x = m.addVars(G.edges, vtype=GRB.BINARY)                                # x_ij: 1 if route from i to j is used, 0 otherwise
    u = m.addVars(G.nodes, vtype=GRB.CONTINUOUS)
    T = m.addVars(G.nodes, vtype=GRB.CONTINUOUS, lb=0, name="T")
    
    
    # Objective: Minimize the total travel cost
    m.setObjective(gp.quicksum(G.edges[i, j]['length'] * x[i, j] for i, j in G.edges), GRB.MINIMIZE)
    
    # all customers have exactly one incoming and one outgoing connection
    m.addConstrs((x.sum("*", j) == 1 for j in customers ), name="incoming")
    m.addConstrs((x.sum(i, "*") == 1 for i in customers ), name="outgoing")
    
    # vehicle limits
    m.addConstr(x.sum(0, "*") <= max_vehicles, name="maxNumVehicles")
    m.addConstr(x.sum(0, "*") >= numVehiclesNeededForCustomers(G,Q,q),name="minNumVehicles",)
    
     # Add the MTZ variable
    u = m.addVars(G.nodes)
    c = m.addConstrs(u[i] - u[j] + len(G.nodes) * x[i, j] <=
                     len(G.nodes) - 1 for i, j in G.edges if j != depot)
    


    
    
    
   
    
    
    time_limit = 300  # for example, 60 seconds
    m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam('LogFile', 'gurobi.log') 
    m.optimize()
    
    output_file_path = f"/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/{dataset_name_with_extension}.sol"

    m.write(output_file_path)


    display_optimal_solution(x)

    return m

def display_optimal_solution(x):
    print("Optimal Routes:")
    for (i, j), var in x.items():
        if var.X > 0.9:  # Threshold to determine if the route is used
            print(f"Route from {i} to {j} is part of the solution.")
 # Assuming the model has been solved and x contains the solution
def get_optimization_results(model):
    """
    Extracts key information from a Gurobi optimization model.
    """
    results = {
        'Optimal Value': None,
        'Number of Iterations': None,
        'Runtime (seconds)': None,
        'Status': None
    }

    if model.status == GRB.OPTIMAL:
        results['Optimal Value'] = model.ObjVal
        results['Number of Iterations'] = model.IterCount
        results['Runtime (seconds)'] = model.Runtime
        results['MIP Gap'] = model.MIPGap if model.IsMIP else 'N/A'  # Set MIP Gap
        results['Status'] = 'Optimal'
    elif model.status == GRB.TIME_LIMIT:
        results['Optimal Value'] = model.ObjVal
        results['Number of Iterations'] = model.IterCount
        results['Runtime (seconds)'] = model.Runtime
        results['MIP Gap'] = model.MIPGap if model.IsMIP else 'N/A'  # Set MIP Gap

        results['Status'] = 'Passed the time limit.'
        # You can still extract and return the best found solution here, if needed
    elif model.status == GRB.INF_OR_UNBD:
        # Handling infeasible or unbounded situation
        results['Status'] = 'Infeasible or Unbounded'
       
        for c in model.getConstrs():
            if c.IISConstr:
                print('Infeasible constraint:', c.constrName)
       
        for v in model.getVars():
         if v.IISLB > 0 or v.IISUB > 0:
             print('Infeasible bound:', v.varName)
    
    else:
        results['Status'] = 'Not Optimal'
        model.computeIIS()


    return results


def get_dimension_from_tsp(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("DIMENSION"):
                # Extract the dimension value
                _, dimension = line.split(':')
                return int(dimension.strip())

    return None
