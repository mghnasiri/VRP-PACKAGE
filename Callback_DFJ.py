""" Contains the core logic of your model.
CVRP-TW
This file  include functions or classes that define your model's behavior, calculations, data processing, etc. """
# model.py
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import math
import time
import matplotlib.pyplot as plt
from output_manager import visualize_graph


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



def find_subtours(solution_edges,m):
    """Function to find subtours in the given set of edges"""
    G = nx.DiGraph()
    G.add_edges_from(solution_edges)
    return list(nx.simple_cycles(G))


def solve_DFJ_CVRP_problem(G,depot, max_vehicles,q,num_data_points,Q,time_windows,service_times,dem_points,dataset_name_with_extension,my_pos):
    start_time = time.time()  # Start time of the function
    time_limit=300
    current_time1 = time.time()
    customers = [*range(1, num_data_points + 1)]  
    locations = [depot] + customers   
    connections = [(i, j) for i in locations for j in locations if i != j]
    # First, solve a relaxation

    m = gp.Model()
    x = m.addVars(G.edges,vtype=GRB.BINARY)
    
    m.setObjective( gp.quicksum( G.edges[i,j]['length'] * x[i,j] for i,j in G.edges ), GRB.MINIMIZE )
    # all customers have exactly one incoming and one outgoing connection
    m.addConstrs((x.sum("*", j) == 1 for j in customers ), name="incoming")
    m.addConstrs((x.sum(i, "*") == 1 for i in customers ), name="outgoing")
    # vehicle limits
    m.addConstr(x.sum(0, "*") <= max_vehicles, name="maxNumVehicles")
    m.addConstr(x.sum(0, "*") >= numVehiclesNeededForCustomers(G,Q,q),name="minNumVehicles",)
    
    m.update()

        # create a function to separate the subtour elimination constraints
    def subtour_elimination(m, where):
          # Check if the callback is at the right stage
         if where == gp.GRB.Callback.MIPSOL:           
            # Get the current solution
            x_vals = m.cbGetSolution(m._x)
            tour_edges = [(i, j) for i, j in m._G.edges if x_vals[i, j] > 0.5]
            # Find subtours in the current solution
            subtours = find_subtours(tour_edges,m)
            # Add a subtour elimination constraint for each subtour found
            for subtour in subtours:
                 if len(subtour) <= m._G.number_of_nodes() / 2:
                     m.cbLazy(gp.quicksum(m._x[i, j] for i, j in tour_edges if i in subtour and j in subtour) <= len(subtour) - 1)

    
    # tell Gurobi that we will be adding (lazy) constraints
    m.Params.lazyConstraints = 1    
    # designate the callback routine to be subtour_elimination()
    m._callback = subtour_elimination   
    m._x = x
    m._G = G
    
 
    m.optimize(m._callback)
 
  
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
