""" Contains the core logic of your model.
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


def solve_VRP_TW_problem(G,depot, k,q,num_data_points,Q,time_windows,service_times,dem_points):
    # Model
    m = gp.Model()
    
    # Decision variables
    x = m.addVars(G.edges, vtype=GRB.BINARY)
    u = m.addVars(G.nodes, vtype=GRB.CONTINUOUS)
    T = m.addVars(G.nodes, vtype=GRB.CONTINUOUS, lb=0, name="T")
    f = m.addVars(G.edges, vtype=GRB.CONTINUOUS, lb=0, ub=Q, name="f")

    

    
    # Objective: Minimize the total travel cost
    m.setObjective(gp.quicksum(G.edges[i, j]['length'] * x[i, j] for i, j in G.edges), GRB.MINIMIZE)
    
    # Constraints
    
    # Enter each demand point once
    m.addConstrs(gp.quicksum(x[i, j]
                 for i in G.predecessors(j)) == 1 for j in dem_points)

    # Leave each demand point once
    m.addConstrs(gp.quicksum(x[i, j]
                 for j in G.successors(i)) == 1 for i in dem_points)

    # Leave the depot k times
    m.addConstr(gp.quicksum(x[depot, j] for j in G.successors(depot)) == k)
    
    # Flow and capacity constraints
    
    for i, j in G.edges:
        m.addConstr(f[i, j] <= Q * x[i, j])
        #m.addConstr(sum(f[j, i] for j in G.nodes if j != i) - sum(f[i, j] for j in G.nodes if j != i) == 0)
     
    
    """ 
    # Time window and service time constraints
    M = 10000  # A large number
    for i, j in G.edges:
        if i != j:
            m.addConstr(T[i] + service_times[i] + G.edges[i, j]['length'] <= T[j] + M * (1 - x[i, j]))

    for i in G.nodes:
        m.addConstr(T[i] >= time_windows[i][0])
        m.addConstr(T[i] <= time_windows[i][1])

     """
   

    
    
    # Subtour Elimination Constraints
    m.addConstrs(u[i] - u[j] + len(G.nodes) * x[i, j] <=
                 len(G.nodes) - 1 for i, j in G.edges if j != depot)
    
    
    
    # Configure the model to find multiple solutions
    m.setParam(GRB.Param.PoolSolutions, 10)  # Store the 10 best solutions
    # Search for more than one optimal solution
    m.setParam(GRB.Param.PoolSearchMode, 2)
    # Set the time limit (in seconds)
    time_limit = 60  # for example, 60 seconds
    m.setParam(GRB.Param.TimeLimit, time_limit)
    
    m.optimize()
    return m


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
