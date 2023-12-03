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



def solve_VRP_TW_problem(G,depot, max_vehicles,q,num_data_points,Q,time_windows,service_times,dem_points,dataset_name_with_extension):
    
    customers = [*range(1, num_data_points + 1)]  
    locations = [depot] + customers   
    connections = [(i, j) for i in locations for j in locations if i != j]

    
    # Model
    m = gp.Model("MCF1d")
    
    # Decision variables
    x = m.addVars(connections, vtype=GRB.BINARY)                                # x_ij: 1 if route from i to j is used, 0 otherwise
    u = m.addVars(G.nodes, vtype=GRB.CONTINUOUS)
    T = m.addVars(G.nodes, vtype=GRB.CONTINUOUS, lb=0, name="T")
    f = m.addVars(connections,  G.nodes, vtype=GRB.CONTINUOUS, lb=0, name="f")      # f_ijk: flow of commodity k on route from i to j


    
    # Objective: Minimize the total travel cost
    m.setObjective(gp.quicksum(G.edges[i, j]['length'] * x[i, j] for i, j in G.edges), GRB.MINIMIZE)
    
    # all customers have exactly one incoming and one outgoing connection
    m.addConstrs((x.sum("*", j) == 1 for j in customers ), name="incoming")
    m.addConstrs((x.sum(i, "*") == 1 for i in customers ), name="outgoing")
    # vehicle limits
    m.addConstr(x.sum(0, "*") <= max_vehicles, name="maxNumVehicles")
    m.addConstr(x.sum(0, "*") >= numVehiclesNeededForCustomers(G,Q,q),name="minNumVehicles",)

    z = m.addVars(connections, lb=0, ub=Q, name="z")

    for i in customers :
        z[0, i].UB = 0

    m.addConstrs(
        (z.sum("*", j) + q[j] == z.sum(j, "*") for j in customers ),
        name="flowConservation",
    )
    m.addConstrs(
        (
            z[i, j] >= q[i] * x[i, j]
            for i in customers
            for j in locations
            if i != j
        ),
        name="loadLowerBound",
    )
    m.addConstrs(
        (
            z[i, j] <= (Q - q[j]) * x[i, j]
            for i in customers
            for j in locations
            if i != j
        ),
        name="loadUpperBound",
    )
    
    
    """ y=m.addVars(connections, lb=0, name="y")
    
    for (i, j) in connections:
        y[i, j].UB = time_windows[i][1]

    m.addConstrs(
        (
            gp.quicksum(
                y[i, j] + (service_times[i] + G.edges[i, j]['length']) * x[i, j]
                for i in locations
                if (i, j) in connections
            )
            <= y.sum(j, "*")
            for j in customers
        ),
        name="flowConservation",
    )
    m.addConstrs(
        (
            y[i, j] >= time_windows[i][0] * x[i, j]
            for i in customers
            for j in locations
            if i != j
        ),
        name="timeWindowStart",
    )
    m.addConstrs(
        (
            y[i, j] <= time_windows[i][1] * x[i, j]
            for i in customers
            for j in locations
            if i != j
        ),
        name="timeWindowEnd",
    ) """

    
    
    
    
    # Subtour Elimination Constraints
    #m.addConstrs(u[i] - u[j] + len(G.nodes) * x[i, j] <=len(G.nodes) - 1 for i, j in G.edges if j != depot)
    # Configure the model to find multiple solutions
    #m.setParam(GRB.Param.PoolSolutions, 10)  # Store the 10 best solutions
    # Search for more than one optimal solution
    #m.setParam(GRB.Param.PoolSearchMode, 2)
    # Set the time limit (in seconds)
    time_limit = 300  # for example, 60 seconds
    m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam('LogFile', 'gurobi.log')

    
    m.optimize()
    
    output_file_path = f"/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/{dataset_name_with_extension}.sol"

    m.write(output_file_path)


    display_optimal_solution(x)

    return m


#MTZ FOR CVRP
def solve_MTZ_CVRP_problem(G,depot, max_vehicles,q,num_data_points,Q,time_windows,service_times,dem_points,dataset_name_with_extension):

    customers = [*range(1, num_data_points + 1)]  
    locations = [depot] + customers   
    connections = [(i, j) for i in locations for j in locations if i != j]
    
    # Model
    m = gp.Model("MTZ_CVRP")
    
    
    # Decision variables
    x = m.addVars(G.edges, vtype=GRB.BINARY)                                # x_ij: 1 if route from i to j is used, 0 otherwise
    u = m.addVars(G.nodes)
    
    # Objective: Minimize the total travel cost
    m.setObjective(gp.quicksum(G.edges[i, j]['length'] * x[i, j] for i, j in G.edges), GRB.MINIMIZE)
   
    # Enter each demand point once
    m.addConstrs( gp.quicksum( x[i,j] for i in G.predecessors(j) ) == 1 for j in dem_points )

    # Leave each demand point once
    m.addConstrs( gp.quicksum( x[i,j] for j in G.successors(i) ) == 1 for i in dem_points )

    # Leave the depot k times
    m.addConstr( gp.quicksum( x[depot,j] for j in G.successors(depot) ) == max_vehicles )
    u[depot].LB = 0
    u[depot].UB = 0

    for i in dem_points:
        u[i].LB = q[i]
        u[i].UB = Q

    m.addConstrs( u[i] - u[j] + Q * x[i,j] <= Q - q[j] for i,j in G.edges if j != depot )

    
    

    
    
    
   
    
    
    time_limit = 300  # for example, 60 seconds
    m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam('LogFile', 'gurobi.log') 
    m.optimize()
    
    output_file_path = f"/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/{dataset_name_with_extension}.sol"

    m.write(output_file_path)


    display_optimal_solution(x)

    return m



def find_subtours(solution_edges):
    """Function to find subtours in the given set of edges"""
    G = nx.DiGraph()
    G.add_edges_from(solution_edges)
    return list(nx.simple_cycles(G))



def solve_DFJ_CVRP_problem(G,depot, max_vehicles,q,num_data_points,Q,time_windows,service_times,dem_points,dataset_name_with_extension,my_pos):
    start_time = time.time()  # Start time of the function
    time_limit=60000
    customers = [*range(1, num_data_points + 1)]  
    locations = [depot] + customers   
    connections = [(i, j) for i in locations for j in locations if i != j]
    # First, solve a relaxation

    m = gp.Model()
    x = m.addVars(G.edges,vtype=GRB.BINARY)
    
    m.setObjective( gp.quicksum( G.edges[i,j]['length'] * x[i,j] for i,j in G.edges ), GRB.MINIMIZE )
    
    """ # Enter each demand point once
    m.addConstrs( gp.quicksum( x[i,j] for i in G.predecessors(j) ) == 1 for j in dem_points )
    
    # Leave each demand point once
    m.addConstrs( gp.quicksum( x[i,j] for j in G.successors(i) ) == 1 for i in dem_points )
    """
    # Leave the depot k times
    #m.addConstr( gp.quicksum( x[depot,j] for j in G.successors(depot) ) == max_vehicles )
    #m.optimize() 
# all customers have exactly one incoming and one outgoing connection
    m.addConstrs((x.sum("*", j) == 1 for j in customers ), name="incoming")
    m.addConstrs((x.sum(i, "*") == 1 for i in customers ), name="outgoing")
    # vehicle limits
    m.addConstr(x.sum(0, "*") <= max_vehicles, name="maxNumVehicles")
    m.addConstr(x.sum(0, "*") >= numVehiclesNeededForCustomers(G,Q,q),name="minNumVehicles",)

    """ tour_edges = [ e for e in G.edges if x[e].x > 0.5 ]
    
    while not nx.is_connected( G.edge_subgraph(tour_edges) ):
    
        for component in nx.connected_components(G.edge_subgraph(tour_edges)):
            print("Adding constraint for this component:",component)
            inner_edges = [ (i,j) for (i,j) in G.edges if i in component and j in component ]
            m.addConstr( gp.quicksum( x[e] for e in inner_edges ) <= len(component) - 1 )

        m.optimize() """

    #c = m.addConstrs( u[i] - u[j] + Q * x[i,j] <= Q - q[j] for i,j in G.edges if j != depot )
    
   
    # Solve initial VRP model
    m.optimize()
    
    solution_edges = [ e for e in G.edges if x[e].x > 0.5 ]
   
    file_path = f"/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/VRP PACKAGE/300 SECONDS RESULTS DFJ-CVRP/R101/{m.ObjVal}.png"
    results = get_optimization_results(m)
    x_vars = m.getVars()
    x = {e: x_var for e, x_var in zip(G.edges, x_vars)}
    visualize_graph(G,depot, nx, x, my_pos, results,dataset_name_with_extension,file_path)
    
    # Subtour elimination process
    while True:
        current_time = time.time()
        if current_time - start_time > time_limit:
            print("Time limit exceeded.")
            break
        
        m.optimize()
        
        file_path = f"/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/VRP PACKAGE/300 SECONDS RESULTS DFJ-CVRP/R101/{m.ObjVal}.png"
        results = get_optimization_results(m)
        x_vars = m.getVars()
        x = {e: x_var for e, x_var in zip(G.edges, x_vars)}
        visualize_graph(G,depot, nx, x, my_pos, results,dataset_name_with_extension,file_path)

        solution_edges = [(i, j) for i, j in G.edges if x[i, j].X > 0.5]
        subtours = find_subtours(solution_edges)

        if not any(len(subtour) < len(G.nodes) for subtour in subtours):
            break  # No subtours found

        for subtour in subtours:
            if len(subtour) < len(G.nodes):
                m.addConstr(gp.quicksum(x[i, j] for i, j in solution_edges if i in subtour and j in subtour) <= len(subtour) - 1)

        
    
   
    
    
    #time_limit = 300  # for example, 60 seconds
    #m.setParam(GRB.Param.TimeLimit, time_limit)
    #m.optimize()

    #m.setParam('LogFile', 'gurobi.log') 
    
    #output_file_path = f"/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/VRP PACKAGE/300 SECONDS RESULTS DFJ-CVRP/{dataset_name_with_extension}.sol"

    #m.write(output_file_path)
    #display_optimal_solution(x)

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
