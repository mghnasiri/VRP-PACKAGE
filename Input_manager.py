import networkx as nx
import os
import pandas as pd
from model import  eucl_dist,solve_VRP_TW_problem,get_optimization_results
from output_manager import visualize_graph




def main():

    # Load the list of dataset
    dataset_paths = [                     
                     #Solomon DataSet
                      '/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R101.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R102.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R103.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R104.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R105.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R106.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R107.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R108.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R109.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R110.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R111.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/R1/R112.csv'
                 
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C101.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C102.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C103.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C104.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C105.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C106.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C107.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C108.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/C1/C109.csv'
                     
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC101.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC102.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC103.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC104.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC105.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC106.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC107.csv'
                     #'/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/Datasets/Solomon/solomon_dataset/RC1/RC108.csv'
                     ]
    
    for data_path in dataset_paths:
        dataset_name_with_extension = os.path.basename(data_path)

        
        # Variables
        k = 1      # number of vehicles
        Q = 200     # capacity of vehicles
        num_data_points = 24  # number of demand points
        depot = 0

        
        data = pd.read_csv(data_path)
        data_subset = data.head(num_data_points + 1)  # +1 to include the depot

        # Create position dictionary
        my_pos = {index: (row['XCOORD.'], row['YCOORD.']) for index, row in data_subset.iterrows()}

        G = nx.complete_graph(num_data_points + 1, nx.DiGraph())

        for i, j in G.edges:
            (x1, y1) = my_pos[i]
            (x2, y2) = my_pos[j]
            G.edges[i, j]['length'] = eucl_dist(x1, y1, x2, y2)

        for i, j in G.edges:
            print(f"Edge ({i}, {j}): Length = {G.edges[i, j]['length']:.2f}")
 

        q = {index: row['DEMAND'] for index, row in data_subset.iterrows()}
        
        print(q)
        
        model = solve_VRP_TW_problem(G, k,q,num_data_points,Q,time_windows,travel_times,service_times)
        
        
        # Assuming model is the returned Gurobi model from solve_TSP_MTZ_problem
        x_vars = model.getVars()
        x = {e: x_var for e, x_var in zip(G.edges, x_vars)}
        results = get_optimization_results(model)
        #output_file_path = f"/home/centor.ulaval.ca/ghafomoh/Downloads/ADM-7900/{dataset_name_with_extension}.png"
        # If graph visualization is needed
        visualize_graph(G,depot, nx, x, my_pos, results,dataset_name_with_extension)
 


        
        
        
    

if __name__ == "__main__":
    main()

        
        
 

        

