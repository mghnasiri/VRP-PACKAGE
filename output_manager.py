# output_manager.py
import matplotlib.pyplot as plt
import pandas as pd
import textwrap

# Function for visualizing the graph and displaying results in a table

# Function to reconstruct the tour from the solution
def reconstruct_tour(G, depot, x):
    tour = [depot]
    while True:
        next_city_found = False
        for j in G.successors(tour[-1]):
            var = x.get((tour[-1], j))
            if var is not None and hasattr(var, 'X') and var.X > 0.9:
                tour.append(j)
                next_city_found = True
                break
        if not next_city_found or tour[-1] == depot:
            break
    return tour


# Helper function to wrap text
def wrap_text(text, char_limit):
    words = text.split(' ')
    wrapped_text = ""
    line = ""
    for word in words:
        if len(line + word) <= char_limit:
            line += word + ' '
        else:
            wrapped_text += line + '\n'
            line = word + ' '
    wrapped_text += line
    return wrapped_text

def visualize_graph(G, depot, nx, x, my_pos, results, dataset_name_with_extension):
    # Create a figure with three subplots arranged vertically
    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 0.5, 1]})
    fig.subplots_adjust(hspace=0.5, top=0.93) # Adjust the space between subplots
    fig.canvas.manager.set_window_title(dataset_name_with_extension)


    # Set titles and turn off axes for the non-graph subplots
    ax2.set_title('Optimization Results Table')
    ax2.axis('off')
    ax3.set_title('Optimal Tour')
    ax3.axis('off')

    # Display the optimal tour in the middle subplot
    tour = reconstruct_tour(G, depot, x)
    tour_str = ' -> '.join(map(str, tour)) + ' (Total nodes: ' + str(len(G.nodes())) + ')'
    wrapped_tour_str = wrap_text(tour_str, 80)  # Adjust the character limit as needed

    # Positioning the wrapped text within ax3
    ax3.annotate(wrapped_tour_str, xy=(0.5, 0.3), xycoords='axes fraction', ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    ax3.set_title('Optimal Tour', pad=20)
    ax3.axis('off')
    # Now the first subplot (ax1) is used for graph visualization
    # The second subplot (ax2) is used for the table
    # The third subplot (ax3) is used for displaying the optimal tour
    if results['Optimal Value'] != float('inf'):
        tour_edges = [e for e in G.edges if x[e].x > 0.9]
        node_colors = ["red" if node == depot else "yellow" for node in G.nodes()]
        node_sizes = [100 if node == depot else 50 for node in G.nodes()]
        num_edges = len(tour_edges)
        num_nodes = len(G.nodes())
        # Visualizing the graph
        nx.draw(G, pos=my_pos, ax=ax1, edgelist=tour_edges, node_color=node_colors, node_size=node_sizes,  
                 edge_color='green', width=2.0, with_labels=True)
        ax1.set_title(f'Graph with {num_nodes} nodes', pad=20)

    # Constructing and displaying the table
    table = ax2.table(cellText=[list(results.values())], colLabels=list(results.keys()), cellLoc='center', loc='center')
    #table.auto_set_font_size(False)
    table.set_fontsize(12)
    #table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.show()
