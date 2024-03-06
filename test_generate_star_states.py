import generate_star_states
from Qiskit_input_graph import draw_graph, calculate_msq
import pickle
import networkx as nx
from qiskit import QuantumCircuit
# test for old version of function 
# edges_list = [(0,2), (1,3), (1,5), (2,5), (2,7), (2,9), (3,6), (3,8), (4,5), (5,9)]
# center_list = [2, 5, 3]
# 
# c = generate_star_states.generate_star_states(edges_list, center_list, add_barries=True)


fname = "./Saved_small_random_graphs/random_graph_n10_p0.1_erdos_renyi_copy_1.pkl"
# load initial random graph from pickle file
G = None
with open(fname, "rb") as f:
    G = pickle.load(f)

# Calculate subgraphs and merging sequence
out = calculate_msq(G, show_status=False)
subgraphs = out[1]


c = QuantumCircuit(len(G.nodes()))
generate_star_states.generate_star_state(subgraphs[0], c)

print(c.draw(output="mpl"))

draw_graph(subgraphs[0])

