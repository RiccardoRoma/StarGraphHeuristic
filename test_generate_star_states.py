import generate_states
from Qiskit_input_graph import draw_graph, calculate_msq
import pickle
import networkx as nx
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
#test for old version of function 
# edges_list = [(0,2), (1,3), (1,5), (2,5), (2,7), (2,9), (3,6), (3,8), (4,5), (5,9)]
# center_list = [2, 5, 3]
# 
# c = generate_star_states.generate_star_states(edges_list, center_list, add_barries=True)
# 
# 
fname = "./Saved_small_random_graphs/random_graph_n10_p0.1_erdos_renyi_copy_1.pkl"
# load initial random graph from pickle file
G = None
with open(fname, "rb") as f:
    G = pickle.load(f)

# Calculate subgraphs and merging sequence
out = calculate_msq(G, show_status=False)
subgraphs = out[1]
print("merging edges list {}".format(out[0]))
cnt = 0
for g in subgraphs:
    print("graph {} nodes: {}".format(cnt, g.nodes))
    draw_graph(g, show=False)
    c = QuantumCircuit(len(G.nodes))
    c = generate_states.generate_graph_state(g, c)
    
    print(c.draw(output="mpl"))
    
    cnt += 1

plt.show()