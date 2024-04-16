import merge_graphs
import shift_center
from Qiskit_input_graph import draw_graph, calculate_msq
import pickle
import networkx as nx
from qiskit import QuantumCircuit
import modify_graph_objects as mgo
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
merging_edges_list, subgraphs, _ = calculate_msq(G, show_status=False)

print("merging edges list {}".format(merging_edges_list))


## Debug
print("subgraph index 0:")
print("current graph nodes {}".format(subgraphs[0].nodes))
draw_graph(subgraphs[0], show=False)
for i in range(1, len(subgraphs)):
    print("subgraph index {}:".format(i))
    print("merge between center {}".format(merging_edges_list[i-1]))
    print("current graph nodes {}".format(subgraphs[i].nodes))
    draw_graph(subgraphs[i])
##

c = QuantumCircuit(len(G.nodes()), 1)

curr_circ = c.copy()
curr_graph0 = subgraphs[0]
curr_center0 = mgo.get_graph_center(curr_graph0) # determine current center
# merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
new_center_tuple = merging_edges_list[0]
if new_center_tuple[0] in curr_graph0.nodes:
    new_center0 = new_center_tuple[0]
elif new_center_tuple[1] in curr_graph0.nodes:
    new_center0 = new_center_tuple[1]
else:
    raise ValueError("new centers {} don't contain a node of current graph {}".format(new_center_tuple, curr_graph0.nodes))
draw_graph(curr_graph0, show=False)

curr_circ, curr_graph0 = shift_center.shift_centers(curr_circ, curr_graph0, curr_center0, new_center0)


print(curr_circ.draw(output="mpl"))

draw_graph(curr_graph0, show=False)

curr_graph1 = subgraphs[1]
curr_center1 = mgo.get_graph_center(curr_graph1) # determine current center
# merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
new_center_tuple = merging_edges_list[0]
if new_center_tuple[0] in curr_graph1.nodes:
    new_center1 = new_center_tuple[0]
elif new_center_tuple[1] in curr_graph1.nodes:
    new_center1 = new_center_tuple[1]
else:
    raise ValueError("new centers {} don't contain a node of current graph {}".format(new_center_tuple, curr_graph1.nodes))
draw_graph(curr_graph1, show=False)

curr_circ, curr_graph1 = shift_center.shift_centers(curr_circ, curr_graph1, curr_center1, new_center1)


print(curr_circ.draw(output="mpl"))

draw_graph(curr_graph1, show=False)

curr_circ, curr_graph0, _ = merge_graphs.merge_graphs(curr_circ, new_center0, curr_graph0, new_center1, curr_graph1, 0)

print(curr_circ.draw(output="mpl"))
draw_graph(curr_graph0)

