import shift_center
import generate_states
from Qiskit_input_graph import draw_graph, calculate_msq
import pickle
import networkx as nx
from qiskit import QuantumCircuit
import modify_graph_objects as mgo
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
import numpy as np
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
merging_edges_list, subgraphs, _ = calculate_msq(G, show_status=False)

print("merging edges list {}".format(merging_edges_list))

c = QuantumCircuit(len(G.nodes()))
fig_cnt = 1
curr_circ = c.copy()
curr_graph = subgraphs[0]
curr_center = mgo.get_graph_center(curr_graph) # determine current center
# merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
new_center_tuple = merging_edges_list[0]
if new_center_tuple[0] in curr_graph.nodes:
    new_center = new_center_tuple[0]
elif new_center_tuple[1] in curr_graph.nodes:
    new_center = new_center_tuple[1]
else:
    raise ValueError("new centers {} don't contain a node of current graph {}".format(new_center_tuple, curr_graph.nodes))
print("Figure {}: subgraph 1 original".format(fig_cnt))
draw_graph(curr_graph, show=False)
fig_cnt += 1

curr_circ = generate_states.generate_graph_state(curr_graph, curr_circ)

print("Figure {}: circuit to generate subgraph 1 original".format(fig_cnt))
print(curr_circ.draw(output="mpl"))
fig_cnt += 1

curr_circ, curr_graph = shift_center.shift_centers(curr_circ, curr_graph, curr_center, new_center)

print("Figure {}: circuit to generate subgraph 1 center shifted".format(fig_cnt))
print(curr_circ.draw(output="mpl"))
fig_cnt += 1

print("Figure {}: subgraph 1 center shifted".format(fig_cnt))
draw_graph(curr_graph, show=False)
fig_cnt += 1

curr_circ_ref = c.copy()
curr_circ_ref = generate_states.generate_graph_state(curr_graph, curr_circ_ref)

print("Figure {}: reference circuit for subgraph 1 center shifted".format(fig_cnt))
print(curr_circ_ref.draw(output="mpl"))

# compare current circuit with current reference
state_simulator = StatevectorSimulator()
job = state_simulator.run(curr_circ)
curr_state = Statevector(job.result().get_statevector(curr_circ))

job_ref = state_simulator.run(curr_circ_ref)
curr_state_ref = Statevector(job_ref.result().get_statevector(curr_circ_ref))

overlap = np.abs(curr_state.inner(curr_state_ref))
print("Overlap with reference: {}".format(overlap))

plt.show()

for i in range(1,len(subgraphs)):
    curr_graph = subgraphs[i]
    curr_circ = c.copy()

    curr_center = mgo.get_graph_center(curr_graph) # determine current center
    # merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
    new_center_tuple = merging_edges_list[i-1]
    if new_center_tuple[0] in curr_graph.nodes:
        new_center = new_center_tuple[0]
    elif new_center_tuple[1] in curr_graph.nodes:
        new_center = new_center_tuple[1]
    else:
        raise ValueError("new centers {} don't contain a node of current graph {}".format(new_center_tuple, curr_graph.nodes))
    draw_graph(curr_graph, show=False)

    curr_circ, curr_graph = shift_center.shift_centers(curr_circ, curr_graph, curr_center, new_center)


    print(curr_circ.draw(output="mpl"))
    
    draw_graph(curr_graph)
