import merge_graphs
import shift_center
import generate_states
from Qiskit_input_graph import draw_graph, calculate_msq
import pickle
import networkx as nx
from networkx import Graph
from qiskit import QuantumCircuit
import modify_graph_objects as mgo
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator, AerSimulator
import numpy as np
from qiskit.visualization import plot_histogram
from qiskit.result import marginal_distribution
import matplotlib.pyplot as plt

# generate subgraphs and merging sequence
sub_g0 = Graph()
sub_g0.add_nodes_from([0,1,2])
sub_g0.add_edges_from([(0,1),(1,2)])

sub_g1 = Graph()
sub_g1.add_nodes_from([3,4,5])
sub_g1.add_edges_from([(3,4),(4,5)])
subgraphs = [sub_g0, sub_g1]

merging_edges_list=[(1,4)]

print("merging edges list {}".format(merging_edges_list))

c = QuantumCircuit(6)

curr_circ = c.copy()
# handle first subgraph
curr_graph0 = subgraphs[0]
# generate corr. star graph state
curr_circ = generate_states.generate_graph_state(curr_graph0, curr_circ)
curr_center0 = mgo.get_graph_center(curr_graph0) # determine current center
# merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
new_center_tuple = merging_edges_list[0]
if new_center_tuple[0] in curr_graph0.nodes:
    new_center0 = new_center_tuple[0]
elif new_center_tuple[1] in curr_graph0.nodes:
    new_center0 = new_center_tuple[1]
else:
    raise ValueError("new centers {} don't contain a node of current graph {}".format(new_center_tuple, curr_graph0.nodes))
# draw_graph(curr_graph0, show=False)
# check if center must be shifted
curr_circ, curr_graph0 = shift_center.shift_centers(curr_circ, curr_graph0, curr_center0, new_center0)


# print(curr_circ.draw(output="mpl"))

draw_graph(curr_graph0, show=False)

# handle second subgraph 
curr_graph1 = subgraphs[1]
# generate corr. star graph state
curr_circ = generate_states.generate_graph_state(curr_graph1, curr_circ)
curr_center1 = mgo.get_graph_center(curr_graph1) # determine current center
# merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
new_center_tuple = merging_edges_list[0]
if new_center_tuple[0] in curr_graph1.nodes:
    new_center1 = new_center_tuple[0]
elif new_center_tuple[1] in curr_graph1.nodes:
    new_center1 = new_center_tuple[1]
else:
    raise ValueError("new centers {} don't contain a node of current graph {}".format(new_center_tuple, curr_graph1.nodes))
#draw_graph(curr_graph1, show=False)
# check if center must be shifted
curr_circ, curr_graph1 = shift_center.shift_centers(curr_circ, curr_graph1, curr_center1, new_center1)


#print(curr_circ.draw(output="mpl"))

draw_graph(curr_graph1, show=False)

# merge the two subgraphs
curr_circ, curr_graph0, _ = merge_graphs.merge_graphs(curr_circ, new_center0, curr_graph0, new_center1, curr_graph1, 0)

print(curr_circ.draw(output="mpl"))

# generate reference of the merged subgraphs
curr_circ_ref = c.copy()
# generate star graph state of the merged graph
curr_circ_ref = generate_states.generate_graph_state(curr_graph0, curr_circ_ref)
print(curr_circ_ref.draw(output="mpl"))

draw_graph(curr_graph0)

# compare circuit with reference. Sample 10-times to test both measurement outcomes of the mid-circuit measurement.
for i in range(10):
    curr_result = StatevectorSimulator().run(curr_circ).result()
    curr_state = curr_result.get_statevector(curr_circ)
    
    curr_result_ref = StatevectorSimulator().run(curr_circ_ref).result()
    curr_state_ref = curr_result_ref.get_statevector(curr_circ_ref)
    
    overlap = np.abs(curr_state.inner(curr_state_ref))
    print("run {}: overlap with reference {}".format(i,overlap))

# circ_comb = curr_circ.compose(curr_circ_ref.inverse())
# circ_comb.measure_all()
# print(circ_comb.draw(output="mpl"))
# result = AerSimulator().run(circ_comb).result()
# 
# statistics = result.get_counts()
# #filtered_statistics = marginal_distribution(statistics, [1])
# #plot_histogram(filtered_statistics)
# plot_histogram(statistics)
# filtered_statistics = marginal_distribution(statistics, list(range(1,6)))
# plot_histogram(filtered_statistics)
# 
# plt.show()

