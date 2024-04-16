from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import copy
import pickle
import shift_center
import generate_star_states
import merge_graphs
from Qiskit_input_graph import draw_graph, calculate_msq
import networkx as nx
from networkx import Graph
import modify_graph_objects as mgo

def convert_star_to_ghz(circ: QuantumCircuit, graph: Graph) -> QuantumCircuit:
    # get the star center from corr. graph
    star_center = mgo.get_graph_center(graph)
    # get leaf qubit list
    leaf_qubits = copy.deepcopy(list(graph.nodes))
    leaf_qubits.remove(star_center)
    # apply hadamard to all leaf qubits to convert star state to a ghz state
    for q in leaf_qubits:
        circ.h(q)

    return circ

def create_ghz_state_circuit(graph_file: str) -> Tuple[QuantumCircuit, Graph]:
    # load initial graph from pickle
    graph_orig = None
    with open(graph_file, "rb") as f:
        graph_orig = pickle.load(f)
    # save initial graph
    init_graph = copy.deepcopy(graph_orig)
    # run calculate_msq form qiskit_input_graph.py
        # calculates the list of subgraphs in sequential merging order (MSQ) 
        # and at what edges we should merge the graphs (merging_edges_list)
    merging_edges_list, subgraphs, sid_total_gates = calculate_msq(graph_orig, show_status=False)

    # set up a QuantumCircuit with all requried qubits and classical bits
    # qregs = QuantumRegister(len(graph_orig.nodes())) # number of nodes in initial graph is the number of qubits needed.
    # cregs = ClassicalRegister(len(subgraphs)-1) # if merging is sequentially than we need to measure (number of subgraphs - 1)-times
    # circ = QuantumCircuit(qregs, cregs)
    circ = QuantumCircuit(len(graph_orig.nodes))

    # # create seperate copy for star state generation
    # circ_substars = circ.copy()
    # # iterate through Graph objects and ...
    # for subgraph in MSQ:
    #     # ... call generate star state
    #     generate_star_states.generate_star_state(subgraph, circ_substars)
    
    # create seperate copy for shifting of star centers and merging
    circ_shift_merge = circ.copy()
    

    # in each iteration two graphs are center-shifted and then merged
    subgraph1 = subgraphs[0] # start with the first graph in MSQ

    circ_shift_merge = generate_star_states.generate_star_state(subgraph1, circ_shift_merge)

    subgraph2 = None # second graph will be iterated through MSQ
    cls_bit_cnt = 0 # counts how many measurements have been made
    for i in range(1,len(subgraphs)):
        # ... call shifting of centers
        curr_center1 = mgo.get_graph_center(subgraph1) # determine current center
        # merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
        new_center_tuple = merging_edges_list[i-1]
        if new_center_tuple[0] in subgraph1.nodes:
            new_center1 = new_center_tuple[0]
        elif new_center_tuple[1] in subgraph1.nodes:
            new_center1 = new_center_tuple[1]
        else:
            raise ValueError("new centers {} don't contain a node of subgraph 1 {}".format(new_center_tuple, subgraph1.nodes))
        
        circ_shift_merge, subgraph1 = shift_center.shift_centers(circ_shift_merge, subgraph1, curr_center1, new_center1) # call shifting function

        subgraph2 = subgraphs[i] # second graph that should be merged with first graph

        circ_shift_merge = generate_star_states.generate_star_state(subgraph2, circ_shift_merge)

        curr_center2 = mgo.get_graph_center(subgraph2) # determine current center
        # merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
        if new_center_tuple[0] in subgraph2.nodes:
            new_center2 = new_center_tuple[0]
        elif new_center_tuple[1] in subgraph2.nodes:
            new_center2 = new_center_tuple[1]
        else:
            raise ValueError("new centers {} don't contain a node of subgraph 2 {}".format(new_center_tuple, subgraph2.nodes))
        
        circ_shift_merge, subgraph2 = shift_center.shift_centers(circ_shift_merge, subgraph2, curr_center2, new_center2) # call shifting function

        # ... call merging for subgraph1 and subgraph2
        circ_shift_merge, subgraph1, cls_bit_cnt = merge_graphs.merge_graphs(circ_shift_merge, new_center1, subgraph1, new_center2, subgraph2, cls_bit_cnt)

        circ_shift_merge.barrier()
        
    # compose circuits
    #circ = circ_substars.compose(circ_shift_merge)
    circ = circ_shift_merge.copy()

    # ToDo: convert star state into GHZ
    circ = convert_star_to_ghz(circ, subgraph1)

    return circ, init_graph, subgraph1

