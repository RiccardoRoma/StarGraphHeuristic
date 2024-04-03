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

def create_ghz_state_circuit(graph_file: str) -> Tuple[QuantumCircuit, Graph]:
    # load initial graph from pickle
    G = None
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    # run calculate_msq form qiskit_input_graph.py
        # calculates the list of subgraphs in sequential merging order (MSQ) 
        # and at what edges we should merge the graphs (merging_edges_list)
    merging_edges_list, MSQ, sid_total_gates = calculate_msq(G, show_status=False)

    # set up a QuantumCircuit with all requried qubits and classical bits
    qregs = QuantumRegister(len(G.nodes())) # number of nodes in initial graph is the number of qubits needed.
    cregs = ClassicalRegister(len(MSQ)-1) # if merging is sequentially than we need to measure (number of subgraphs - 1)-times
    circ = QuantumCircuit(qregs, cregs)

    # create seperate copy for star state generation
    circ_substars = circ.copy()
    # iterate through Graph objects and ...
    for subgraph in MSQ:
        # ... call generate star state
        generate_star_states.generate_star_state(subgraph, circ_substars)
    
    # create seperate copy for shifting of star centers and merging
    circ_shift_merge = circ.copy()
    
    # in each iteration two graphs are center-shifted and then merged
    subgraph1 = MSQ[0] # start with the first graph in MSQ
    subgraph2 = None # second graph will be iterated through MSQ
    cls_bit_cnt = 0 # counts how many measurements have been made
    for i in range(1,len(MSQ)):
        # ... call shifting of centers
        curr_center1 = mgo.get_graph_center(subgraph1) # determine current center
        new_center1 = merging_edges_list[i-1][0] # get new center
        circ_shift_merge, subgraph1 = shift_center.shift_centers(circ_shift_merge, subgraph1, curr_center1, new_center1) # call shifting function

        subgraph2 = MSQ[i] # second graph that should be merged with first graph
        curr_center2 = mgo.get_graph_center(subgraph2) # determine current center
        new_center2 = merging_edges_list[i-1][1] # get new center
        circ_shift_merge, subgraph2 = shift_center.shift_centers(circ_shift_merge, subgraph2, curr_center2, new_center2) # call shifting function

        # ... call merging for subgraph1 and subgraph2
        circ_shift_merge, subgraph1, cls_bit_cnt = merge_graphs.merge_graphs(circ_shift_merge, subgraph1, subgraph2, cls_bit_cnt)
        
    # compose circuits
    circ = circ_substars.compose(circ_shift_merge)

    # ToDo: convert star state into GHZ

    return circ, subgraph1

