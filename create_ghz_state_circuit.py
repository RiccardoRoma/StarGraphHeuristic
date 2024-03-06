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
import modify_graph_objects as mgo

def create_ghz_state_circuit(graph_file: str) -> QuantumCircuit:
    # load initial graph
    G = None
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    # run calculate_msq form qkit_input_graph.py
    merging_edges_list, MSQ, sid_total_gates = calculate_msq(G, show_status=False)

    # set up a QuantumCircuit with all requried qubits and classical bits
    qregs = QuantumRegister(len(G.nodes()))
    cregs = ClassicalRegister(len(MSQ)-1) # if merging is sequentially than we need to measure (number of subgraphs -1)-times
    circ = QuantumCircuit(qregs, cregs)

    # create seperate copy for star state generation
    circ_substars = circ.copy()
    # iterate through Graph objects and ...
    for subgraph in MSQ:
        # ... call generate star state
        generate_star_states.generate_star_state(subgraph, circ_substars)
    
    # create seperate copy for shifting of star centers
    # ToDo: It seems appropriate to re-arrange the outputs of calculate_msq to make this step easier!
    #   Iteration should be trough to-be-merged pairs (sG1, sG2) with corr. element of merging_edges_list
    #   we first call shifting of centers and then call merging function
    circ_shift = circ.copy()
    for i in range(len(MSQ)):
        subgraph = MSQ[i]
        curr_center = mgo.get_graph_center(subgraph)
        new_center = 
        circ_shift, subgraph = shift_center(circ_shift, subgraph, curr_center, )

    # ... call shifitng of centers
    # ... call merging 
    # ToDo need to know how many ClassicalRegister are needed for merging
    # Riccardo: we can re-use the classical bit after Pauli corrections
    return 

