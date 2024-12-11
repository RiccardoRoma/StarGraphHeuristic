from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import copy
import pickle
import shift_center
import generate_star_states
import merge_graphs
from Qiskit_input_graph import MergePattern, draw_graph, calculate_msq
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

# wrapper function to generate circuit from a file that contains the graph
def create_ghz_state_circuit(graph_file: str) -> Tuple[QuantumCircuit, Graph]:
    # load initial graph from pickle
    graph_orig = None
    with open(graph_file, "rb") as f:
        graph_orig = pickle.load(f)
    
    return create_ghz_state_circuit_graph(graph_orig)

def create_ghz_state_circuit_debug(graph_orig: Graph,
                                   total_num_qubits: int) -> Tuple[QuantumCircuit, Graph]:
    # consistency check
    if max(list(graph_orig)) > total_num_qubits:
        raise ValueError("Node indices of input graph exceed total number of qubits in circuit!")
    # save initial graph
    init_graph = copy.deepcopy(graph_orig)
    circ = QuantumCircuit(total_num_qubits)

    max_degree_node = mgo.get_graph_center(init_graph)

    considered_nodes = [max_degree_node]
    circ.h(max_degree_node)
    while len(considered_nodes) < len(init_graph):
        for edge in init_graph.edges:
            if edge[0] in considered_nodes:
                if edge[1] not in considered_nodes:
                    circ.cx(edge[0], edge[1])
                    considered_nodes.append(edge[1])
            else:
                if edge[1] in considered_nodes:
                    circ.cx(edge[1], edge[0])
                    considered_nodes.append(edge[0])

    return circ, graph_orig

def create_ghz_state_circuit_graph(graph_orig: Graph,
                                   total_num_qubits: int, star = False) -> Tuple[QuantumCircuit, Graph]:
    # False = GHZ state, True = star state

    # consistency check
    if max(list(graph_orig)) > total_num_qubits:
        raise ValueError("Node indices of input graph exceed total number of qubits in circuit!")
    # save initial graph
    init_graph = copy.deepcopy(graph_orig)
    # run calculate_msq form qiskit_input_graph.py
        # calculates the list of subgraphs in sequential merging order (MSQ) 
        # and at what edges we should merge the graphs (merging_edges_list)
    merging_edges_list, subgraphs, sid_total_gates = calculate_msq(graph_orig, show_status=False)
    
    circ = QuantumCircuit(total_num_qubits)

    # create seperate copy for shifting of star centers and merging
    circ_shift_merge = circ.copy()
    

    # in each iteration two graphs are center-shifted and then merged
    subgraph1 = subgraphs[0] # start with the first graph in MSQ

    if star:
        circ_shift_merge = generate_star_states.generate_star_state(subgraph1, circ_shift_merge)
    else:
        circ_shift_merge = generate_star_states.generate_ghz_state(subgraph1, circ_shift_merge)

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
        
        if star:
            circ_shift_merge, subgraph1 = shift_center.shift_centers(circ_shift_merge, subgraph1, curr_center1, new_center1) # call shifting function
        else:
            # For GHZ states only centers in nx.Graph's have to be shifted, to merge at the right edges.
            subgraph1 = mgo.update_graph_center(subgraph1, new_center1)
            

        subgraph2 = subgraphs[i] # second graph that should be merged with first graph

        if star:
            circ_shift_merge = generate_star_states.generate_star_state(subgraph2, circ_shift_merge)
        else: 
            circ_shift_merge = generate_star_states.generate_ghz_state(subgraph2, circ_shift_merge)
        curr_center2 = mgo.get_graph_center(subgraph2) # determine current center
        # merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
        if new_center_tuple[0] in subgraph2.nodes:
            new_center2 = new_center_tuple[0]
        elif new_center_tuple[1] in subgraph2.nodes:
            new_center2 = new_center_tuple[1]
        else:
            raise ValueError("new centers {} don't contain a node of subgraph 2 {}".format(new_center_tuple, subgraph2.nodes))
        
        if star:
            circ_shift_merge, subgraph2 = shift_center.shift_centers(circ_shift_merge, subgraph2, curr_center2, new_center2) # call shifting function
        else:
            # For GHZ states only centers in nx.Graph's have to be shifted, to merge at the right edges.
            subgraph2 = mgo.update_graph_center(subgraph2, new_center2)
        

        # ... call merging for subgraph1 and subgraph2
        if star:
            circ_shift_merge, subgraph1, cls_bit_cnt = merge_graphs.merge_graphs(circ_shift_merge, new_center1, subgraph1, new_center2, subgraph2, cls_bit_cnt)
        else:
            circ_shift_merge, subgraph1, cls_bit_cnt = merge_graphs.merge_ghz(circ_shift_merge, new_center1, subgraph1, new_center2, subgraph2, cls_bit_cnt)

        circ_shift_merge.barrier()
        
    # compose circuits
    #circ = circ_substars.compose(circ_shift_merge)
    circ = circ_shift_merge.copy()

    # ToDo: convert star state into GHZ
    if star:
        circ = convert_star_to_ghz(circ, subgraph1)

    return circ, init_graph, subgraph1

def create_ghz_state_circuit_graph_pattern(pattern: MergePattern,
                                           total_num_qubits: int,
                                           star: bool = False) -> Tuple[QuantumCircuit, Graph, Graph]:
    # consistency check
    if max(list(pattern.initial_graph)) > total_num_qubits:
        raise ValueError("Node indices of input graph exceed total number of qubits in circuit!")
    
    init_graph = pattern.initial_graph

    circ_shift_merge = QuantumCircuit(total_num_qubits)
    # construct initial subgraphs
    for graph in pattern.get_initial_subgraphs():
        if star:
            circ_shift_merge = generate_star_states.generate_star_state(graph, circ_shift_merge)
        else:
            circ_shift_merge = generate_star_states.generate_ghz_state(graph, circ_shift_merge)

    cls_bit_cnt = 0 # counts how many measurements have been made
    for layer in range(len(pattern)):
        for graph_pair in pattern.get_merge_pairs(layer):
            subgraph1 = graph_pair[0]
            subgraph2 = graph_pair[1]
            new_center_tuple = graph_pair[2] # merging edge

            curr_center1 = mgo.get_graph_center(subgraph1) # determine current center
            curr_center2 = mgo.get_graph_center(subgraph2) # determine current center
            # merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
            if new_center_tuple[0] in subgraph1.nodes:
                new_center1 = new_center_tuple[0]
            elif new_center_tuple[1] in subgraph1.nodes:
                new_center1 = new_center_tuple[1]
            else:
                raise ValueError("new centers {} don't contain a node of subgraph 1 {}".format(new_center_tuple, subgraph1.nodes))
            if new_center_tuple[0] in subgraph2.nodes:
                new_center2 = new_center_tuple[0]
            elif new_center_tuple[1] in subgraph2.nodes:
                new_center2 = new_center_tuple[1]
            else:
                raise ValueError("new centers {} don't contain a node of subgraph 2 {}".format(new_center_tuple, subgraph2.nodes))
            
            # shift star graph centers
            if star:
                circ_shift_merge = shift_center.shift_centers_circ(circ_shift_merge, subgraph1, curr_center1, new_center1) # call shifting function
                circ_shift_merge = shift_center.shift_centers_circ(circ_shift_merge, subgraph2, curr_center2, new_center2) # call shifting function
            
            # merging of subgraph1 and subgraph2
            if star:
                circ_shift_merge, cls_bit_cnt = merge_graphs.merge_graphs_circ(circ_shift_merge, new_center1, subgraph1, new_center2, subgraph2, cls_bit_cnt)
            else:
                circ_shift_merge, cls_bit_cnt = merge_graphs.merge_ghz_circ(circ_shift_merge, new_center1, subgraph1, new_center2, subgraph2, cls_bit_cnt)
    
            circ_shift_merge.barrier()

    subgraph = pattern.subgraphs[-1]
    if star:
        circ_shift_merge = convert_star_to_ghz(circ_shift_merge, subgraph)

    return circ_shift_merge, init_graph, subgraph

