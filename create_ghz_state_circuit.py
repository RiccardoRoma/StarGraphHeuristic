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
def create_ghz_state_circuit(graph_file: str, parallel_merge: bool = True, generate_star_states: bool = False) -> Tuple[QuantumCircuit, Graph]:
    # load initial graph from pickle
    graph_orig = None
    with open(graph_file, "rb") as f:
        graph_orig = pickle.load(f)

    num_qubits = max(list(graph_orig.nodes())) + 1 # count of nodes starts at zero
    
    if parallel_merge:
        merge_pattern = MergePattern.from_graph_parallel(graph_orig)
    else:
        merge_pattern = MergePattern.from_graph_sequential(graph_orig)

    return create_ghz_state_circuit_graph(merge_pattern, num_qubits, star=generate_star_states)

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

def create_ghz_state_circuit_grow(graph_orig: Graph,
                                  total_num_qubits: int,
                                  state_size: int = None) -> Tuple[QuantumCircuit, Graph]:
    # consistency check
    if max(list(graph_orig)) >= total_num_qubits:
        raise ValueError("Node indices of input graph exceed total number of qubits in circuit!")
    # save initial graph
    init_graph = copy.deepcopy(graph_orig)
    circ = QuantumCircuit(total_num_qubits)

    if state_size is None:
        state_size = len(init_graph)

    max_degree_node = mgo.get_graph_center(init_graph)

    considered_nodes = [max_degree_node]
    circ.h(max_degree_node)
    while len(considered_nodes) < state_size:
        for n0 in considered_nodes:
            if len(considered_nodes) > state_size:
                break
            for n1 in init_graph.neighbors(n0):
                if n1 not in considered_nodes:
                    if len(considered_nodes) > state_size:
                        break
                    else:
                        circ.cx(n0, n1)
                        considered_nodes.append(n1)

    return circ, graph_orig


def create_ghz_state_circuit_graph(pattern: MergePattern,
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
        ## To-Do: This has to be generalized for merge pairs of arbitrary size!
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

