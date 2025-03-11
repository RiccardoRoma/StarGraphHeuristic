from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.passes import RemoveBarriers
import copy
import pickle
import shift_center
import generate_star_states
import merge_graphs
from Qiskit_input_graph import MergePattern, draw_graph, calculate_msq, generate_graph, parallel_merge, sequential_merge, find_merging_tree, generate_random_graph
import networkx as nx
from networkx import Graph
import modify_graph_objects as mgo
import matplotlib.pyplot as plt

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
    
    # handle star graphs with shifting of centers before each merge
    # this means that we have to merge them binary
    if not pattern.is_binary and star:
        raise ValueError("Merge pattern is not represented by a binary tree, but merging star graphs can only be handled binary!")

    
    init_graph = pattern.initial_graph

    circ_shift_merge = QuantumCircuit(total_num_qubits)
    # construct initial subgraphs
    for graph in pattern.get_initial_subgraphs():
        if star:
            circ_shift_merge = generate_star_states.generate_star_state(graph, circ_shift_merge)
        else:
            circ_shift_merge = generate_star_states.generate_ghz_state(graph, circ_shift_merge)

    circ_shift_merge.barrier()

    cls_bit_cnt = 0 # counts how many measurements have been made
    for layer in range(len(pattern)):
        for graph_set, merging_edges in pattern.get_merge_sets(layer):
            if star:
                # iterate through merging edges
                already_merged = []
                for edge in merging_edges:
                    if edge in already_merged:
                        continue

                    # find corresponding subgraph pair
                    subgraph1 = None
                    subgraph2 = None
                    for graph in graph_set:
                        if edge[0] in graph.nodes:
                            subgraph1 = graph
                        elif edge[1] in graph.nodes:
                            subgraph2 = graph

                    if subgraph1 is None:
                        raise ValueError(f"Unable to find graph with node {edge[0]} in graph set.")
                    if subgraph2 is None:
                        raise ValueError(f"Unable to find graph with node {edge[1]} in graph set.")
                    
                    # get current graph centers
                    curr_center1 = mgo.get_graph_center(subgraph1)
                    curr_center2 = mgo.get_graph_center(subgraph2)
                    
                    # shift centers
                    circ_shift_merge = shift_center.shift_centers_circ(circ_shift_merge, subgraph1, curr_center1, edge[0]) # call shifting function
                    circ_shift_merge = shift_center.shift_centers_circ(circ_shift_merge, subgraph2, curr_center2, edge[1]) # call shifting function

                    # determine the remaining center
                    subgraph1_idx = pattern.find_subgraph(subgraph1)
                    if subgraph1_idx is None:
                        raise ValueError(f"Unable to find subgraph {subgraph1.nodes} in subgraph list of MergePattern!")
                    succ_graph_idx = list(pattern.pattern_graph.successors(subgraph1_idx))
                    if len(succ_graph_idx) > 1:
                        raise ValueError(f"More than one succesor was found for subgraph {subgraph1_idx} in the merge pattern")
                    succ_graph_idx = succ_graph_idx[0]
                    remaining_center = mgo.get_graph_center(pattern.subgraphs[succ_graph_idx])
                    
                    # merge
                    if remaining_center == edge[0]:
                        circ_shift_merge, cls_bit_cnt = merge_graphs.merge_graphs_circ(circ_shift_merge, edge[0], subgraph1, edge[1], subgraph2, cls_bit_cnt)
                    elif remaining_center == edge[1]:
                        circ_shift_merge, cls_bit_cnt = merge_graphs.merge_graphs_circ(circ_shift_merge, edge[1], subgraph2, edge[0], subgraph1, cls_bit_cnt)
                    else:
                        raise ValueError(f"Remaining center {remaining_center} is not contained in the merging edge {edge}!")

                    already_merged.append(edge)

            else:
                # handle ghz states
                circ_shift_merge, cls_bit_cnt = merge_graphs.merge_ghz_circ_linear(circ_shift_merge, graph_set, merging_edges[1:], cls_bit_cnt, reuse_meas_qubit=True) # merging edges contains the first merging edge two times to have the same length as graph sets

            circ_shift_merge.barrier()
    subgraph = pattern.subgraphs[-1]
    if star:
        circ_shift_merge = convert_star_to_ghz(circ_shift_merge, subgraph)

    return circ_shift_merge, init_graph, subgraph


if __name__ == "__main__":
    # # List of nodes
    # nodes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # # Edge list (each tuple represents an edge between two nodes)
    # edges_list = [(0, 1), (0, 4), (1, 6), (2, 3), (3, 7), (4, 9), (5, 6), (6, 8), (7, 9), (8, 9), (0, 10), (10,11),(11,12)]
    # init_graph = generate_graph(nodes_list, edges_list, use_barabasi=False)
    num_qubits = 60
    init_graph = generate_random_graph(num_qubits, 0.1, False, show_plot=False)
    draw_graph(init_graph, show=False)

    # caclualte msq
    _,msq,_= calculate_msq(init_graph, show_status=False)
    
    #draw_binary_tree(bt1, show=False)
    # Define MergePattern instance for binary parallel merge
    #ptrn1 = MergePattern.from_graph_parallel(init_graph, binary_merge=True)
    bt1, msq1 = parallel_merge(init_graph, msq)
    ptrn1 = MergePattern(init_graph, msq1, bt1)

    #draw_binary_tree(bt2, show=True)
    # Define MergePattern instance for sequential merge
    #ptrn2 = MergePattern.from_graph_sequential(init_graph)
    bt2, msq2 = sequential_merge(init_graph, msq)
    ptrn2 = MergePattern(init_graph, msq2, bt2)

    # Define MergePattern instance for non-binary parallel merge
    #ptrn3 = MergePattern.from_graph_parallel(init_graph, binary_merge=False)
    bt3, msq3 = find_merging_tree(init_graph, msq)
    ptrn3 = MergePattern(init_graph, msq3, bt3)


    # create circuits
    circ1, _ , _ = create_ghz_state_circuit_graph(ptrn1, num_qubits, star=True)
    circ2, _ , _ = create_ghz_state_circuit_graph(ptrn2, num_qubits, star=True)
    circ3, _ , _ = create_ghz_state_circuit_graph(ptrn3, num_qubits, star=False)

    # print all merging pairs
    print("merge sets for binary parallel merge:")
    for layer in range(len(ptrn1)):
        curr_merge_set = ptrn1.get_merge_sets(layer)
        curr_set = ptrn1._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge sets {curr_set}")
        for graph_set_idcs, graph_set in zip(curr_set,curr_merge_set):
            if len(graph_set_idcs) != len(graph_set[0]):
                raise ValueError("graph set indices and graph set do not have the same length!")
            print(f"    merging set {sorted(graph_set_idcs)}:")
            for graph_idx, graph, edge in zip(sorted(graph_set_idcs), graph_set[0], graph_set[1]):
                print(f"    graph {graph_idx}: nodes={graph.nodes()}, merging edge {edge}")
            print()
    print(f"merge tree is binary {ptrn1.is_binary}")
    print(f"corresponding circuit depth (no barriers): {RemoveBarriers()(circ1).depth()}")
    #print("Corresponding circuit: ")
    #print(circ1.draw())
    print()
    print("merge sets for sequential merge:")
    for layer in range(len(ptrn2)):
        curr_merge_set = ptrn2.get_merge_sets(layer)
        curr_set = ptrn2._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge sets {curr_set}")
        for graph_set_idcs, graph_set in zip(curr_set,curr_merge_set):
            if len(graph_set_idcs) != len(graph_set[0]):
                raise ValueError("graph set indices and graph set do not have the same length!")
            print(f"    merging set {sorted(graph_set_idcs)}:")
            for graph_idx, graph, edge in zip(sorted(graph_set_idcs), graph_set[0], graph_set[1]):
                print(f"    graph {graph_idx}: nodes={graph.nodes()}, merging edge {edge}")
            print()
    print(f"merge tree is binary {ptrn2.is_binary}")
    print(f"corresponding circuit depth (no barriers): {RemoveBarriers()(circ2).depth()}")
    #print("Corresponding circuit: ")
    #print(circ2.draw())
    print()
    print("merge sets for non-binary parallel merge:")
    for layer in range(len(ptrn3)):
        curr_merge_set = ptrn3.get_merge_sets(layer)
        curr_set = ptrn3._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge sets {curr_set}")
        for graph_set_idcs, graph_set in zip(curr_set,curr_merge_set):
            if len(graph_set_idcs) != len(graph_set[0]):
                raise ValueError("graph set indices and graph set do not have the same length!")
            print(f"    merging set {sorted(graph_set_idcs)}:")
            for graph_idx, graph, edge in zip(sorted(graph_set_idcs), graph_set[0], graph_set[1]):
                print(f"    graph {graph_idx}: nodes={graph.nodes()}, merging edge {edge}")
            print()
    print(f"merge tree is binary {ptrn3.is_binary}")
    print(f"corresponding circuit depth (no barriers): {RemoveBarriers()(circ3).depth()}")
    #print("Corresponding circuit: ")
    #print(circ3.draw())

    # draw the binary tree representing the merge pattern and the corresponding circuit
    ptrn1.draw_pattern_graph(show_weights=True, show=False)
    
    ptrn2.draw_pattern_graph(show_weights=True, show=False)
    ptrn3.draw_pattern_graph(show_weights=True, show=False)

    plt.show()
