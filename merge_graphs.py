from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
import copy
from networkx import Graph
import modify_graph_objects as mgo

def is_single_qubit_graph(G: Graph) -> bool:
    if G.number_of_nodes() == 1:
        return True
    else:
        return False
    
def validate_input(circ: QuantumCircuit, C1: int, graph1: Graph, C2: int, graph2: Graph, cls_bit_cnt: int) -> None:
    if C1 not in graph1.nodes:
        print("graph nodes {}".format(graph1.nodes))
        print("center {}".format(C1))
        raise ValueError("center 1 is not found in graph 1 nodes!")
    if C2 not in graph2.nodes:
        print("graph nodes {}".format(graph2.nodes))
        print("center {}".format(C2))
        raise ValueError("center 2 is not found in graph 2 nodes!")
    if C1 == C2:
        raise ValueError("to-be-merged centers coincide!")
    if circ.num_qubits < max(list(graph1.nodes)):
        print("graph nodes {}".format(graph1.nodes))
        print(circ.draw())
        raise ValueError("Not all nodes of graph 1 are contained in input circuit")
    if circ.num_qubits-1 < max(list(graph2.nodes)):
        print("graph nodes {}".format(graph2.nodes))
        print(circ.draw())
        raise ValueError("Not all nodes of graph 2 are contained in input circuit")
    if circ.num_clbits-1 > cls_bit_cnt:
        print(circ.draw())
        print("classical bit {}".format(cls_bit_cnt))
        raise ValueError("Classical bit is already contained in input circuit!")
    
def validate_input_list(circ: QuantumCircuit, merging_edges: list[tuple[int, int]], graph_set: list[Graph], cls_bit_cnt: int) -> None:
    for edge in merging_edges:
        if edge[0] == edge[1]:
            raise ValueError(f"Edge {edge} is a self connection to node {edge[0]}!")
        found_edge0 = False
        found_edge1 = False
        for graph in graph_set:
            if edge[0] in graph.nodes:
                if edge[1] in graph.nodes:
                    raise ValueError(f"Merging edge {edge} is contained in single graph {graph.nodes}")
                else:
                    found_edge0 = True
            if edge[1] in graph.nodes:
                found_edge1 = True

        if not found_edge0:
            raise ValueError(f"Unable to find correct node {edge[0]} in graph set for merging edge {edge}!")
        if not found_edge1:
            raise ValueError(f"Unable to find correct node {edge[1]} in graph set for merging edge {edge}!")
    
    for graph in graph_set:
        if circ.num_qubits - 1  < max(list(graph.nodes)):
            print("graph nodes {}".format(graph.nodes))
            print(circ.draw())
            raise ValueError(f"Not all nodes of graph {graph.nodes} are contained as qubits in input circuit")
        
    if circ.num_clbits-1 > cls_bit_cnt:
        print(circ.draw())
        print("classical bit {}".format(cls_bit_cnt))
        raise ValueError("Classical bit is already contained in input circuit!")
    

def merge_graphs(circ: QuantumCircuit, C1: int, graph1: Graph, C2: int, graph2: Graph, cls_bit_cnt: int, reuse_meas_qubit: bool=True) -> Tuple[QuantumCircuit, Graph, int]:
    if is_single_qubit_graph(graph1):
        circ.cz(C1, C2)
    elif is_single_qubit_graph(graph2):
        circ.cz(C1, C2)
    else:
        validate_input(circ, C1, graph1, C2, graph2, cls_bit_cnt)

        #Apply CNOT between the centers of two stars
        circ.cx(C1, C2)
        
        # C2 is the target of cx and is thus the measured center
        curr_meas = ClassicalRegister(1, "m"+str(cls_bit_cnt))
        circ.add_register(curr_meas)
        circ.measure(C2, curr_meas)
        # get list of leaf qubits
        leaf_qubits_g2= copy.deepcopy(list(graph2.nodes))
        leaf_qubits_g2.remove(C2)
        # Applying Pauli corrections
        with circ.if_test((curr_meas[0], 1)):
            for i in leaf_qubits_g2:
                circ.z(i)
            if reuse_meas_qubit:
                # flip qubit into state 0 if it was projected to 1 and qubit is resused 
                circ.x(C2)
        # update classical bit for measurements
        cls_bit_cnt += 1
        # include measured qubit into merged graph state again (if desired)
        if reuse_meas_qubit:
            circ.h(C2)
            circ.cz(C1, C2)
    # merge graph obejects accordingly
    graph_new = mgo.merging_two_graphs(graph1, graph2, (C1, C2))
    # If the measured center is not included again in the graph state, 
    # the edge has also to be removed from merged graph object.
    if not reuse_meas_qubit:
        graph_new.remove_edge(C1, C2)
    
    return circ, graph_new, cls_bit_cnt

def merge_graphs_circ(circ: QuantumCircuit, C1: int, graph1: Graph, C2: int, graph2: Graph, cls_bit_cnt: int, reuse_meas_qubit: bool=True) -> Tuple[QuantumCircuit, int]:
    if is_single_qubit_graph(graph1):
        circ.cz(C1, C2)
    elif is_single_qubit_graph(graph2):
        circ.cz(C1, C2)
    else:
        validate_input(circ, C1, graph1, C2, graph2, cls_bit_cnt)

        #Apply CNOT between the centers of two stars
        circ.cx(C1, C2)
        
        # C2 is the target of cx and is thus the measured center
        curr_meas = ClassicalRegister(1, "m"+str(cls_bit_cnt))
        circ.add_register(curr_meas)
        circ.measure(C2, curr_meas)
        # get list of leaf qubits
        leaf_qubits_g2= copy.deepcopy(list(graph2.nodes))
        leaf_qubits_g2.remove(C2)
        # Applying Pauli corrections
        with circ.if_test((curr_meas[0], 1)):
            for i in leaf_qubits_g2:
                circ.z(i)
            if reuse_meas_qubit:
                # flip qubit into state 0 if it was projected to 1 and qubit is resused 
                circ.x(C2)
        # update classical bit for measurements
        cls_bit_cnt += 1
        # include measured qubit into merged graph state again (if desired)
        if reuse_meas_qubit:
            circ.h(C2)
            circ.cz(C1, C2)
    
    return circ, cls_bit_cnt

def merge_ghz(circ: QuantumCircuit, C1: int, graph1: Graph, C2: int, graph2: Graph, cls_bit_cnt: int, reuse_meas_qubit: bool=True) -> Tuple[QuantumCircuit, Graph, int]:
    if is_single_qubit_graph(graph1):
        circ.cx(C2, C1)
    elif is_single_qubit_graph(graph2):
        circ.cx(C1, C2)
    else:
        validate_input(circ, C1, graph1, C2, graph2, cls_bit_cnt)
        #Apply CNOT between the centers of two stars
        circ.cx(C1, C2)
        
        # C2 is the target of cx and is thus the measured center
        curr_meas = ClassicalRegister(1, "m"+str(cls_bit_cnt))
        circ.add_register(curr_meas)
        circ.measure(C2, curr_meas)
        # get list of leaf qubits
        leaf_qubits_g2= copy.deepcopy(list(graph2.nodes))
        leaf_qubits_g2.remove(C2)
        # Applying Pauli corrections
        with circ.if_test((curr_meas[0], 1)):
            for i in leaf_qubits_g2:
                circ.x(i)
            if reuse_meas_qubit:
                # flip qubit into state 0 if it was projected to 1 and qubit is resused 
                circ.x(C2)
        # update classical bit for measurements
        cls_bit_cnt += 1
        # include measured qubit into merged graph state again (if desired)
        if reuse_meas_qubit:
            circ.cx(C1, C2)
    # merge graph obejects accordingly
    graph_new = mgo.merging_two_graphs(graph1, graph2, (C1, C2))
    # If the measured center is not included again in the graph state, 
    # the edge has also to be removed from merged graph object.
    if not reuse_meas_qubit:
        graph_new.remove_edge(C1, C2)
    
    return circ, graph_new, cls_bit_cnt

def merge_ghz_circ_binary(circ: QuantumCircuit, C1: int, graph1: Graph, C2: int, graph2: Graph, cls_bit_cnt: int, reuse_meas_qubit: bool=True) -> Tuple[QuantumCircuit, int]:
    if is_single_qubit_graph(graph1):
        circ.cx(C2, C1)
    elif is_single_qubit_graph(graph2):
        circ.cx(C1, C2)
    else:
        validate_input(circ, C1, graph1, C2, graph2, cls_bit_cnt)
        #Apply CNOT between the centers of two stars
        circ.cx(C1, C2)
        
        # C2 is the target of cx and is thus the measured center
        curr_meas = ClassicalRegister(1, "m"+str(cls_bit_cnt))
        circ.add_register(curr_meas)
        circ.measure(C2, curr_meas)
        # get list of leaf qubits
        leaf_qubits_g2= copy.deepcopy(list(graph2.nodes))
        leaf_qubits_g2.remove(C2)
        # Applying Pauli corrections
        with circ.if_test((curr_meas[0], 1)):
            for i in leaf_qubits_g2:
                circ.x(i)
            if reuse_meas_qubit:
                # flip qubit into state 0 if it was projected to 1 and qubit is resused 
                circ.x(C2)
        # update classical bit for measurements
        cls_bit_cnt += 1
        # include measured qubit into merged graph state again (if desired)
        if reuse_meas_qubit:
            circ.cx(C1, C2)

    return circ, cls_bit_cnt

def merge_ghz_circ_linear(circ: QuantumCircuit, graph_set: list[Graph], merging_edges: list[tuple[int, int]], cls_bit_cnt: int, reuse_meas_qubit: bool=True) -> Tuple[QuantumCircuit, int]:
    # check if all inputs are valid
    validate_input_list(circ, merging_edges, graph_set, cls_bit_cnt)

    # list of all graphs and edges that have been merged via measurements
    meas_merged = []
    # iterate through all merging edges and merge the corresponding graphs
    for edge in merging_edges:
        # find the corresponding graphs to current merging edge
        graph1 = None
        graph2 = None
        for graph in graph_set:
            if edge[0] in graph.nodes:
                graph1 = graph
                continue
            if edge[1] in graph.nodes:
                graph2 = graph
                continue
        # check if both graphs for merge have been found
        if graph1 is None:
            raise ValueError(f"unable to find first graph for merging edge {edge}!")
        if graph2 is None:
            raise ValueError(f"unable to find second graph for merging edge {edge}!")
        
        # handle single qubit graph cases
        if is_single_qubit_graph(graph1):
            circ.cx(edge[1], edge[0])
        elif is_single_qubit_graph(graph2):
            circ.cx(edge[0], edge[1])
        else:
            #Apply CNOT between the centers of two stars
            circ.cx(edge[0], edge[1])
            
            # C2 is the target of cx and is thus the measured center
            curr_meas = ClassicalRegister(1, "m"+str(cls_bit_cnt))
            circ.add_register(curr_meas)
            circ.measure(edge[1], curr_meas)
            # include measured qubit into merged graph state again (if desired)
            # This operation commutes with the Pauli corrections (PC's) and is moved from the end to here (before PC's) to potentially reduce the circuit depth
            if reuse_meas_qubit:
                circ.cx(edge[0], edge[1])
            # update classical bit for measurements
            cls_bit_cnt += 1
            # save this merge to apply pauli corrections and reuse qubits
            meas_merged.append((graph1, graph2, edge, curr_meas[0]))
            

        
    # apply pauli corrections to all measured merged graphs
    for graph1, graph2, edge, cls_bit in meas_merged:
        # get list of leaf qubits
        leaf_qubits_g2= copy.deepcopy(list(graph2.nodes))
        leaf_qubits_g2.remove(edge[1])
        # Applying Pauli corrections
        with circ.if_test((cls_bit, 1)):
            for i in leaf_qubits_g2:
                circ.x(i)
            if reuse_meas_qubit:
                # flip qubit into state 0 if it was projected to 1 and qubit is resused 
                circ.x(edge[1])
    
    # # include measured qubit into merged graph state again (if desired)
    # if reuse_meas_qubit:        
    #     for graph1, graph2, edge, cls_bit in meas_merged:
    #         circ.cx(edge[0], edge[1])

    return circ, cls_bit_cnt

if __name__ == "__main__":
    # small test with bell pairs
    graph1 = Graph()
    graph1.add_nodes_from([0,1])
    graph1.add_edge(0,1)

    graph2 = Graph()
    graph2.add_nodes_from([2,3])
    graph2.add_edge(2,3)

    graph3 = Graph()
    graph3.add_nodes_from([4,5])
    graph3.add_edge(4,5)

    graphs = [graph1, graph2, graph3]
    merging_edges = [(1,2), (3,4)]

    circ = QuantumCircuit(6)
    # create bell states
    circ.h(0)
    circ.h(2)
    circ.h(4)
    
    circ.cx(0,1)
    circ.cx(2,3)
    circ.cx(4,5)

    cls_bit_cnt= 0
    circ, cls_bit_cnt = merge_ghz_circ_linear(circ, graphs, merging_edges, cls_bit_cnt, reuse_meas_qubit=True)

    print(circ.draw())
    print(f"circuit depth {circ.depth()}")
