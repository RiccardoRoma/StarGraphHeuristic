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
def merge_ghz(circ: QuantumCircuit, C1: int, graph1: Graph, C2: int, graph2: Graph, cls_bit_cnt: int, reuse_meas_qubit: bool=True) -> Tuple[QuantumCircuit, Graph, int]:
    ## To-Do: Implement merging of GHZ states
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