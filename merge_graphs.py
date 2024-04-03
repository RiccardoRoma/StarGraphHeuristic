from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
import copy
from networkx import Graph
import modify_graph_objects as mgo
from generate_star_states import get_leaf_qubits_from_edges


def merge_graphs(circ: QuantumCircuit, C1: int, C2: int, cls_bit_cnt: int, edge_list_G1) -> Tuple[QuantumCircuit, int]:
    
    #Apply CNOT between the centers of two stars
    circ.cnot(C1, C2)
    
    # Assume C1 is the target center
    meas= circ.measure([C1], [cls_bit_cnt])

    # Applying Pauli corrections
    if meas==1:
        leaf_qubits= get_leaf_qubits_from_edges(edge_list_G1, C1)
        for i in leaf_qubits:
            circ.Z[i]
    else:
        None
    return circ