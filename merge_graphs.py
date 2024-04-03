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
    circ.measure([C1], [cls_bit_cnt])
    m_0=  ClassicalRegister(1, "m0")
    leaf_qubits= get_leaf_qubits_from_edges(edge_list_G1, C1)
    # Applying Pauli corrections
    with circ.if_test((m_0, 1)):
        
        for i in leaf_qubits:
            circ.Z[i]

    return circ