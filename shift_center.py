from typing import List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import SXGate, SGate
from generate_star_states import get_leaf_qubits_from_edges


def shift_centers(qubit_list: List[int], center_in: int, center_fin: int, circ: QuantumCircuit, add_barries: bool = False) -> QuantumCircuit:
    # make the two lists of leaf qubits before and after center shift 
    qub_lst_in = qubit_list.remove(center_in)
    qub_lst_fin = qubit_list.remove(center_fin)
    # shift away from the initial center
    circ.SXGate(center_in)
    for qubit in qub_lst_in:
        circ.SGate(qubit)
    # shift to the final center
    circ.SXGate(center_fin)
    for qubit in qub_lst_fin:
        circ.SGate(qubit)
    
    return circ