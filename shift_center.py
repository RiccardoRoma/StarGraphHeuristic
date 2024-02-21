from typing import List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import h


def shift_centers(center_in: int, center_fin: int, circ: QuantumCircuit) -> QuantumCircuit:
    # shift away from the initial center
    circ.h(center_in)

    # shift to the final center
    circ.h(center_fin)
    
    return circ