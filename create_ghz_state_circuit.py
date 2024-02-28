from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import copy
import pickle
import shift_center
import generate_star_states
import merge_graphs

def create_ghz_state_circuit(graph_file: str) -> QuantumCircuit:
    # load initial graph
    # run calculate_msq form qkit_input_graph.py

    # ToDo: QuantumCircuit template
    # iterate through Graph objects and ...
    # ... call generate star state
    # ... call shifitng of centers
    # ... call merging 
    # ToDo need to know how many ClassicalRegister are needed for merging
    # Riccardo: we can re-use the classical bit after Pauli corrections
    return 

