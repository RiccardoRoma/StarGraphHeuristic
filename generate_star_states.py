from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import copy
from networkx import Graph
import modify_graph_objects as mgo

# This function should return a lists, with all leaf qubits in it
def get_leaf_qubits_from_edges(edges_in: Sequence[Tuple[int, int]],
                               star_centers: Sequence[int]) -> np.ndarray:
    leaf_list = []
    for edge in edges_in:
        e1 = edge[0]
        e2 = edge[1]

        if e1 not in star_centers:
            if e1 not in leaf_list:
                leaf_list.append(e1)

        if e2 not in star_centers:
            if e2 not in leaf_list:
                leaf_list.append(e2)
    # sort leaf list 
    leaf_list.sort()
    #leaf_list = np.asarray(leaf_list)
    return leaf_list

def generate_star_state(G: Graph, circ: QuantumCircuit) -> QuantumCircuit:
    # assume that circ has one single QuantumRegister with all qubits
    center_index = mgo.get_graph_center(G)

    # initialize all used qubits in |+>
    for q in G.nodes():
        circ.h(q)

    # apply CZ gates according to the edges in the graph object
    for e in G.edges():
        circ.cz(e[0], e[1])

    return circ

    