from typing import Sequence, Tuple, Union
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

def generate_graph_state(G: Graph, circ: QuantumCircuit) -> QuantumCircuit:
    """Generates a graph state according to the given graph. The graph generation is added to the given QuantumCircuit

    Args:
        G: Graph corresponding to the graph state that should be generated.
        circ: Quantum circuit to which the graph state generation is added.

    Raises:
        ValueError: If not all graph nodes are included as qubits in the quantum circuit

    Returns:
        Modified quantum circuit. The modification is done inplace.
    """
    # consistency check
    if max(list(G)) >= circ.num_qubits:
        raise ValueError("Node indices of input graph exceed total number of qubits in circuit!")
    # initialize all used qubits in |+>
    for q in G.nodes():
        circ.h(q)

    # apply CZ gates according to the edges in the graph object
    for e in G.edges():
        circ.cz(e[0], e[1])

    return circ


def generate_ghz_state(G: Graph, circ: QuantumCircuit, state_size: Union[int, None] = None) -> QuantumCircuit:
    """Generate a ghz state from a given graph by performing a BFS search starting from the node with the highest degree.

    Args:
        G: Input graph for the BFS search
        circ: Quantum circuit to which the state generation is added
        state_size: Target state size of the final GHZ state. Defaults to None, which means that the state size equals the number of nodes in G.

    Raises:
        ValueError: If not all graph nodes are included as qubits in the quantum circuit

    Returns:
        Modifies quantum circuit. Note that the state generation is added to the circuit inplace.
    """
    # consistency check
    if max(list(G)) >= circ.num_qubits:
        raise ValueError("Node indices of input graph exceed total number of qubits in circuit!")
    
    # set required size of the final ghz state if unset
    if state_size is None:
        state_size = len(G)
    
    # check if G is not a trivial case
    if state_size > 1:
        max_degree_node = mgo.get_graph_center(G) # get the node with the highest degree in the initial graph
    
        # Perform a BFS for the ghz state generation until the required size is reached
        considered_nodes = [max_degree_node]
        circ.h(max_degree_node)
        while len(considered_nodes) < state_size:
            for n0 in considered_nodes:
                if len(considered_nodes) > state_size:
                    break
                for n1 in G.neighbors(n0):
                    if n1 not in considered_nodes:
                        if len(considered_nodes) > state_size:
                            break
                        else:
                            circ.cx(n0, n1)
                            considered_nodes.append(n1)

    return circ

    