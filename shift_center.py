from typing import Tuple
from qiskit import QuantumCircuit
from networkx import Graph
import modify_graph_objects as mgo
import copy
import numpy as np

def is_bell_pair(g: Graph) -> bool:
    if g.number_of_nodes() == 2:
        if g.number_of_edges() == 1:
            return True
        else:
            return False
    elif g.number_of_nodes() > 0:
        if g.number_of_edges() > 0:
            return False
        else: 
            raise ValueError("current subgraph {} is fully disconnected (no edges)".format(g.nodes))
    else:
        raise ValueError("current graph object has invalid number of nodes {}".format(g.number_of_nodes()))
    
def is_single_qubit_graph(G: Graph) -> bool:
    if G.number_of_nodes() == 1:
        return True
    else:
        return False
    
def validate_input(circ: QuantumCircuit, graph: Graph, center_in: int, center_fin: int) -> None:
    if center_in not in graph.nodes:
        print("graph nodes {}".format(graph.nodes))
        print("center_in {}".format(center_in))
        raise ValueError("initial center is not found in graph nodes!")
    if center_fin not in graph.nodes:
        print("graph nodes {}".format(graph.nodes))
        print("center_fin {}".format(center_fin))
        raise ValueError("new center is not found in graph nodes!")
    if circ.num_qubits < max(list(graph.nodes)):
        print("graph nodes {}".format(graph.nodes))
        print(circ.draw())
        raise ValueError("Not all nodes are contained in input circuit")

def shift_centers(circ: QuantumCircuit, graph: Graph, center_in: int, center_fin: int) -> Tuple[QuantumCircuit, Graph]:
    validate_input(circ, graph, center_in, center_fin)
    if is_single_qubit_graph(graph):
        # if graph contains just a single qubit, old and new center must coincide. Do nothing in this case!
        if center_in != center_fin:
            print("center_in {}".format(center_in))
            print("center_fin {}".format(center_fin))
            raise ValueError("Unexpected missmatch of old and new center for single-qubit graph!")
        return circ, graph
    elif is_bell_pair(graph):
        # if graph is a bell pair, center is not well defined and no center shifting is needed.
        return circ, graph
    elif center_in == center_fin:
        # if old and new center coincide, do nothing.
        return circ, graph 
    else:
        leaf_qubits= copy.deepcopy(list(graph.nodes))
        leaf_qubits.remove(center_in)

        # LC about the old center
        circ.sx(center_in)
        for q in leaf_qubits:
            #circ.s(q)
            circ.rz(-np.pi/2,q)

        leaf_qubits= copy.deepcopy(list(graph.nodes))
        leaf_qubits.remove(center_fin)
        # LC about the new center
        circ.sx(center_fin)
        for q in leaf_qubits:
            #circ.s(q)
            circ.rz(-np.pi/2,q)

        # update graph object
        graph = mgo.update_graph_center(graph, center_fin)

        return circ, graph
    
def shift_centers_circ(circ: QuantumCircuit, graph: Graph, center_in: int, center_fin: int) -> QuantumCircuit:
    validate_input(circ, graph, center_in, center_fin)
    if is_single_qubit_graph(graph):
        # if graph contains just a single qubit, old and new center must coincide. Do nothing in this case!
        if center_in != center_fin:
            print("center_in {}".format(center_in))
            print("center_fin {}".format(center_fin))
            raise ValueError("Unexpected missmatch of old and new center for single-qubit graph!")
        return circ
    elif is_bell_pair(graph):
        # if graph is a bell pair, center is not well defined and no center shifting is needed.
        return circ
    elif center_in == center_fin:
        # if old and new center coincide, do nothing.
        return circ
    else:
        leaf_qubits= copy.deepcopy(list(graph.nodes))
        leaf_qubits.remove(center_in)

        # LC about the old center
        circ.sx(center_in)
        for q in leaf_qubits:
            #circ.s(q)
            circ.rz(-np.pi/2,q)

        leaf_qubits= copy.deepcopy(list(graph.nodes))
        leaf_qubits.remove(center_fin)
        # LC about the new center
        circ.sx(center_fin)
        for q in leaf_qubits:
            #circ.s(q)
            circ.rz(-np.pi/2,q)

        return circ