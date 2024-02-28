from qiskit import QuantumCircuit
from networkx import Graph
from modify_graph_objects import mgo


def shift_centers(circ: QuantumCircuit, graph: Graph, center_in: int, center_fin: int) -> QuantumCircuit:
    # shift away from the initial center
    circ.h(center_in)

    # shift to the final center
    circ.h(center_fin)
    
    graph = mgo.update_graph_center(graph, center_fin)
    return circ, graph