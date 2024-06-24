import networkx as nx
from networkx import Graph
from dotenv import load_dotenv
from typing import Callable, Dict, List, Optional, Tuple, Union
from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
#import modify_graph_objects as mgo
import networkx as nx


# This is a small code for getting number of qubits and couplinng map from a certain IBM device.
 
# service = QiskitRuntimeService(channel="ibm_quantum")
# 
# backend= service.backend("ibm_brisbane")
# # backend = service.backend("ibmq_qasm_simulator")
# num_qubits= backend.num_qubits
# coupling_map= backend.coupling_map
# 
# print('number of qubits', num_qubits)
# print('Coupling map', coupling_map)
# 
# # This givees the dynamic properties of every qubit such as readout error
# for i in range(num_qubits):
#     mp=backend.target["measure"][(i,)]
#     print(mp)

def generate_layout_graph(backend: BackendV2) -> Graph:
    # get nodes list
    nodes_list = list(range(backend.num_qubits))
    # get edges list
    edges_list = list(backend.coupling_map.get_edges())

    ## To-Do: throw out inactive qubits (higher measurement error) and edges
    # qubit readout error
    # for i in range(num_qubits):
    #     mp=backend.target["measure"][(i,)].error
    # cx error
    # backend.target["cx"][(i,j)].error
    ##

    # generate graph object from nodes list and edges list
    # Init a graph G object
    G = Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(nodes_list)
    
    # Add edges to the graph
    G.add_edges_from(edges_list)
    
    return G
