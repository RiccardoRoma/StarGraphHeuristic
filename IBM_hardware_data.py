import networkx as nx
from networkx import Graph
from itertools import combinations
from dotenv import load_dotenv
from typing import Callable, Dict, List, Optional, Tuple, Union
from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit.transpiler import CouplingMap
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

def generate_layout_graph(backend: BackendV2,
                          add_errs_as_weigths: bool = True,
                          k: Union[int, None] = None) -> Graph:
    # get nodes list
    nodes_list = list(range(backend.num_qubits))
    # get edges list
    edges_list = list(backend.coupling_map.get_edges())

    # generate graph object from nodes list and edges list
    # Init a graph G object
    G = Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(nodes_list)
    
    # Add edges to the graph
    G.add_edges_from(edges_list)

    # check measurement error of qubits (throw those out with too large error)
    meas_err_thrs = 0.4 # error threshold
    for i in range(backend.num_qubits):
        # get current measurement error
        meas_err = backend.target["measure"][(i,)].error
        # check if above threshold
        if meas_err > meas_err_thrs:
            # delete node from graph if this is the case
            G.remove_node(i)
        if add_errs_as_weigths:
            # add error [%] as a node weight 
            G.nodes[i]['weight'] = round(meas_err * 100, 2)
    
    # check two-qubit gate error (throw out those edges with too large error)
    two_q_gate_err_thrs = 0.4 # error threshold
    # check available two-qubit gate
    if "cx" in backend.operation_names:
        two_qubit_gate_str = "cx"
    elif "ecr" in backend.operation_names:
        two_qubit_gate_str = "ecr"
    # iterate through all edges
    for i,j in G.edges:
        # get current two-qubit gate error
        two_q_gate_err = backend.target[two_qubit_gate_str][(i,j)].error
        # check if above threshold
        if two_q_gate_err > two_q_gate_err_thrs:
            # remove this edges
            G.remove_edge(i, j)
        if add_errs_as_weigths:
            # add error [%] as an edges weight
            G[i][j]['weight'] = round(two_q_gate_err * 100, 2)

    # check if some nodes got disconnected
    isolated_nodes = list(nx.isolates(G))
    # remove isolated nodes
    G.remove_nodes_from(isolated_nodes)


    if k is not None:
        ## To-Do: This can be more efficient, by just saving the current subgraph with the lowest error
        subgraphs = []
        for nodes in combinations(G.nodes, k):
            G_sub = G.subgraph(nodes)
            if nx.is_connected(G_sub):
                subgraphs.append(G_sub)

        # pick best subgraph
        return get_best_subgraph(subgraphs, backend)
    else:
        return G

def get_best_subgraph(subgraphs: List[Graph],
                      backend: BackendV2) -> Graph:
    ## To-Do: add option to return subgraph with lowest mean CNOT error and/or readout error
    return subgraphs[0]