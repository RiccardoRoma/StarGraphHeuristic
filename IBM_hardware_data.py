import networkx as nx
from networkx import Graph
from itertools import combinations
from dotenv import load_dotenv
from typing import Callable, Dict, List, Optional, Tuple, Union
from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, IBMBackend
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel
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
    # check available two-qubit gate
    if "cx" in backend.operation_names:
        two_qubit_gate_str = "cx"
    elif "ecr" in backend.operation_names:
        two_qubit_gate_str = "ecr"
    # get edges list
    coupling_map = backend.coupling_map
    if coupling_map is None:
        # if couling map is None, create coupling map from target 
        edges_list = []
        for i in nodes_list:
            for j in range(i,len(nodes_list)):
                if (i,j) in list(backend.target[two_qubit_gate_str].keys()):
                    edges_list.append((i,j))
    else:
        edges_list = list(backend.coupling_map.get_edges())

    noise_model = NoiseModel.from_backend(backend)
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
        if isinstance(backend, IBMBackend):
            meas_err = backend.target["measure"][(i,)].error
        else:
            if noise_model.is_ideal():
                meas_err = 0.0
            elif i in noise_model.noise_qubits:
                meas_err = noise_model._local_readout_errors[(i,)].probabilities[0,1]
            else:
                meas_err = 0.0

        # check if above threshold
        if meas_err > meas_err_thrs:
            # delete node from graph if this is the case
            G.remove_node(i)
        if add_errs_as_weigths:
            # add error [%] as a node weight 
            G.nodes[i]['weight'] = round(meas_err, 4)
    
    # check two-qubit gate error (throw out those edges with too large error)
    two_q_gate_err_thrs = 0.4 # error threshold
    # iterate through all edges
    for i,j in G.edges:
        # construct both tuple representations of the current edge
        et1 = (i,j)
        et2 = (j,i)
        # get current two-qubit gate error
        if isinstance(backend, IBMBackend):
            # IBMBackend class delivers all required info in target attribute
            if et1 in backend.target[two_qubit_gate_str].keys():
                two_q_gate_err = backend.target[two_qubit_gate_str][et1].error
            elif et2 in backend.target[two_qubit_gate_str].keys():
                two_q_gate_err = backend.target[two_qubit_gate_str][et2].error
            else:
                raise KeyError("Edge tuple {},{} from layout graph not in backend target operation {} keys {}". format(i,j, two_qubit_gate_str, backend.target[two_qubit_gate_str].keys()))
        else:
            # For AerBackend/AerSimulator target attribute info maybe incomplete, thus work with noise model
            if noise_model.is_ideal():
                two_q_gate_err = 0.0
            elif two_qubit_gate_str in noise_model.noise_instructions:
                err_dict_target = backend.target[two_qubit_gate_str]
                if et1 in err_dict_target.keys():
                    two_q_gate_err = err_dict_target[et1].error
                elif et2 in err_dict_target.keys():
                    two_q_gate_err = err_dict_target[et2].error
                else:
                    raise KeyError("Edge tuple {},{} from layout graph not in backend target operation {} keys {}". format(i,j, two_qubit_gate_str, backend.target[two_qubit_gate_str].keys()))
            else:
                two_q_gate_err = 0.0
        # check if above threshold
        if two_q_gate_err > two_q_gate_err_thrs:
            # remove this edges
            G.remove_edge(i, j)
        elif add_errs_as_weigths:
            # add error [%] as an edges weight
            G[i][j]['weight'] = round(two_q_gate_err, 4)

    # check if some nodes got disconnected
    isolated_nodes = list(nx.isolates(G))
    # remove isolated nodes
    G.remove_nodes_from(isolated_nodes)


    if k is not None:
        return find_connected_subgraph_with_lowest_weight(G, k)            
    else:
        return G

def find_connected_subgraph_with_lowest_weight(graph: Graph, 
                             k: int, 
                             weight_cost_trsh: float = 0.05) -> Graph:
    if not isinstance(k, int):
        raise ValueError("k is expected to be integer!")
    
    if k > len(graph):
        raise ValueError("k cannot be larger than the number of nodes in the graph")
    
    # Function to calculate the combined weight of a subgraph
    def combined_weight_cost(subgraph: list) -> float:
        # calculate the node weigth cost
        node_weight_cost = sum(graph.nodes[n].get('weight', 0) for n in subgraph)/len(subgraph) # mean weight
        # calculate the node weigth cost
        edge_weight_cost = sum(graph[u][v].get('weight', 0) for u, v in graph.subgraph(subgraph).edges())/len(graph.subgraph(subgraph).edges()) # mean weight
        return round(node_weight_cost + edge_weight_cost, 2)
    
    def is_connected(subgraph: list):
        # Check if the subgraph is connected
        return nx.is_connected(graph.subgraph(subgraph))

    def backtrack(current_set: list,
                  best_weight_cost: list[float], 
                  best_subgraph: list[list[int]]):
        if len(current_set) == k:
            if is_connected(current_set):
                weight_cost = combined_weight_cost(current_set)
                if weight_cost < best_weight_cost[0]:
                    best_weight_cost[0] = weight_cost
                    best_subgraph[0] = current_set.copy()
            return
        
        for neighbor in set.union(*(set(graph.neighbors(node)) for node in current_set)):
            if neighbor not in current_set:
                current_set.append(neighbor)
                backtrack(current_set, best_weight_cost, best_subgraph)
                current_set.pop()

                if best_weight_cost[0] <= weight_cost_trsh:  # Early exit if we find a subgraph which has a weight cost below the set threshold
                    return
    
    curr_best_weight_cost = [float('inf')]
    curr_best_subgraph = [[]]
    for node in graph.nodes():
        backtrack([node], curr_best_weight_cost, curr_best_subgraph)
    
    return graph.subgraph(curr_best_subgraph[0])