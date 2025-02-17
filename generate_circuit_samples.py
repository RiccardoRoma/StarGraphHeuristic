import os
import pickle
import numpy as np
import networkx as nx
from networkx import Graph
from qiskit.providers import BackendV2
from typing import Union
from IBM_hardware_data import find_connected_subgraph_with_lowest_weight
from Qiskit_input_graph import generate_random_graph
import random
import warnings
from tqdm import tqdm

def get_ibm_layout_graph(backend: BackendV2) -> Graph:
    """This function extracts the layout graph from a given qiskit.BackendV2 object

    Args:
        backend: qiskit.providers.BackendV2 object

    Returns:
        Layout Graph as a networkx.Graph object.
    """
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

    # generate graph object from nodes list and edges list
    # Init a graph G object
    G = Graph()

    # Add nodes to the graph
    G.add_nodes_from(nodes_list)

    # Add edges to the graph
    G.add_edges_from(edges_list)

    return G

def random_connected_subgraph(G: Graph, 
                              k: int) -> Graph:
    """Generate a random connected subgraph of size k from G.

    Args:
        G: The initial graph.
        k: The number of nodes in the subgraph.

    Returns:
        A connected subgraph of size k.
    """
    # Start from a random node
    start_node = random.choice(list(G.nodes))
    
    # Use BFS/DFS to grow a connected component of size k
    visited = {start_node}
    queue = [start_node]
    
    while len(visited) < k and queue:
        current = queue.pop(0)  # BFS: queue.pop(0), DFS: queue.pop()
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)  # Randomize exploration
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
            if len(visited) == k:
                break

    return G.subgraph(visited).copy()

def create_samples_random_graph(dir: str,
                                rnd_graph_sizes: list[int],
                                p: Union[float, int],
                                samples: Union[list[int], int] = 10,
                                use_barabasi: bool = False) -> None:
    """Create samples of random graphs and save it to pickle files in a given directory.

    Args:
        dir: directory into which outpput should be saved
        rnd_graph_sizes: list of graph sizes that should be generated
        p: Measure of connectedness in the random graph
        samples: number of samples for each random graph of a certain size. Defaults to 10.
        use_barabasi: Bool flag to use Barabasi-Albert model for random graph generation. Defaults to False, i.e., use Endrös-Rényi model.

    Raises:
        FileExistsError: If a unique filename for a certain generated graph already exists.
    """
    # if directory does not exist, create the directory and all parent directories
    if not os.path.isdir(dir):
        os.makedirs(dir)

    # create proper sub-directory
    if use_barabasi:
        sub_dir = "random_graphs_barabasi_albert_p{}/".format(p)
    else:
        sub_dir = "random_graphs_endros_renyi_p{}/".format(p)

    output_dir = os.path.join(dir, sub_dir)

    if os.path.isdir(output_dir):
        warnings.warn("Sub-directory {} does already exist in parent directory {}!".format(sub_dir, dir))
    else:
        os.mkdir(output_dir)
    
    # check input type of samples
    if isinstance(samples, int):
        # samples are [0,1,...,samples-1]
        samples_itr = range(samples)
    elif isinstance(samples, list):
        # samples are input list
        samples_itr = samples
    else:
        raise ValueError("Input samples must be a integer or a list of integers!")

    # Iterate through number of qubits and samples
    print("Generate random graphs for {} different graph sizes between {} and {}, p={}, samples/size={}.".format(len(graph_sizes), min(graph_sizes),max(graph_sizes), p, len(samples_itr)))    
    for graph_size in tqdm(rnd_graph_sizes):
        for smpl in samples_itr:
            # create filename for current graph
            if use_barabasi:
                fname_graph = "random_graph_n{}_p{}_barabasi_albert_smpl{}.pkl".format(graph_size, p, smpl)
            else:
                fname_graph = "random_graph_n{}_p{}_endros_renyi_smpl{}.pkl".format(graph_size, p, smpl)
            
            # create current random graph
            curr_graph = generate_random_graph(graph_size, p, use_barabasi, show_plot=False)

            # Check if current filename already exists in output directory
            if os.path.isfile(os.path.join(output_dir, fname_graph)):
                raise FileExistsError("File {} does already exist in output directory {}!".format(fname_graph, output_dir))
            
            with open(os.path.join(output_dir, fname_graph), "wb") as f:
                pickle.dump(curr_graph, f)

if __name__=="__main__":
    graph_sizes = np.linspace(10, 400, 50, dtype=int)
    dir = os.path.join(os.getcwd(), "graph_samples/")
    p_list = [0.1, 0.4, 0.7, 1.0]
    #p_list = [0.1]
    samples = 10

    for p in p_list:
        create_samples_random_graph(dir, graph_sizes, p, samples=samples, use_barabasi=False)