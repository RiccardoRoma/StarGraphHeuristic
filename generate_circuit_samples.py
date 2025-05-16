import os
import pickle
import numpy as np
import networkx as nx
from networkx import Graph
from qiskit.providers import BackendV2
from typing import Union
from Qiskit_input_graph import generate_random_graph
import random
import warnings
from tqdm import tqdm
import itertools
import simulation_utils as utils
import modify_graph_objects as mgo

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

def random_connected_subgraph_2(G: Graph, k: int) -> Graph:
    if k == 0:
        return G.subgraph([]).copy()
    
    visited = {random.choice(list(G.nodes))}
    queue = [next(iter(visited))]
    
    while len(visited) < k and queue:
        # Randomly select next node from frontier
        current = queue.pop(random.randrange(len(queue)))
        
        # Process neighbors in random order
        for neighbor in random.sample(list(G.neighbors(current)), len(list(G.neighbors(current)))):
            if len(visited) == k: break
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return G.subgraph(visited).copy()

def create_samples_ibm_layout_graph(output_dir: str,
                                    graph_sizes: list[int],
                                    backend_str: str,
                                    use_premium_access: bool = False,
                                    samples: Union[list[int], int] = 10) -> None:
    """Create samples of random connected subgraphs from IBM Quantum layout graph and save it to pickle files in a given directory.

    Args:
        output_dir: directory into which output should be saved
        graph_sizes: list of graph sizes that should be generated
        backend_str: name of the IBM Quantum backend
        use_premium_access: Bool flag if IBM Quantum premium access credentials should be used from .env file. Defaults to False.
        samples: Number of samples to generate for each graph size. Defaults to 10.

    Raises:
        ValueError: If input samples is not an integer or a list of integers.
        FileExistsError: If a unique filename for a certain generated graph already exists.
    """
    # if directory does not exist, create the directory and all parent directories
    if os.path.isdir(output_dir):
        warnings.warn("Result directory {} does already exist!".format(output_dir))
    else:
        os.mkdir(output_dir)

    # check input type of samples
    if isinstance(samples, int):
        # samples are [0,1,...,samples-1]
        samples_itr = list(range(samples))
    elif isinstance(samples, list):
        # samples are input list
        samples_itr = samples
    else:
        raise ValueError("Input samples must be a integer or a list of integers!")
    
    # setup backend options
    backend_opt = {"backend_str": backend_str, 
                   "noise_model_id": 0, 
                   "fname_noise_model": "", 
                   "noise_model_str": "",
                   "coupling_map_id": 0,
                   "fname_coupling_map": "",
                   "coupling_map_str": "",
                   "native_basis_gates_str": "",
                   "run_locally": True}

    # load IBM Quantum credentials
    service = utils.load_ibm_credentials(premium_access=use_premium_access)
    
    # load backend from passmanager calibration
    backend = utils.get_backend(service, backend_opt)

    # get layout graph from backend
    layout_graph = get_ibm_layout_graph(backend)

    # Iterate through number of qubits and samples
    param_iter = list(itertools.product(graph_sizes, samples_itr))
    print("Generate {} random connected subgraphs for {} different graph sizes between {} and {}, from initial layout graph of ibm backend {}.".format(len(param_iter), len(graph_sizes), min(graph_sizes),max(graph_sizes), backend_str))
    for graph_size, smpl in tqdm(param_iter):
        # create filename for current graph
        fname_graph = "ibm_layout_graph_n{}_smpl{}.pkl".format(graph_size, smpl)
        
        # create current random graph
        #curr_graph = random_connected_subgraph(layout_graph, graph_size)
        curr_graph = random_connected_subgraph_2(layout_graph, graph_size)

        # Check if current filename already exists in output directory
        if os.path.isfile(os.path.join(output_dir, fname_graph)):
            raise FileExistsError("File {} does already exist in output directory {}!".format(fname_graph, output_dir))
        
        with open(os.path.join(output_dir, fname_graph), "wb") as f:
            pickle.dump(curr_graph, f)
    
def create_rectangular_grid_graph(height: int,
                                  width: int) -> Graph:
    """Create a rectangular grid graph with given height and width. This Layout graph is used for different Quantum hardware backends, e.g., Google Willow, Rigetti.

    Args:
        height: height of the grid
        width: width of the grid

    Returns:
        A rectangular grid graph.
    """
    # Init a graph G object
    G = Graph()
    # Add nodes to the graph
    node_counter = 0
    node_map = {}
    for i in range(height):
        for j in range(width):
            G.add_node(node_counter)
            node_map[(i, j)] = node_counter
            node_counter += 1
    
    # Add edges to the graph
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                G.add_edge(node_map[(i, j)], node_map[(i + 1, j)])
            if j < width - 1:
                G.add_edge(node_map[(i, j)], node_map[(i, j + 1)])
    
    return G

def create_samples_rectangular_grid_graph(output_dir: str,
                                          graph_sizes: list[tuple[int, int]],
                                          samples: Union[list[int], int] = 10) -> None:
    """Create samples of random connected subgraphs from rectangular grid graphs and save it to pickle files in a given directory.

    Args:
        output_dir: directory into which output should be saved
        graph_sizes: list of graph sizes that should be generated
        samples: number of samples for each graph of a certain size. Defaults to 10.
        graph_sizes: _description_
        samples: _description_. Defaults to 10.
    """
    # if directory does not exist, create the directory and all parent directories
    if os.path.isdir(output_dir):
        warnings.warn("Result directory {} does already exist!".format(output_dir))
    else:
        os.makedirs(output_dir)

    # check input type of samples
    if isinstance(samples, int):
        # samples are [0,1,...,samples-1]
        samples_itr = list(range(samples))
    elif isinstance(samples, list):
        # samples are input list
        samples_itr = samples
    else:
        raise ValueError("Input samples must be a integer or a list of integers!")
    
    # find maximum graph size
    max_graph_size = max(graph_sizes)

    # Create rectangular grid graph that is larger than the maximum graph size
    width = np.ceil(np.sqrt(max_graph_size))+2
    rect_grid_graph = create_rectangular_grid_graph(int(width), int(width))
    
    # Iterate through number of qubits and samples
    param_iter = list(itertools.product(graph_sizes, samples_itr))
    print("Generate {} random connected subgraphs for {} different graph sizes.".format(len(param_iter), len(graph_sizes)))
    for graph_size, smpl in tqdm(param_iter):
        # create filename for current graph
        fname_graph = "rectangular_grid_graph_n{}_smpl{}.pkl".format(graph_size, smpl)
        
        # create current random graph
        #curr_graph = random_connected_subgraph(rect_grid_graph, graph_size)
        curr_graph = random_connected_subgraph_2(rect_grid_graph, graph_size)

        # Check if current filename already exists in output directory
        if os.path.isfile(os.path.join(output_dir, fname_graph)):
            raise FileExistsError("File {} does already exist in output directory {}!".format(fname_graph, output_dir))
        
        with open(os.path.join(output_dir, fname_graph), "wb") as f:
            pickle.dump(curr_graph, f)


def create_samples_random_graph(output_dir: str,
                                graph_sizes: list[int],
                                p_vals: Union[list[float], list[int]],
                                samples: Union[list[int], int] = 10,
                                use_barabasi: bool = False) -> None:
    """Create samples of random graphs and save it to pickle files in a given directory.

    Args:
        output_dir: directory into which output should be saved
        graph_sizes: list of graph sizes that should be generated
        p_vals: List of p values. The p value is a measure of connectedness in the random graph.
        samples: number of samples for each random graph of a certain size. Defaults to 10.
        use_barabasi: Bool flag to use Barabasi-Albert model for random graph generation. Defaults to False, i.e., use Endrös-Rényi model.

    Raises:
        FileExistsError: If a unique filename for a certain generated graph already exists.
    """
    # if directory does not exist, create the directory and all parent directories
    if os.path.isdir(output_dir):
        warnings.warn("Result directory {} does already exist!".format(output_dir))
    else:
        os.mkdir(output_dir)
    
    # check input type of samples
    if isinstance(samples, int):
        # samples are [0,1,...,samples-1]
        samples_itr = list(range(samples))
    elif isinstance(samples, list):
        # samples are input list
        samples_itr = samples
    else:
        raise ValueError("Input samples must be a integer or a list of integers!")

    # Iterate through number of qubits and samples
    param_iter = list(itertools.product(graph_sizes, p_vals, samples_itr))
    print("Generate {} random graphs for {} different graph sizes between {} and {} and {} different p values between {} and {}, samples/run={}.".format(len(param_iter), len(graph_sizes), min(graph_sizes),max(graph_sizes), len(p_vals), min(p_vals), max(p_vals), len(samples_itr)))    
    for graph_size, p, smpl in tqdm(param_iter):
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
    # graph_sizes = np.linspace(10, 400, 50, dtype=int)
    # dir = os.path.join(os.getcwd(), "graph_samples/random_graph_endros_renyi_1/")
    # #p_list = [0.1, 0.4, 0.7, 1.0]
    # p_list = np.round(np.linspace(0.05, 1.0, 20), 2) # round to two decimal places
    # #p_list = [0.1]
    # # samples = 10
    # samples = list(range(10, 100)) # increase samples to 100
#
    # create_samples_random_graph(dir, graph_sizes, p_list, samples=samples, use_barabasi=False)

    #graph_sizes = np.linspace(10, 127, 10, dtype=int)
    graph_sizes = np.linspace(6, 16, 6, dtype=int)
    dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_ibm_brisbane_2/")
    samples = 100
    #samples = list(range(10, 101)) # increase samples to 100
# 
    create_samples_ibm_layout_graph(dir, graph_sizes, "ibm_brisbane", use_premium_access=False, samples=samples)

    # height = 4
    # width = 6
    # graph = create_rectangular_grid_graph(height, width)
#
    # mgo.draw_graph(graph, title="Rectangular Grid Graph", layout="graphviz")

    # graph_sizes = np.linspace(10, 400, 50, dtype=int)
    # dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_rect_grid_1/")
    # # samples = 10
    # samples = list(range(10, 100)) # increase samples to 100
# # 
    # create_samples_rectangular_grid_graph(dir, graph_sizes, samples=samples)
