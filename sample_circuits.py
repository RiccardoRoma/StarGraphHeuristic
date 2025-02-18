import numpy as np
import matplotlib.pyplot as plt
import pickle
from Qiskit_input_graph import MergePattern
import modify_graph_objects as mgo
from networkx import Graph
from qiskit import QuantumCircuit
import os
import glob
import create_ghz_state_circuit as cgsc
import simulation_utils as utils
from qiskit.transpiler.passes import RemoveBarriers
from IBM_hardware_data import generate_layout_graph
import yaml
from qiskit import qpy
from typing import Union
from tqdm import tqdm
import copy
import re

# properties for graph construction
graph_str = "ibm" # ibm, reg_square_latt, random
# create layout graphs separately and load here just the pickle files
# create graphs for IBM 127, reg_square_latt (105) & 400, random 400 for different p
# Would be better to create for every ghz_state_size a seperate graph such that we only need to load the graph 
# and directly generate the circuit without extracting a subgraphw
#ghz_state_size=100



def create_circuit(graph_file: str,
                   method: str="grow") -> tuple[QuantumCircuit, Graph]:
    """Create a circuit from a given layout graph. The circuit is created using the specified method.
    The valid methods are "grow", "merge_sequential", and "merge_parallel".

    Args:
        graph_file: Path to the layout graph file.
        method: Identifier string for the specific method that should be used. Defaults to "grow".

    Raises:
        FileNotFoundError: If the graph file is not found.
        ValueError: If an unknown circuit generation method is specified.

    Returns:
        tuple[QuantumCircuit, Graph]: The generated quantum circuit and the corresponding layout graph.
    """
    # load/generate layout graph
    if not os.path.isfile(graph_file):
        raise FileNotFoundError("File {} not found!".format(graph_file))
    
    with open(graph_file, "rb") as f:
        graph: Graph = pickle.load(f)

    try:
        # create circuit from layout graph
        if method=="grow":
            circ, _ = cgsc.create_ghz_state_circuit_grow(graph, max(list(graph))+1)
        elif method == "merge_sequential":
            merge_pattern =MergePattern.from_graph_sequential(graph)
            circ, _, _ = cgsc.create_ghz_state_circuit_graph(merge_pattern, max(list(graph))+1)
        elif method == "merge_parallel":
            merge_pattern =MergePattern.from_graph_parallel(graph)
            circ, _, _ = cgsc.create_ghz_state_circuit_graph(merge_pattern, max(list(graph))+1)
        else:
            raise ValueError("Unkown circuit generation method {}!".format(method))
    except Exception as e:
        print("Error in creating circuit for graph file {} with method {}!".format(graph_file, method))    
    
    return circ, graph

def eval_circuit(circuit_in: QuantumCircuit) -> dict[str, int]:
    """Evaluate a given quantum circuit and extract figures of merit.
    The figures of merit are the number of two qubit gates, the number of measurements, and the circuit depth.

    Args:
        circuit_in: Quantum circuit that should be evaluated.

    Returns:
        dict[str, int]: Dictionary containing the figures of merit.
    """
    circuit = circuit_in.copy()

    # remove all barriers
    circuit = RemoveBarriers()(circuit)
    # extract all figures of merit (number of two qubit gates, number of measurements, circuit depth)
    num_2qubit_gates = len(circuit.get_instructions("cx"))+len(circuit.get_instructions("cz"))
    num_meas = len(circuit.get_instructions("measure"))
    depth = circuit.depth()

    return {"number_two_qubit_gates": num_2qubit_gates, "number_measurements": num_meas, "circuit_depth": depth}


# circuit_method = "merge_parallel" # grow, merge_sequential, merge_parallel
# # fix method to using ghz states instead of star graph states
# 
# graph_file = "/Users/as56ohop/Documents/NAS_sync/PhD/code/ghz_state_generation_in_com_networks/ghz_generation_heuristic_alg/Saved_small_random_graphs/random_graph_n10_p0.1_erdos_renyi_copy_1.pkl"
# 
# circ, graph = create_circuit(graph_file, method=circuit_method)
# 
# fig_merit = eval_circuit(circ)
# 
# print(circ.draw())
# print(fig_merit)
# mgo.draw_graph(graph)
# 

def extract_parameters(filename: str):
    """Extract parameters from a filename. The function searches for keys n, p, and smpl in the filename.

    Args:
        filename: Filename from which parameters should be extracted.

    Returns:
        Parameters extracted from the filename.
    """
    #pattern = r".*?_n(\d+)_p([\d\.eE+-]+)_.*?_smpl(\d+)\.pkl"
    pattern = r".*?_n(\d+)_p([\d\.eE+-]+)_.*?_smpl(\d+).*?"
    match = re.match(pattern, filename)
    
    if match:
        n = int(match.group(1))
        p = float(match.group(2)) if '.' in match.group(2) or 'e' in match.group(2).lower() else int(match.group(2))
        smpl = int(match.group(3))
        return n, p, smpl
    else:
        #raise ValueError("Filename format does not match expected pattern.")
        #pattern2 = r".*?_n(\d+).*?_smpl(\d+)\.pkl"
        pattern2 = r".*?_n(\d+).*?_smpl(\d+).*?"
        match2 = re.match(pattern2, filename)

        if match2:
            n = int(match2.group(1))
            smpl = int(match2.group(2))
            return n, None, smpl


def run_circuit_sampling(graph_dir: str, 
                         circuit_method: str, 
                         out_dir: Union[str, None] = None, 
                         overwrite_results: bool = False):
    """Run the circuit sampling for all graph files in a given directory. The circuit sampling is performed using the specified method.

    Args:
        graph_dir: Directory containing the graph files.
        circuit_method: Identifier string for the specific method that should be used.
        out_dir: Optional output directory to save the sampling results. Defaults to None, i.e., save results in the graph directory.
        overwrite_results: Bool flag to overwrite already existing sampling results. Defaults to False.

    Raises:
        ValueError: If the graph directory does not exist.
        ValueError: If the output directory does not exist.
    """

    if not os.path.isdir(graph_dir):
        raise ValueError("Directory {} was not found!".format(graph_dir))
    
    # Default output directory
    if out_dir is None:
        out_dir = os.path.join(graph_dir, "circuit_sampling_results")
        copy_input_graphs = False
    else:
        copy_input_graphs = True

    # Check if output directory exists
    if not os.path.isdir(out_dir):
        # if not create it and all non-existing parent directories
        os.makedirs(out_dir)

    # create a sub-directory for the used method
    if not os.path.isdir(os.path.join(out_dir, "circuit_method_{}".format(circuit_method))):
        os.mkdir(os.path.join(out_dir, "circuit_method_{}".format(circuit_method)))

    result_dir = os.path.join(out_dir, "circuit_method_{}".format(circuit_method))

    all_foms = {}
    qubit_nums = []

    for file_path in tqdm(glob.glob(os.path.join(graph_dir, "*.pkl"))):
        fname = os.path.basename(file_path) # get current graph file name

        # get parameters from filename 
        n, p, smpl = extract_parameters(fname)

        fname, _ = os.path.splitext(fname) # split extension

        # filenames for the results
        fname_circ = fname+"_circ_method_{}.qpy".format(circuit_method)
        fname_foms = fname+"_circ_method_{}_results.yaml".format(circuit_method)

        # Check if results exist already and if should be overwritten
        if os.path.exists(os.path.join(result_dir, fname_foms)) and not overwrite_results:
            print("Result file {} already exists. Skip this graph file!".format(os.path.join(result_dir, fname_foms)))
            continue
        elif os.path.exists(os.path.join(result_dir, fname_foms)) and overwrite_results:
            print("Result file {} already exists but will be overwritten!".format(os.path.join(result_dir, fname_foms)))

        if os.path.exists(os.path.join(result_dir, fname_circ)) and not overwrite_results:
            print("Result file {} already exists. Skip this graph file!".format(os.path.join(result_dir, fname_circ)))
            continue
        elif os.path.exists(os.path.join(result_dir, fname_circ)) and overwrite_results:
            print("Result file {} already exists but will be overwritten!".format(os.path.join(result_dir, fname_circ)))
        
        # sample the circuit
        curr_circ, curr_graph = create_circuit(file_path, circuit_method)

        # consistency check
        if curr_circ.num_qubits != n:
            raise ValueError("Number of qubits in the circuit does not match graph size!")
        
        curr_foms = eval_circuit(curr_circ)

        # save current results for visualization
        if not all_foms:
            # if all_foms dictionary is empty, add all keys and set values to lists
            for k, v in curr_foms.items():
                all_foms[k] = [v]
        else:
            for k, v in curr_foms.items():
                all_foms[k].append(v)
        qubit_nums.append(curr_circ.num_qubits)

        result_dic = copy.deepcopy(curr_foms)
        result_dic["num_qubits"] = n
        if p is not None:
            result_dic["p_value"] = p

        # save circuits and foms (and graph files)
        with open(os.path.join(result_dir, fname_circ), 'wb') as fd:
            qpy.dump(curr_circ, fd)
        with open(os.path.join(result_dir, fname_foms), "w") as f:
            yaml.dump(result_dic, f)

        if copy_input_graphs:
            fname_graph = os.path.basename(file_path)
            with open(os.path.join(result_dir, fname_graph), "wb") as f:
                pickle.dump(curr_graph, f)

def create_data_subset(result_dir: str, 
                       output_dir: str, 
                       n: Union[int, None] = None, 
                       p: Union[float, None] = None):
    """Create a subset of data files from a given directory. The subset is created based on the number of qubits and the p value.

    Args:
        result_dir: Directory containing the data files.
        output_dir: Directory into which the subset should be saved.
        n: Certain number of qubits that the subset should contain. Defaults to None, i.e., no restriction on the number of qubits.
        p: Certain p value that the subset should contain. Defaults to None, i.e., no restriction on the p value.

    Raises:
        ValueError: If the result directory does not exist.
    """
    if not os.path.isdir(result_dir):
        raise ValueError("Directory {} was not found!".format(result_dir))
    
    # Default output directory
    if not os.path.isdir(output_dir):
        # if not create it and all non-existing parent directories
        os.makedirs(output_dir)

    for file_path in tqdm(glob.glob(os.path.join(result_dir, "*.yaml"))):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        if n is not None and data["num_qubits"] != n:
            continue
        if p is not None and data.get("p_value") is not None and data["p_value"] != p:
            continue

        fname = os.path.basename(file_path)
        with open(os.path.join(output_dir, fname), "w") as f:
            yaml.dump(data, f)


def load_yaml(file_path: str) -> dict:
    """Load a yaml file and return the content as a dictionary.

    Args:
        file_path: Path to the yaml file.

    Raises:
        FileNotFoundError: If the file is not found.

    Returns:
        dict: Dictionary containing the content of the yaml file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File {} not found!".format(file_path))
    
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def plot_all_datasets(result_dirs: list[str],
                      labels: list[str],
                      xdata_str: str = "num_qubits",
                      xlabel: str = "Number of qubits",
                      fig_size: tuple[int, int] = (16,9),
                      title: str = "",
                      fname_pre: str = ""):
    """Plot all datasets in given directories. The datasets are plotted based on the x-axis data and the labels.

    Args:
        result_dirs: List of directories containing the data files.
        labels: List of labels for the datasets.
        xdata_str: Identifier string what data property should be plotted on the x-axis. Defaults to "num_qubits".
        xlabel: Label of the x-axis. Defaults to "Number of qubits".
        fig_size: Size of each figure window. Defaults to (16,9).
        title: Title stirng for each figure window. Defaults to "".
        fname_pre: Preamble for the files to save the plots to. The Preamble should also contain the path to the desired directory. Defaults to "", i.e., no saving.

    Raises:
        KeyError: If the x-axis data property is not found in the data files.
    """
    colors = ["blue", "red", "green", "orange"]
    plots_figs = {}
    for dataset_dir, label in zip(result_dirs, labels):
        if os.path.isdir(dataset_dir):  # Ensure it's a directory
            dataset = {}
            xdata = []
            for yaml_file in sorted(os.listdir(dataset_dir)):
                if yaml_file.endswith(".yaml") or yaml_file.endswith(".yml"):
                    file_path = os.path.join(dataset_dir, yaml_file)
                    data: dict = load_yaml(file_path)
                    if xdata_str not in data:
                        raise KeyError("Key {} used for x-axis not found in data file {}!".format(xdata_str, file_path))
                    if not dataset:
                        for k,v in data.items():
                            if k != xdata_str:
                                dataset[k] = [v]
                            else:
                                xdata.append(v)
                    else:
                        for k, v in data.items():
                            if k != xdata_str:
                                dataset[k].append(v)
                            else:
                                xdata.append(v)
            
            for k, v in dataset.items():
                if plots_figs.get(k) is None:
                    plots_figs[k] = {}
                plots_figs[k][label] = (xdata, v)
    
    for k,v in plots_figs.items():
        plt.figure(figsize=fig_size)
        curve_cnt=0
        for label, data in v.items():
            add_data_average_plot(np.asarray(data[0]), np.asarray(data[1]), colors[curve_cnt], label=label)
            curve_cnt += 1
        plt.xlabel(xlabel)
        plt.ylabel(k)
        plt.grid()
        plt.legend()
        plt.title(title)

        if len(fname_pre) > 0:
            fname = fname_pre + "_{}.pdf".format(k)
            plt.savefig(fname, bbox_inches="tight")
    plt.show()
            
            

# Example usage
#base_directory = "path/to/datasets"  # Change this to your dataset directory
#plot_all_datasets(base_directory)


def add_data_average_plot(xdata, ydata, color="blue", label=None):
    """Add a plot to the current figure with averaged y-values for each unique x-value, with error bars representing standard deviation.

    Args:
        xdata: Data plotted on the x-axis
        ydata: Data plotted on the y-axis
        color: Color of the curves. Defaults to "blue".
        label: Curve label. Defaults to None.
    """
    # Find unique x-values and compute mean and standard deviation of y-values
    unique_x = np.unique(xdata)
    averaged_y = np.array([np.mean(ydata[xdata == val]) for val in unique_x])
    std_dev_y = np.array([np.std(ydata[xdata == val]) for val in unique_x])

    # add data plot to figure
    if label:
        label = label + ", mean"
    else:
        label = "mean"
    plt.errorbar(unique_x, averaged_y, yerr=std_dev_y, fmt='o', color=color, label=label, capsize=5)
    plt.plot(unique_x, averaged_y, linestyle='-', color=color, alpha=0.6)


def plot_averaged_data_foms(num_qubits: list, 
                            all_foms: dict,
                            fig_size: tuple[int, int] = (16,9), 
                            fname_pre: str = "",
                            new_fig: bool = True,
                            show_plots: bool = True):
    """
    Plots the averaged y-values for each unique x-value, 
    separately for each dataset in the dictionary, with error bars representing standard deviation.

    Parameters:
        num_qubits (list or np.array): different number of qubits in the circuit
        all_foms (dict): figure of merits dictionary where keys corresponds to the name of the fom 
                       and the value corresponds to the list of values of the fom
        fig_size (tuple[int, int]): Size of the figure windows
        fname_pre (str): Preamble string of the filename to save the plots
        new_fig: Bool flag to create a new figure.
        show_plots (bool): flag to show plots in the end
    """
    x_data = np.array(num_qubits)

    for key, y_data in all_foms.items():
        y_data = np.array(y_data)
        
        # Find unique x-values and compute mean and standard deviation of y-values
        unique_x = np.unique(x_data)
        averaged_y = np.array([np.mean(y_data[x_data == val]) for val in unique_x])
        std_dev_y = np.array([np.std(y_data[x_data == val]) for val in unique_x])

        # Create a new figure for each dataset
        if new_fig:
            plt.figure(figsize=fig_size)
        plt.scatter(x_data, y_data, marker="x", color="blue", label="Raw data")
        #plt.errorbar(unique_x, averaged_y, yerr=std_dev_y, fmt='o', color='blue', label='Averaged Data', capsize=5)
        #plt.plot(unique_x, averaged_y, linestyle='-', color='blue', alpha=0.6)
        add_data_average_plot(x_data, y_data, color="blue")
        plt.xlabel('number of qubits')
        plt.ylabel(f'Averaged {key}')
        plt.legend()
        if len(fname_pre) > 0:
            fname = fname_pre + "_{}.pdf".format(key)
            plt.savefig(fname, bbox_inches="tight")
        #plt.show()
    
    if show_plots:
        plt.show()

if __name__ == "__main__":
    #circuit_method = "merge_parallel"
    #circuit_method = "grow"
    # #graph_dir = "/Users/as56ohop/Documents/NAS_sync/PhD/code/ghz_state_generation_in_com_networks/ghz_generation_heuristic_alg/Saved_small_random_graphs/sample_circuits_test"
    # #graph_dir = os.path.join(os.getcwd(), "Saved_small_random_graphs/sample_circuits_test")
    #graph_dir = os.path.join(os.getcwd(), "graph_samples/random_graph_endros_renyi_1/")
# 
    #out_dir = None
    # out_dir = os.path.join(os.getcwd(), "simulation_results/sample_circuits/random_graphs_endros_renyi_p0.0/")
    # 
    #run_circuit_sampling(graph_dir, circuit_method, out_dir=out_dir, overwrite_results=True)

    # result_dirs = [os.path.join(os.getcwd(), "simulation_results/sample_circuits/random_graphs_endros_renyi_p0.4/circuit_method_grow"),
    #                os.path.join(os.getcwd(), "simulation_results/sample_circuits/random_graphs_endros_renyi_p0.4/circuit_method_merge_parallel")]
    # labels = ["state growing", "merge_parallel"]
    # plot_all_datasets(result_dirs, labels=labels)

    # Example usage
    #filename = "random_graph_n10_p0.5_endros_renyi_smpl3_something.yaml"
    #print(extract_parameters(filename))  # Output: (10, 0.5, 3)

    #filename = "random_graph_n10_smpl3.pkl"
    #print(extract_parameters(filename))  # Output: (10, 3)

    #result_dir = os.path.join(os.getcwd(), "graph_samples/random_graph_endros_renyi_1/circuit_sampling_results/circuit_method_grow")
    #output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n400/circuit_method_grow")

    #create_data_subset(result_dir, output_dir, n=400)

    output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n57/circuit_method_grow"),
                  os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n153/circuit_method_grow"),
                  os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n272/circuit_method_grow"),
                  os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n400/circuit_method_grow")]

    plot_all_datasets(output_dir, ["n=57", "n=153", "n=272", "n=400"], xdata_str="p_value", xlabel="p value", title="Random graph sampling evaluation", fname_pre="")

