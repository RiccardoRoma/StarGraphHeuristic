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
from qiskit.converters import circuit_to_dag
import traceback as tb
import tarfile

# properties for graph construction
#graph_str = "ibm" # ibm, reg_square_latt, random
# create layout graphs separately and load here just the pickle files
# create graphs for IBM 127, reg_square_latt (105) & 400, random 400 for different p
# Would be better to create for every ghz_state_size a seperate graph such that we only need to load the graph 
# and directly generate the circuit without extracting a subgraphw
#ghz_state_size=100



def create_circuit(graph_file: str,
                   method: str="grow",
                   substate_size_fac: Union[float, None] = None,
                   substate_size: Union[int, None] = None,
                   star_states: bool = False,
                   binary_merge: bool = False) -> tuple[QuantumCircuit, Graph, float, int]:
    """Create a circuit from a given layout graph. The circuit is created using the specified method.
    The valid methods are "grow", "merge_sequential", and "merge_parallel".

    Args:
        graph_file: Path to the layout graph file.
        method: Identifier string for the specific method that should be used. Defaults to "grow".
        substate_size_fac: Percentage factor between the target center node degree of each subgraph (star graphs) and the average degree in the initial graph. If this is not None this factor is used for the subgraph creation. Defaults to None, which means the algorithm picks always the highest degree node if substate_size is also None. This argument is only used for methods = "merge_sequential", "merge_parallel"
        substate_size: Target size of the to-be-merged substates. Defaults to None which means to don't use a similar target size for all substates. This argument is only used for methods = "merge_sequential", "merge_parallel"
        star_states: Bool flag to work with star graph states for merging. Defaults to False, which means to work directly with GHZ states. This argument is only used for methods = "merge_sequential", "merge_parallel"
        binary_merge: bool flag to use a binary merging tree or a non-binary merging tree. Defaults to False, which means to use a non-binary merging tree. This argument is only used for method = "merge_parallel"

    Raises:
        FileNotFoundError: If the graph file is not found.
        ValueError: If an unknown circuit generation method is specified.

    Returns:
        tuple[QuantumCircuit, Graph, float, int]: The generated quantum circuit, the corresponding layout graph, substate size factor and substate size (used for merging and none for growing).
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
            curr_substate_size = None
            curr_substate_size_fac = None
        elif method == "merge_sequential":
            msq, bt, curr_substate_size_fac, curr_substate_size = MergePattern.create_msq_and_merge_tree(graph, substate_size_fac=substate_size_fac, substate_size=substate_size, parallel=False, binary_merge=True)
            merge_pattern = MergePattern(graph, msq, bt)
            circ, _, _ = cgsc.create_ghz_state_circuit_graph(merge_pattern, max(list(graph))+1, star=star_states)
        elif method == "merge_parallel":
            msq, bt, curr_substate_size_fac, curr_substate_size = MergePattern.create_msq_and_merge_tree(graph, substate_size_fac=substate_size_fac, substate_size=substate_size, parallel=True, binary_merge=binary_merge)
            merge_pattern = MergePattern(graph, msq, bt)
            circ, _, _ = cgsc.create_ghz_state_circuit_graph(merge_pattern, max(list(graph))+1, star=star_states)
        else:
            raise ValueError("Unkown circuit generation method {}!".format(method))
        return circ, graph, curr_substate_size_fac, curr_substate_size
    except Exception as e:
        print("Error in creating circuit for graph file {} with method {}!".format(graph_file, method))
        tb.print_exc()
        return None, graph, None, None
    

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
                         substate_size_fac: Union[float, None] = None,
                         substate_size: Union[int, None] = None,
                         star_states: bool = False,
                         binary_merge: bool = False,
                         out_dir: Union[str, None] = None, 
                         overwrite_results: bool = False):
    """Run the circuit sampling for all graph files in a given directory. The circuit sampling is performed using the specified method.

    Args:
        graph_dir: Directory containing the graph files.
        circuit_method: Identifier string for the specific method that should be used.
        substate_size_fac: Percentage factor between the target center node degree of each subgraph (star graphs) and the average degree in the initial graph. If this is not None this factor is used for the subgraph creation. Defaults to None, which means the algorithm picks always the highest degree node if substate_size is also None. This argument is only used for methods = "merge_sequential", "merge_parallel"
        substate_size: Target size of the to-be-merged substates. Defaults to None which means to don't use a similar target size for all substates. This argument is only used for methods = "merge_sequential", "merge_parallel"
        star_states: Bool flag to work with star graph states for merging. Defaults to False, which means to work directly with GHZ states. This argument is only used for methods = "merge_sequential", "merge_parallel"
        binary_merge: bool flag to use a binary merging tree or a non-binary merging tree. Defaults to False, which means to use a non-binary merging tree. This argument is only used for method = "merge_parallel"
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

    if substate_size_fac:
        if substate_size:
            raise ValueError("Substate size and substate size factor are both not None. Use just one of them and set the other to None.")
        substate = substate_size_fac
    else:
        substate = substate_size
            
    # generate result string
    if circuit_method == "grow":
        out_str = f"circuit_method_{circuit_method}"
    elif circuit_method == "merge_sequential":
        if substate:
            out_str = f"circuit_method_{circuit_method}_substate_{substate}"
        else:
            out_str = f"circuit_method_{circuit_method}_substate_none"

        if star_states:
            out_str = out_str + "_star"
        else:
            out_str = out_str + "_ghz"
    elif circuit_method == "merge_parallel":
        out_str = f"circuit_method_{circuit_method}"
        if binary_merge:
            out_str = out_str + "_bin"
        else:
            out_str = out_str + "_non-bin"

        if substate:
            out_str = out_str + f"_substate_{substate}"
        else:
            out_str = out_str + "_substate_none"

        if star_states:
            out_str = out_str + "_star"
        else:
            out_str = out_str + "_ghz"
    else:
        raise ValueError(f"Unkown method {circuit_method} for output string generation!")

    # create a sub-directory for the used method
    if not os.path.isdir(os.path.join(out_dir, out_str)):
        os.mkdir(os.path.join(out_dir, out_str))

    result_dir = os.path.join(out_dir, out_str)

    all_foms = {}
    qubit_nums = []

    for file_path in tqdm(glob.glob(os.path.join(graph_dir, "*.pkl"))):
        fname = os.path.basename(file_path) # get current graph file name

        # get parameters from filename 
        n, p, smpl = extract_parameters(fname)

        fname, _ = os.path.splitext(fname) # split extension

        # filenames for the results
        fname_circ = fname+"_"+out_str+".qpy"
        fname_foms = fname+"_"+out_str+"_results.yaml"

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
        curr_circ, curr_graph, curr_substate_size_fac, curr_substate_size = create_circuit(file_path, circuit_method, substate_size_fac = substate_size_fac, substate_size=substate_size, star_states=star_states, binary_merge=binary_merge)

        # consistency check
        if len(curr_graph.nodes) != n:
            raise ValueError("Number of nodes in the graph {}, does not match extracted graph size {} from file {}!".format(len(curr_graph.nodes), n, file_path))
        # if curr_circ.num_qubits != n:
        #     dag = circuit_to_dag(curr_circ)
        #     num_nonidle_qubits = len([qubit for qubit in curr_circ.qubits if qubit not in dag.idle_wires()])
        #     if num_nonidle_qubits != n:
        #         print("number of nodes {}".format(len(curr_graph.nodes)))
        #         print(curr_circ.draw())
        #         num_2qubit_gates = len(curr_circ.get_instructions("cx"))+len(curr_circ.get_instructions("cz"))
        #         print("number of two-qubit gate {}".format(num_2qubit_gates))
        #         mgo.draw_graph(curr_graph)
        #         plt.show()
        #         raise ValueError("Number of non-idle qubits in the circuit {}, does not match graph size {}!".format(num_nonidle_qubits, n))
        
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

        result_dict = copy.deepcopy(curr_foms)
        result_dict["num_qubits"] = n
        if p is not None:
            result_dict["p_value"] = p
        
        if circuit_method != "grow":
            # add substate sizes to results if merging methods are used and they are not None
            if curr_substate_size_fac:
                result_dict["substate_size_fac"] = curr_substate_size_fac
            if curr_substate_size:
                result_dict["substate_size"] = curr_substate_size

        # save circuits and foms (and graph files)
        with open(os.path.join(result_dir, fname_circ), 'wb') as fd:
            qpy.dump(curr_circ, fd)
        with open(os.path.join(result_dir, fname_foms), "w") as f:
            yaml.dump(result_dict, f)

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

def create_data_subset_tar(result_dir: str,
                           output_dir: str,
                           fname_tar: str = "subset_data.tar.gz",
                           n: Union[int, None] = None,
                           p: Union[float, None] = None):
    """Create a subset of data files from a given directory and save them as a tarball in the given output directory. The subset is created based on the number of qubits and the p value.

    Args:
        result_dir: Directory containing the data files.
        output_dir: Directory into which the tarball should be saved.
        fname_tar: Name of the tarball. Defaults to "subset_data.tar.gz".
        n: Certain number of qubits that the subset should contain. Defaults to None, i.e., no restriction on the number of qubits.
        p: Certain p value that the subset should contain. Defaults to None, i.e., no restriction on the p value.

    Raises:
        ValueError: If the result directory does not exist.
    """
    if not os.path.isdir(result_dir):
        raise ValueError("Directory {} was not found!".format(result_dir))
    
    subset_files = []

    for file_path in tqdm(glob.glob(os.path.join(result_dir, "*.yaml"))):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        if n is not None and data["num_qubits"] != n:
            continue
        if p is not None and data.get("p_value") is not None and data["p_value"] != p:
            continue

        subset_files.append(file_path)

    tarball_path = os.path.join(output_dir, fname_tar)
    with tarfile.open(tarball_path, "w:gz") as tar:
        for file_path in subset_files:
            tar.add(file_path, arcname=os.path.basename(file_path))

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

def plot_all_datasets_dirs(result_dirs: list[str],
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
    
    plot_data(plots_figs, xlabel=xlabel, fig_size=fig_size, title=title, fname_pre=fname_pre)
            
def plot_all_datasets_tars(result_tars: list[str],
                           labels: list[str],
                           xdata_str: str = "num_qubits",
                           xlabel: str = "Number of qubits",
                           fig_size: tuple[int, int] = (16,9),
                           title: str = "",
                           fname_pre: str = ""):
    """Plot all datasets in given tarballs. The datasets are plotted based on the x-axis data and the labels.

    Args:
        result_tars: List of tarballs (result.tar.gz) containing the data files.
        labels: List of labels for the datasets.
        xdata_str: Identifier string what data property should be plotted on the x-axis. Defaults to "num_qubits".
        xlabel: Label of the x-axis. Defaults to "Number of qubits".
        fig_size: Size of each figure window. Defaults to (16,9).
        title: Title stirng for each figure window. Defaults to "".
        fname_pre: Preamble for the files to save the plots to. The Preamble should also contain the path to the desired directory. Defaults to "", i.e., no saving.

    Raises:
        KeyError: If the x-axis data property is not found in the data files.
    """
    plots_figs = {}
    for dataset_tar, label in zip(result_tars, labels):
        if os.path.isfile(dataset_tar) and tarfile.is_tarfile(dataset_tar):  # Ensure it's a valid tarball
            dataset = {}
            xdata = []
            with tarfile.open(dataset_tar, "r:*") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith((".yaml", ".yml")):
                        with tar.extractfile(member) as f:
                            if f:
                                data: dict = yaml.safe_load(f)
                                if xdata_str not in data:
                                    raise KeyError("Key {} used for x-axis not found in all data files of current tarball {}!".format(xdata_str, dataset_tar))
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
    
    plot_data(plots_figs, xlabel=xlabel, fig_size=fig_size, title=title, fname_pre=fname_pre)
            
def plot_data(data: dict[str, dict[str, tuple[list, list]]],
              xlabel: str="Number of qubits",
              fig_size: tuple[int, int] = (16,9),
              title: str="",
              fname_pre: str = ""):
    """Plot the given data in different figure windows. The data is plotted based on the y-axis data (different figures) and the labels (different curves in the figures).

    Args:
        data: structured data dictionary. Outer dictionary keys are the different y-axis labels and inner dictionary keys are the different curve labels in the figures. The tuples carry the actual data lists(xdata, ydata).
        xlabel: Label of the x-axis. Defaults to "Number of qubits".
        fig_size: Size of each figure window. Defaults to (16,9).
        title: Title stirng for each figure window. Defaults to "".
        fname_pre: Preamble for the files to save the plots to. The Preamble should also contain the path to the desired directory. Defaults to "", i.e., no saving.
    """
    colors = ["blue", "red", "green", "orange", "magenta", "cyan"]
    for k,v in data.items():
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
    unique_x.sort()
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
    # circuit_method = "grow"
    # substate_size_fac = None
    # substate_size = None
    # star_states=False
    # binary_merge=False
    
    # circuit_method = "merge_parallel"
    # substate_size_fac = 1.3
    # substate_size = None
    # star_states = False
    # binary_merge = False

    # circuit_method = "merge_sequential"
    # substate_size_fac = None
    # substate_size = None
    # star_states = False
    # binary_merge = True

    # #graph_dir = "/Users/as56ohop/Documents/NAS_sync/PhD/code/ghz_state_generation_in_com_networks/ghz_generation_heuristic_alg/Saved_small_random_graphs/sample_circuits_test"
    # #graph_dir = os.path.join(os.getcwd(), "Saved_small_random_graphs/sample_circuits_test")
    #graph_dir = os.path.join(os.getcwd(), "graph_samples/random_graph_endros_renyi_1/")
    #graph_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_ibm_brisbane_1/")
    #graph_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_rect_grid_1/")
# 
    #out_dir = None
    # out_dir = os.path.join(os.getcwd(), "simulation_results/sample_circuits/random_graphs_endros_renyi_p0.0/")
    # 
    #run_circuit_sampling(graph_dir, circuit_method, substate_size_fac=substate_size_fac, substate_size=substate_size, star_states=star_states, binary_merge=binary_merge, out_dir=out_dir, overwrite_results=False)

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
    #result_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_rect_grid_1/circuit_sampling_results/circuit_method_merge_sequential")
    #result_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_ibm_brisbane_1/circuit_sampling_results/circuit_method_merge_parallel_non-bin_substate_none_ghz")
    #result_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_ibm_brisbane_1/circuit_sampling_results/circuit_method_merge_parallel_non-bin_substate_4_ghz")
    #result_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_ibm_brisbane_1/circuit_sampling_results/circuit_method_merge_parallel_non-bin_substate_1.0_ghz")

# 
    # result_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_ibm_brisbane_1/circuit_sampling_results/circuit_method_merge_parallel_non-bin_substate_4_ghz")
    # #output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_4_ghz")
    # #create_data_subset(result_dir, output_dir)
    # output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/")
    # fname_tar = "circuit_method_merge_parallel_non-bin_substate_4_ghz.tar.gz"
    # create_data_subset_tar(result_dir, output_dir, fname_tar=fname_tar)
# # 
    # result_dir = os.path.join(os.getcwd(), "graph_samples/layout_graph_rect_grid_1/circuit_sampling_results/circuit_method_merge_parallel_non-bin_substate_none_ghz")
    # #output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_method_merge_parallel_non-bin_substate_none_ghz")
    # #create_data_subset(result_dir, output_dir)
    # output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/")
    # fname_tar = "circuit_method_merge_parallel_non-bin_substate_none_ghz.tar.gz"
    # create_data_subset_tar(result_dir, output_dir, fname_tar=fname_tar)

    # result_dir = os.path.join(os.getcwd(), "graph_samples/random_graph_endros_renyi_1/circuit_sampling_results/circuit_method_merge_parallel_non-bin_substate_none_ghz")
    # output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_p1.0/")
    # fname_tar = "circuit_method_merge_parallel_non-bin_substate_none_ghz.tar.gz"
    # create_data_subset_tar(result_dir, output_dir, fname_tar=fname_tar, p=1.0)


    #output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n400/circuit_method_grow")
    #output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_none_ghz")
    #output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_4_ghz")
    #output_dir = os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_1.0_ghz")

    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n57/circuit_method_grow"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n153/circuit_method_grow"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n272/circuit_method_grow"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_n400/circuit_method_grow")]
    #plot_all_datasets(output_dir, ["n=57", "n=153", "n=272", "n=400"], xdata_str="p_value", xlabel="p value", title="Random graph sampling evaluation", fname_pre="")
    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_grow"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_sequential")]
    # # plot_all_datasets(output_dir, ["state growing", "merge parallel binary, hd", "merge sequential, hd"], xdata_str="num_qubits", xlabel="number of qubits", title="Random subgraph sampling from ibm brisbane layout", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_sampling_eval_layout_graph_ibm_brisbane_1_old_merge_version"))
    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_grow"),
    #                os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel")]
    # plot_all_datasets(output_dir, ["state growing", "merge parallel binary, hd"], xdata_str="num_qubits", xlabel="number of qubits", title="Random subgraph sampling from ibm brisbane layout", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_sampling_eval_layout_graph_ibm_brisbane_1_old_merge_version_no_sequential"))
    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_none_ghz"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_4_ghz")]
    # plot_all_datasets(output_dir, ["merge parallel binary, hd", "merge parallel non-binary, hd", "merge parallel non-binary, size_fac=1.0"], xdata_str="num_qubits", xlabel="number of qubits", title="Random subgraph sampling from ibm brisbane layout", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_sampling_eval_layout_graph_ibm_brisbane_1_non-bin_parallel_merge_version_2"))
    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_none_ghz"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_4_ghz"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_method_merge_parallel_non-bin_substate_1.0_ghz")]
    # plot_all_datasets(output_dir, ["merge parallel non-binary, hd", "merge parallel non-binary, size=4", "merge parallel non-binary, size_fac=1.0"], xdata_str="num_qubits", xlabel="number of qubits", title="Random subgraph sampling from ibm brisbane layout", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_1/circuit_sampling_eval_layout_graph_ibm_brisbane_1_non-bin_parallel_merge_version_3"))
    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_method_grow"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_method_merge_parallel"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_method_merge_sequential")]
    #plot_all_datasets(output_dir, ["state growing", "merge parallel binary, hd", "merge sequential, hd"], xdata_str="num_qubits", xlabel="number of qubits", title="Random subgraph sampling from rectangular grid layout", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_sampling_eval_layout_graph_rect_grid_1_old_merge_version"))
    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_method_grow"),
    #               os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_method_merge_parallel")]
    # plot_all_datasets(output_dir, ["state growing", "merge parallel binary, hd"], xdata_str="num_qubits", xlabel="number of qubits", title="Random subgraph sampling from rectangular grid layout", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_sampling_eval_layout_graph_rect_grid_1_old_merge_version_no_sequential"))
    
    # output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/circuit_method_grow.tar.gz")]
    # 
    # plot_all_datasets_tars(output_dir, ["state growing"], xdata_str="num_qubits", xlabel="number of qubits", title="Random subgraph sampling from rectangular grid layout", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_layout_graph_rect_grid_1/test_tarball"))

    output_dir = [os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_p1.0/circuit_method_grow.tar.gz"),
                  os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_p1.0/circuit_method_merge_parallel_non-bin_substate_0.7_ghz.tar.gz"),
                  os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_p1.0/circuit_method_merge_parallel_non-bin_substate_1.0_ghz.tar.gz"),
                  os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_p1.0/circuit_method_merge_parallel_non-bin_substate_1.3_ghz.tar.gz"),
                  os.path.join(os.getcwd(), "graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_p1.0/circuit_method_merge_parallel_non-bin_substate_none_ghz.tar.gz")]
 
    plot_all_datasets_tars(output_dir, ["state growing", "merge parallel non-binary, size_fac=0.7", "merge parallel non-binary, size_fac=1.0", "merge parallel non-binary, size_fac=1.3", "merge parallel non-binary, size=hd"], xdata_str="num_qubits", xlabel="number of qubits", title="Random Endros-Renyi graph sampling for p=1.0", fname_pre=os.path.join(os.getcwd(),"graph_samples_eval/circuit_sampling_random_graph_endros_renyi_1_p1.0/circuit_sampling_eval_random_graph_endros_renyi_1_p1.0"))


