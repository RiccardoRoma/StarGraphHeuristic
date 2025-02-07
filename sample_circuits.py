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

# properties for graph construction
graph_str = "ibm" # ibm, reg_square_latt, random
# create layout graphs separately and load here just the pickle files
# create graphs for IBM 127, reg_square_latt (105) & 400, random 400 for different p
# Would be better to create for every ghz_state_size a seperate graph such that we only need to load the graph 
# and directly generate the circuit without extracting a subgraphw
#ghz_state_size=100



def create_circuit(graph_file: str,
                   method: str="grow") -> tuple[QuantumCircuit, Graph]:
    # load/generate layout graph
    if not os.path.isfile(graph_file):
        raise FileNotFoundError("File {} not found!".format(graph_file))
    
    with open(graph_file, "rb") as f:
        graph: Graph = pickle.load(f)

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
    
    return circ, graph

def eval_circuit(circuit_in: QuantumCircuit) -> dict[str, int]:
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
def run_circuit_sampling(graph_dir: str, circuit_method: str, overwrite_results: bool = False):
    if not os.path.isdir(graph_dir):
        raise ValueError("Directory {} was not found!".format(graph_dir))
    
    # create new directory for the sampling results
    if not os.path.isdir(os.path.join(graph_dir, "circuit_sampling_results")):
        os.mkdir(os.path.join(graph_dir, "circuit_sampling_results"))
    if not os.path.isdir(os.path.join(graph_dir, "circuit_sampling_results", "circuit_method_{}".format(circuit_method))):
        os.mkdir(os.path.join(graph_dir, "circuit_sampling_results", "circuit_method_{}".format(circuit_method)))

    result_dir = os.path.join(graph_dir, "circuit_sampling_results", "circuit_method_{}".format(circuit_method))

    all_foms = {}
    qubit_nums = []

    for file_path in glob.glob(os.path.join(graph_dir, "*.pkl")):
        fname = os.path.basename(file_path) # get current graph file name
        fname, _ = os.path.splitext(fname) # split extension

        # filenames for the results
        fname_circ = fname+"_circ_method_{}.qpy".format(circuit_method)
        fname_foms = fname+"_circ_method_{}_foms.yaml".format(circuit_method)

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

        # save circuits and foms
        with open(os.path.join(result_dir, fname_circ), 'wb') as fd:
            qpy.dump(curr_circ, fd)
        with open(os.path.join(result_dir, fname_foms), "w") as f:
            yaml.dump(curr_foms, f)

    plot_averaged_data_foms(qubit_nums, all_foms, fname_pre=os.path.join(result_dir, "evaluation"))

def plot_averaged_data_foms(num_qubits: list, 
                            all_foms: dict,
                            fig_size: tuple[int, int] = (16,9), 
                            fname_pre: str = "",
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
        plt.figure(figsize=fig_size)
        plt.scatter(x_data, y_data, marker="x", color="blue", label="Raw data")
        plt.errorbar(unique_x, averaged_y, yerr=std_dev_y, fmt='o', color='blue', label='Averaged Data', capsize=5)
        plt.plot(unique_x, averaged_y, linestyle='-', color='blue', alpha=0.6)
        plt.xlabel('number of qubits')
        plt.ylabel(f'Averaged {key}')
        plt.legend()
        if len(fname_pre) > 0:
            fname = fname_pre + "_{}.pdf".format(key)
            plt.savefig(fname, bbox_inches="tight")
        #plt.show()
    
    if show_plots:
        plt.show()


circuit_method = "grow"
#graph_dir = "/Users/as56ohop/Documents/NAS_sync/PhD/code/ghz_state_generation_in_com_networks/ghz_generation_heuristic_alg/Saved_small_random_graphs/sample_circuits_test"
graph_dir = os.path.join(os.getcwd(), "Saved_small_random_graphs/sample_circuits_test")

run_circuit_sampling(graph_dir, circuit_method, overwrite_results=True)