import numpy as np
import pickle
from Qiskit_input_graph import MergePattern
import modify_graph_objects as mgo
from witness import witness_fancy, witness_plain, fidelity_est_simple, fidelity_full
import networkx as nx
from networkx import Graph
from qiskit import QuantumCircuit
from typing import Sequence, Tuple
import os
import create_ghz_state_circuit as cgsc
import simulation_utils as utils
import copy
from dotenv import load_dotenv
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime.options.utils import Unset
from qiskit.quantum_info import SparsePauliOp
from IBM_hardware_data import generate_layout_graph
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler.passes import RemoveBarriers

import csv
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# Setup argument parser to read in required inputs from cmd line
parser = argparse.ArgumentParser(description='Script to run heuristic circuit for generating GHZ state on IBM hardware')
parser.add_argument('simulation_id', metavar='id', type=str, help='a unique id for this simulation run')
parser.add_argument('-f', '--file', dest='fname', metavar='/path/to/filename.yaml', action='store', type=str, default=None, help='path to yaml file which contains the calibration data of this simulation run.')
parser.add_argument('--show_graph', action='store_true', help="Set this flag to true to show the layout graph plot before running simulation.")


args = parser.parse_args()

# Define a id for the simulation run
sim_id = args.simulation_id

# Define calibration file
if args.fname is None:
    cal_file = "example_calibration.yaml"
else:
    cal_file = args.fname

show_graph = args.show_graph

# get estimator calibration
est_cal = utils.get_EstimatorCalibration_from_yaml(cal_file)
# get passmanager calibration
pm_cal = utils.get_PresetPassManagerCalibration_from_yaml(cal_file)


# read calibration dictionary
if not os.path.isfile(cal_file):
    raise ValueError("file {} does not exist!".format(cal_file))

cal_dict = None
raw_data = None
with open(cal_file, "r") as f:
    raw_data = f.read()

cal_dict = yaml.load(raw_data, Loader=yaml.Loader)
if cal_dict is None:
    raise ValueError("Something went wrong while reading in yaml text file! resulting dictionary is empty!")
    

# Define the directory to save the results
## To-Do: Result directory should also be a cmd line input
result_dir = cal_dict.get("result_dir", None)
##
if result_dir is None:
    print("Could not retrieve a result directory from calibration file. Set it to the current working directory!")
    result_dir = os.getcwd()

# make directory for this simulation run
result_dir = os.path.join(result_dir, sim_id)
if os.path.isdir(result_dir):
    raise FileExistsError("directory "+result_dir+" does already exist!")
else:
    os.mkdir(result_dir)


use_premium_access = cal_dict["use_premium_access"]
generate_star_states = cal_dict["generate_star_states"] # flag for ghz generation (False) or star graphs (True)
parallel_merge = cal_dict["parallel_merge"] # flag for using parallel merging
binary_parallel = cal_dict["binary_parallel"] # flag for using binary parallel merging or full parallel merging
substate_size = cal_dict["substate_size"] # target size of each substate
fidelity_witness = cal_dict["fidelity_witness"]
backend_str = cal_dict["backend_str"]
noise_model_id = cal_dict["noise_model_id"]
fname_noise_model = cal_dict["fname_noise_model"]
fname_noise_model = os.path.join(result_dir, fname_noise_model)
noise_model_str = cal_dict["noise_model_str"]
coupling_map_id = cal_dict["coupling_map_id"]
fname_coupling_map = cal_dict["fname_coupling_map"]
fname_coupling_map = os.path.join(result_dir, fname_coupling_map)
coupling_map_str = cal_dict["coupling_map_str"]
native_basis_gates_str = cal_dict["native_basis_gates_str"]
# setup backend options
backend_opt = {"backend_str": backend_str, 
               "noise_model_id": noise_model_id, 
               "fname_noise_model": fname_noise_model, 
               "noise_model_str": noise_model_str,
               "coupling_map_id": coupling_map_id,
               "fname_coupling_map": fname_coupling_map,
               "coupling_map_str": coupling_map_str,
               "native_basis_gates_str": native_basis_gates_str,
               "run_locally": cal_dict["run_locally"]}

# load IBM Quantum credentials
service = utils.load_ibm_credentials(premium_access=use_premium_access)

# load backend from passmanager calibration
backend = utils.get_backend(service, backend_opt)

# get noise model from backend
noise_model = backend.options.get("noise_model", None)
if noise_model is None:
    noise_model = NoiseModel.from_backend(backend)

# save noise model if not already done
if noise_model:
    if not noise_model.is_ideal():
        if not os.path.isfile(fname_noise_model):
            with open(fname_noise_model, "wb") as f:
                pickle.dump(noise_model,f)

# save coupling map if not already done
coupling_map = backend.coupling_map
if coupling_map:
    if not os.path.isfile(fname_coupling_map):
        with open(fname_coupling_map, "wb") as f:
            pickle.dump(coupling_map,f)


subgraph_size = cal_dict.get("subgraph_size", None)
print("Create layout graph from backend.")
graph = generate_layout_graph(backend, k=subgraph_size)
# double check if graph is connected!
if not nx.is_connected(graph):
    raise ValueError("Detected isolated nodes in layout graph!")

# add graph nodes as initial layout to passmanager calibration
if pm_cal.initial_layout is not None:
    print("Warning: Initial layout of passmanager calibration is not None but will be overwritten due to algorithm.")
init_layout = list(range(backend.num_qubits)) # list of all qubits, i.e., a trivial layout i -> i
pm_cal.initial_layout = init_layout

# create pass manager from calibration
pass_manager = utils.get_passmanager(backend, pm_cal)

# create the circuit to generate GHZ state
#curr_circ, curr_init_graph, curr_star_graph = cgsc.create_ghz_state_circuit_graph(graph, backend.num_qubits, star=generate_star_states)
if parallel_merge:
    merge_pattern = MergePattern.from_graph_parallel(graph, binary_merge=binary_parallel, substate_size=substate_size)
else:
    if not binary_parallel:
        print("Warning: Used sequential merging but set flag for non-binary parallel merging! This flag is not used for sequential merging")
    merge_pattern = MergePattern.from_graph_sequential(graph, substate_size=substate_size)

curr_circ, curr_init_graph, curr_star_graph = cgsc.create_ghz_state_circuit_graph(merge_pattern, backend.num_qubits, star=generate_star_states)
#curr_circ, curr_init_graph = cgsc.create_ghz_state_circuit_debug(graph, backend.num_qubits)

# draw initial layout graph and save the plot
fname_graph = f"sim_{sim_id}_layout_graph_backend_{backend_str}.pdf"
fname_graph = os.path.join(result_dir, fname_graph)
mgo.draw_graph(merge_pattern.initial_graph, show_weights=True, title="Layout graph from "+backend_str+ " backend", fname=fname_graph, show=show_graph)

# draw merge pattern graph and save the plot
fname_graph = f"sim_{sim_id}_merge_pattern.pdf"
fname_graph = os.path.join(result_dir, fname_graph)
merge_pattern.draw_pattern_graph(show_weights=True, title="Merge pattern graph", fname=fname_graph, show=show_graph)

# draw all subgraphs and save the plots
dir_subgraphs = os.path.join(result_dir, "subgraphs")
os.mkdir(dir_subgraphs)

for idx in range(len(merge_pattern.subgraphs)):
    fname_subgraph = f"sim_{sim_id}_subgraph_{idx}.pdf"
    fname_subgraph = os.path.join(dir_subgraphs, fname_subgraph)
    subgraph = merge_pattern.subgraphs[idx]
    mgo.draw_graph(subgraph, show_weights=False, title=f"Subgraph {idx}", fname=fname_subgraph, show=False)

# Get fidelity observable
#observalbe = SparsePauliOp(["X"*backend.num_qubits], coeffs=np.asarray([1.0])) # This is just a dummy observable used for debugging
if fidelity_witness == "simple":
    observalbe = fidelity_est_simple(curr_circ.num_qubits, list(graph))
elif fidelity_witness == "full":
    observalbe = fidelity_full(curr_circ.num_qubits, list(graph))
else:
    raise ValueError("Keyword for fidelity witness {} is unkown".format(fidelity_witness))

# transpile circuit with passmanager
transp_circ = utils.transpile_circuits(pass_manager, [curr_circ], transpilation_trials=pm_cal.transpilation_trials, remove_barriers=True)[0]

isa_observable = observalbe.apply_layout(transp_circ.layout)

# create estimator from calibration
estimator = utils.get_estimator(est_cal, mode=backend)
job = estimator.run([(transp_circ, isa_observable)])
est_result = job.result()
# extract value from 0-d numpy array
fidelity = float(est_result[0].data.evs)
fidelity_std = est_result[0].data.stds

print("created {} qubit GHZ state with fidelity {} +/- {}!".format(len(graph), fidelity, fidelity_std))

# save the result and all calibration data, graph, transpiled circuit, etc. in the result dir
# save to csv file
header_est, data_est = est_cal.get_filevector()
header_pm, data_pm = pm_cal.get_filevector()
csv_header = ["fidelity", "fidelity_std"] 
for k in backend_opt.keys():
    csv_header = csv_header + [k]
csv_header = csv_header + header_est + header_pm
csv_data = [fidelity]
if fidelity_std is None:
    csv_data = csv_data + ["None"]
else:
    csv_data = csv_data + [fidelity_std]
for v in backend_opt.values():
    csv_data = csv_data + [v]
csv_data = csv_data + data_est + data_pm
fname_csv = f"sim_{sim_id}_results.csv"
fname_csv = os.path.join(result_dir, fname_csv)

with open(fname_csv, "w", newline="") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ')
    spamwriter.writerow(csv_header)
    spamwriter.writerow(csv_data)
##

# save calibration as yaml file
fname_cal = f"sim_{sim_id}_calibration.yaml"
fname_cal = os.path.join(result_dir, fname_cal)
# check if file already exists
if os.path.isfile(fname_cal):
    raise ValueError("file {} does already exist!".format(fname_cal))
# prepare calibration classes for yaml
est_cal_dict = est_cal.to_dict()
pm_cal_dict = pm_cal.to_dict()
cal_dict["estimator_calibration"] = est_cal_dict
cal_dict["passmanager_calibration"] = pm_cal_dict
# dump calibration dictionary into yaml file
with open(fname_cal, "w") as f:
    yaml.dump(cal_dict, f)


# pickle merge pattern
fname_pattern = f"sim_{sim_id}_merge_pattern.pickle"
fname_pattern = os.path.join(result_dir, fname_pattern)
with open(fname_pattern, "wb") as f:
    pickle.dump(merge_pattern,f)

# pickle initial circuit and save plot
fname_circ = f"sim_{sim_id}_circuit_backend_{backend_str}.pickle"
fname_circ = os.path.join(result_dir, fname_circ)
with open(fname_circ, "wb") as f:
    pickle.dump(curr_circ, f)
fname_circ = f"sim_{sim_id}_circuit_backend_{backend_str}.pdf"
fname_circ = os.path.join(result_dir, fname_circ)
curr_circ.draw("mpl", filename=fname_circ)


# pickle transpiled circuit and save plot
fname_circ_transp = f"sim_{sim_id}_circuit_transpiled_backend_{backend_str}.pickle"
fname_circ_transp = os.path.join(result_dir, fname_circ_transp)
with open(fname_circ_transp, "wb") as f:
    pickle.dump(transp_circ, f)
fname_circ_transp = f"sim_{sim_id}_circuit_transpiled_backend_{backend_str}.pdf"
fname_circ_transp = os.path.join(result_dir, fname_circ_transp)
transp_circ.draw("mpl", filename=fname_circ_transp)

# pickle estimator result
fname_est_result = f"sim_{sim_id}_estimator_result.pickle"
fname_est_result = os.path.join(result_dir, fname_est_result)
with open(fname_est_result, "wb") as f:
    pickle.dump(est_result, f)


