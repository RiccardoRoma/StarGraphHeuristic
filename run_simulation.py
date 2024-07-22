import numpy as np
import pickle
from Qiskit_input_graph import draw_graph, calculate_msq
import modify_graph_objects as mgo
import networkx as nx
from networkx import Graph
from qiskit import QuantumCircuit
from typing import Sequence, Tuple
import os
import create_ghz_state_circuit as cgsc
import calibration_classes as cal
import copy
from dotenv import load_dotenv
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit_ibm_runtime import Estimator
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

args = parser.parse_args()

# Define a id for the simulation run
sim_id = args.simulation_id

# Define calibration file
if args.fname is None:
    cal_file = "example_calibration.yaml"
else:
    cal_file = args.fname

# get estimator calibration
est_cal = cal.get_EstimatorCalibration_from_yaml(cal_file)
# get passmanager calibration
pm_cal = cal.get_PresetPassManagerCalibration_from_yaml(cal_file)

# load backend from passmanager calibration
backend = cal.get_backend(pm_cal.backend_str)


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

graph_dir = cal_dict.get("graph_dir", None)
graph_fname = cal_dict.get("graph_fname", None)
if graph_fname is None and graph_dir is None:
    print("No graph filename and directory were given. Use backend string from passmanager calibration to create graph.")
    graph = generate_layout_graph(backend)
else:
    if graph_fname is None:
        raise ValueError("Graph directory was given but could not retrieve graph filename! To import from backend directly, leave both empty!")
    if graph_dir is None:
        raise ValueError("Graph filename was given but could not retrieve graph directory! To import from backend directly, leave both empty!")
    graph_file = os.path.join(graph_dir, graph_fname)
    # load initial graph from pickle
    graph = None
    with open(graph_file, "rb") as f:
        graph = pickle.load(f)

    if not isinstance(graph, Graph):
        raise ValueError("Loaded graph object is not a networkx graph object!")

# make directory for this simulation run
result_dir = os.path.join(result_dir, sim_id)
if os.path.isdir(result_dir):
    raise FileExistsError("directory "+result_dir+" does already exist!")
else:
    os.mkdir(result_dir)

# create pass manager from calibration
pass_manager = cal.get_passmanager(pm_cal)

# load IBM Quantum credentials
load_dotenv()
ibmq_api_token = os.environ["TOKEN"]
ibmq_hub = os.environ["HUB"]
ibmq_group = os.environ["GROUP"]
ibmq_project = "multiflavorschwi"
# ibmq_project = "reservations"
provider = IBMProvider(token=ibmq_api_token, instance=f"{ibmq_hub}/{ibmq_group}/{ibmq_project}")
service = QiskitRuntimeService(channel="ibm_quantum", token=ibmq_api_token, instance=f"{ibmq_hub}/{ibmq_group}/{ibmq_project}")


# create the circuit to generate GHZ state
curr_circ, curr_init_graph, curr_star_graph = cgsc.create_ghz_state_circuit_graph(graph)

# draw graph and save the plot
fname_graph = f"sim_{sim_id}_layout_graph_backend_{pm_cal.backend_str}.pdf"
fname_graph = os.path.join(result_dir, fname_graph)
mgo.draw_graph(curr_init_graph, title="Layout graph for "+pm_cal.backend_str+ " backend", fname=fname_graph)

## To-Do integrate funciton to get the observable here, once it's finished!
# Get fidelity observable
observalbe = SparsePauliOp(["X"*backend.num_qubits], coeffs=np.asarray([1.0])) # This is just a dummy observable used for debugging
##

# remove all barriers
curr_circ = RemoveBarriers()(curr_circ)
## To-Do: add list of graph nodes as initial layout (want to label logical -> physical qubits as i -> i)!
# transpile circuit with passmanager
transp_circ = pass_manager.run(curr_circ)
isa_observable = observalbe.apply_layout(transp_circ.layout)

with Session(service, backend=backend) as session:
    # create estimator from calibration
    estimator = cal.get_estimator(est_cal, session=session)

    # For fake backends to have a true noisy simulation
    if "fake" in pm_cal.backend_str:
        # set coupling map for estimator to backend.coupling_map
        if estimator.options.simulator["coupling_map"] == Unset:
            estimator.set_options(coupling_map= [list(t) for t in list(backend.coupling_map.get_edges())])
            # update calibration class
            est_cal.estimator_options["simulator_options"]["coupling_map"] =  [list(t) for t in list(backend.coupling_map.get_edges())]
            est_cal.coupling_map_str = pm_cal.backend_str + "_" + datetime.today().strftime('%Y-%m-%d')
    
        # set noise model of estimator to backend noise model
        if estimator.options.simulator["noise_model"] == Unset:
            estimator.set_options(noise_model = NoiseModel.from_backend(backend))
            # update calibration class
            est_cal.estimator_options["simulator_options"]["noise_model"] = NoiseModel.from_backend(backend)
            est_cal.noise_model_str = pm_cal.backend_str + "_" + datetime.today().strftime('%Y-%m-%d')
            
    
        if estimator.options.simulator["basis_gates"] == Unset:
            estimator.set_options(basis_gates = NoiseModel.from_backend(backend).basis_gates)
            # update calibration class
            est_cal.estimator_options["simulator_options"]["basis_gates"] = NoiseModel.from_backend(backend).basis_gates
            est_cal.basis_gates_str = pm_cal.backend_str + "_" + datetime.today().strftime('%Y-%m-%d')
    
    # run circuit and observable on backend
    job = estimator.run(transp_circ, isa_observable)
    est_result = job.result()

    fidelity = est_result.values[0]

# save the result and all calibration data, graph, transpiled circuit, etc. in the result dir
## To-Do: passmanager calibration data, that is None must be converted to string "None" before writing to csv file
# save to csv file
header_est, data_est = est_cal.get_filevector()
header_pm, data_pm = pm_cal.get_filevector()
csv_header = ["fidelity"] + header_est + header_pm
csv_data = [fidelity] + data_est + data_pm
fname_csv = f"sim_{sim_id}_results.csv"
fname_csv = os.path.join(result_dir, fname_csv)

with open(fname_csv, "w", newline="") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ')
    spamwriter.writerow(csv_header)
    spamwriter.writerow(csv_data)
##

# save calibration as yaml files
## To-Do: Save both calibration data in one file (similar as the input calibration file)
fname_est_cal = f"sim_{sim_id}_estimator_calibration.yaml"
fname_est_cal = os.path.join(result_dir, fname_est_cal)
est_cal.to_yaml(fname_est_cal)

fname_pm_cal = f"sim_{sim_id}_passmanager_calibration.yaml"
fname_pm_cal = os.path.join(result_dir, fname_pm_cal)
pm_cal.to_yaml(fname_pm_cal)
##

# pickle intial layout graph
fname_graph = f"sim_{sim_id}_layout_graph_backend_{pm_cal.backend_str}.pickle"
fname_graph = os.path.join(result_dir, fname_graph)
with open(fname_graph, "wb") as f:
    pickle.dump(curr_init_graph,f)

# pickle initial circuit and save plot
fname_circ = f"sim_{sim_id}_circuit_backend_{pm_cal.backend_str}.pickle"
fname_circ = os.path.join(result_dir, fname_circ)
with open(fname_circ, "wb") as f:
    pickle.dump(curr_circ, f)
fname_circ = f"sim_{sim_id}_circuit_backend_{pm_cal.backend_str}.pdf"
fname_circ = os.path.join(result_dir, fname_circ)
curr_circ.draw("mpl", filename=fname_circ)


# pickle transpiled circuit and save plot
fname_circ_transp = f"sim_{sim_id}_circuit_transpiled_backend_{pm_cal.backend_str}.pickle"
fname_circ_transp = os.path.join(result_dir, fname_circ_transp)
with open(fname_circ_transp, "wb") as f:
    pickle.dump(transp_circ, f)
fname_circ_transp = f"sim_{sim_id}_circuit_transpiled_backend_{pm_cal.backend_str}.pdf"
fname_circ_transp = os.path.join(result_dir, fname_circ_transp)
transp_circ.draw("mpl", filename=fname_circ_transp)

# pickle estimator result
fname_est_result = f"sim_{sim_id}_estimator_result.pickle"
fname_est_result = os.path.join(result_dir, fname_est_result)
with open(fname_est_result, "wb") as f:
    pickle.dump(est_result, f)


