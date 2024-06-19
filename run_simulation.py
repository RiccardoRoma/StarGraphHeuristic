import numpy as np
import pickle
from Qiskit_input_graph import draw_graph, calculate_msq
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

import csv

# Define what estimator and passmanager calibration should be used
est_cal_file = "example_estimator_calibration.yaml"
pm_cal_file = "example_passmanager_calibration.yaml"

# Define the directory to save the results
result_dir = "/path/to/result/dir/"

# Define a id for the simulation run
sim_id = "test_run_1"

# make directory for this simulation run
result_dir = os.path.join(result_dir, sim_id)
if os.path.isdir(result_dir):
    raise FileExistsError("directory "+result_dir+" does already exist!")
else:
    os.mkdir(result_dir)

# get estimator calibration
est_cal = cal.get_EstimatorCalibration_from_yaml(est_cal_file)
# get passmanager calibration
pm_cal = cal.get_PresetPassManagerCalibration_from_yaml(pm_cal_file)

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


## To-Do: this must be replaced by loading the graph topology from a given backend
# Define what graph topologies should be considered
graph_dir = "/Users/as56ohop/Documents/NAS_sync/PhD/code/ghz_state_generation_in_com_networks/ghz_generation_heuristic_alg/Saved_small_random_graphs/"
graph_file="random_graph_n10_p0.1_erdos_renyi_copy_1.pkl"

curr_fname = os.path.join(graph_dir, graph_file)
##

## To-Do: Change function such that it takes graph object as input
# create the circuit to generate GHZ state
curr_circ, curr_init_graph, curr_star_graph = cgsc.create_ghz_state_circuit(curr_fname)
##

## To-Do integrate funciton to get the observable here, once it's finished!
# Get fidelity observable
observalbe = None
##

# transpile circuit with passmanager
transp_circ = pass_manager.run(curr_circ)
isa_observable = observalbe.apply_layout(transp_circ.layout)

with Session(service, backend=pm_cal.backend_str) as session:
    # create estimator from calibration
    estimator = cal.get_estimator(est_cal, session=session)
    # run circuit and observable on backend
    job = estimator.run(transp_circ, isa_observable)
    est_result = job.result()

    fidelity = est_result.values[0]

# save the result and all calibration data, graph, transpiled circuit, etc. in the result dir
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

# save calibration as yaml files
fname_est_cal = f"sim_{sim_id}_estimator_calibration.yaml"
fname_est_cal = os.path.join(result_dir, fname_est_cal)
est_cal.to_yaml(fname_est_cal)

fname_pm_cal = f"sim_{sim_id}_passmanager_calibration.yaml"
fname_pm_cal = os.path.join(result_dir, fname_pm_cal)
pm_cal.to_yaml(fname_pm_cal)

# pickle intial layout graph
fname_graph = f"sim_{sim_id}_layout_graph_backend_{pm_cal.backend_str}.pickle"
fname_graph = os.path.join(result_dir, fname_graph)
with open(fname_graph, "wb") as f:
    pickle.dump(curr_init_graph,f)

# pickle initial circuit
fname_circ = f"sim_{sim_id}_circuit_backend_{pm_cal.backend_str}.pickle"
fname_circ = os.path.join(result_dir, fname_circ)
with open(fname_circ, "wb") as f:
    pickle.dump(curr_circ, f)

# pickle transpiled circuit
fname_circ_transp = f"sim_{sim_id}_circuit_transpiled_backend_{pm_cal.backend_str}.pickle"
fname_circ_transp = os.path.join(result_dir, fname_circ_transp)
with open(fname_circ_transp, "wb") as f:
    pickle.dump(curr_circ, f)

# pickle estimator result
fname_est_result = f"sim_{sim_id}_estimator_result.pickle"
fname_est_result = os.path.join(result_dir, fname_est_result)
with open(fname_est_result, "wb") as f:
    pickle.dump(est_result, f)


