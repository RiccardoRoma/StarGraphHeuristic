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
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
import copy
from dotenv import load_dotenv
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options

# Define what estimator and passmanager calibration should be used
cal_file = "example_calibration.yaml"

# get estimator calibration
est_cal = cal.get_EstimatorCalibration_from_yaml(cal_file)
# get passmanager calibration
pm_cal = cal.get_PresetPassManagerCalibration_from_yaml(cal_file)

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

# create the circuit to generate GHZ state
curr_circ, curr_init_graph, curr_star_graph = cgsc.create_ghz_state_circuit(curr_fname)

## To-Do integrate funciton to get the observable here, once it's finished!
# Get fidelity observable
observalbe = None
##

# transpile circuit with passmanager
transp_circ = pass_manager.run(curr_circ)

## To-Do: This still needs to be implemented
with Session(service, backend=pm_cal.backend_str) as session:
    # create estimator from calibration
    estimator = cal.get_estimator(est_cal, session=session)
    # run circuit and observable on backend
    job = estimator.run(transp_circ, observalbe)
    res = job.result()

# save the result and all calibration data, graph, transpiled circuit, etc. in the result dir
##

