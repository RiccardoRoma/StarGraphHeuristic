import numpy as np
import pickle
from Qiskit_input_graph import MergePattern, compute_scaling_factor
import modify_graph_objects as mgo
from witness import witness_fancy, witness_plain, fidelity_est_simple, fidelity_full, ghz_hellinger_fidelity
from generate_circuit_samples import get_ibm_layout_graph
from sample_circuits import eval_circuit
import networkx as nx
from networkx import Graph
from qiskit import QuantumCircuit, ClassicalRegister
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
from qiskit import qpy
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime import Batch, Session
from qiskit.quantum_info import hellinger_fidelity

import csv
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import PrimitiveResult
from collections import defaultdict
import glob


if __name__ == "__main__":
    # Setup argument parser to read in required inputs from cmd line
    parser = argparse.ArgumentParser(description='Script to run heuristic circuit for generating GHZ state on IBM hardware')
    parser.add_argument('simulation_id', metavar='id', type=str, help='a unique id for this simulation run')
    parser.add_argument('circ_files_dir', metavar='/path/to/circ_files_dir/', type=str, help='path to directory, containing the qpy files of the circuits of this simulation run.')
    parser.add_argument('-f', '--file', dest='fname', metavar='/path/to/filename.yaml', action='store', type=str, default=None, help='path to yaml file which contains the calibration data of this simulation run.')
    
    args = parser.parse_args()
    
    # Define a id for the simulation run
    sim_id = args.simulation_id
    
    circ_files_dir = args.circ_files_dir
    
    # Define calibration file
    if args.fname is None:
        cal_file = "example_calibration_sample_fidelity_sampler.yaml"
    else:
        cal_file = args.fname
    
    # make directory for this simulation run
    result_dir = os.path.join(circ_files_dir, sim_id)
    if os.path.isdir(result_dir):
        raise FileExistsError("directory "+result_dir+" does already exist!")
    else:
        os.mkdir(result_dir)
    
    #list of all circuit files
    circ_files = list(glob.glob(os.path.join(circ_files_dir, "*.qpy")))
    
    # load circuits from file
    circ_list = utils.load_circuits_from_file(circ_files)
    
    if len(circ_list) != len(circ_files):
        raise ValueError("Detected serialized list of quantum circuits in on of the found qpy files. This is currently not supported.")
    
    # Load calibration dictionary from YAML file
    cal_dict = utils.load_calibration_from_file(cal_file)

    # add z-basis measurements for Hellinger fidelity calculation
    for curr_circ, curr_circ_file in zip(circ_list, circ_files):
        # draw original circuit and save it to file
        curr_file_pream = os.path.splitext(os.path.basename(curr_circ_file))[0]
        curr_circ_file_plt, _ = os.path.splitext(curr_circ_file)
        curr_circ_file_plt = curr_circ_file_plt + ".pdf"
        curr_circ.draw("mpl", filename=curr_circ_file_plt)

        # determine idle qubits in current circuit
        idle_qubits = utils.get_idle_qubits(curr_circ)
        
        # determine non-idle qubits in current circuit
        nonidle_qubit_idcs = list(range(curr_circ.num_qubits))
        for q in idle_qubits:
            q_idx, _ = curr_circ.find_bit(q)
            nonidle_qubit_idcs.remove(q_idx)
        # add classical register for fidelity measurements
        curr_cls_reg = ClassicalRegister(len(nonidle_qubit_idcs), name="output")
        curr_circ.add_register(curr_cls_reg)

        # measure all non-idle qubits
        for creg_idx, q_idx in enumerate(nonidle_qubit_idcs):
            curr_circ.measure(q_idx, curr_cls_reg[creg_idx])

    print("Added fidelity measurements!")
    
    # run circuits 
    smplr_result, transp_circs = utils.run_circuits_sampler(circ_list, cal_dict, result_dir=result_dir, sim_id=sim_id)
    
    if len(circ_list) != len(transp_circs):
        raise ValueError("number of transpied circuits does not match number of input circuits!")
    
    # save results
    for pub_res, transp_circ, init_circ, init_circ_file in zip(smplr_result, transp_circs, circ_list, circ_files):
        curr_file_pream = os.path.splitext(os.path.basename(init_circ_file))[0]

        # save transpiled circuit
        transp_circ_file = os.path.join(result_dir, curr_file_pream + "_transp.qpy")
        with open(transp_circ_file, "wb") as f:
            qpy.dump(transp_circ, f)
        # draw transpiled circuit and save figure
        transp_circ_file, _ = os.path.splitext(transp_circ_file)
        transp_circ_file = transp_circ_file +".pdf"
        transp_circ.draw("mpl", filename=transp_circ_file)

        # calculate the Hellinger fidelity from the bitarray result of the measurements in the classical output register
        curr_fidelity = ghz_hellinger_fidelity(pub_res.data["output"])
        curr_fidelity_std = 0.0
        
        fname_result = os.path.join(result_dir, curr_file_pream + "_result.yaml")
        if os.path.exists(fname_result):
            raise ValueError("Result file does already exists!")
        else:
            idle_qubits = utils.get_idle_qubits(init_circ)
            
            nonidle_qubit_idcs = list(range(init_circ.num_qubits))
            for q in idle_qubits:
                q_idx, _ = init_circ.find_bit(q)
                nonidle_qubit_idcs.remove(q_idx)
            # evaluate transpiled circuit
            result_data = eval_circuit(transp_circ)
            # add number of qubits and fidelity to result data
            result_data["num_qubits"] = len(nonidle_qubit_idcs)
            result_data["fidelity"] = [curr_fidelity, curr_fidelity_std]
            # save results to yaml file
            with open(fname_result, "w") as f:
                yaml.dump(result_data, f)
    
    