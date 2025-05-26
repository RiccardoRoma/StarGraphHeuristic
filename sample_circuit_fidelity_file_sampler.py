import argparse
import os
from witness import ghz_hellinger_fidelity
from sample_circuits import eval_circuit
import simulation_utils as utils
from qiskit import qpy, ClassicalRegister
import pickle
import yaml
import time

if __name__ == "__main__":
    # Get the number of allocated CPUs
    num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 2))
    # Setup argument parser to read in required inputs from cmd line
    parser = argparse.ArgumentParser(description='Script to run heuristic circuit for generating GHZ state on IBM hardware')
    parser.add_argument('simulation_id', metavar='id', type=str, help='a unique id for this simulation run')
    parser.add_argument('circ_file', metavar='/path/to/circ_file.qpy', type=str, help='path to qpy file which contains the circuit of this simulation run.')
    parser.add_argument('-f', '--file', dest='fname', metavar='/path/to/filename.yaml', action='store', type=str, default=None, help='path to yaml file which contains the calibration data of this simulation run.')
    
    args = parser.parse_args()
    
    # Define a id for the simulation run
    sim_id = args.simulation_id
    
    circ_file = args.circ_file
    
    # Define calibration file
    if args.fname is None:
        cal_file = "example_calibration_sample_fidelity_sampler.yaml"
    else:
        cal_file = args.fname
    
    # make directory for this simulation run if it not already exists
    result_dir = os.path.dirname(circ_file)
    result_dir = os.path.join(result_dir, sim_id)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # the current circuit file gets an extra subfolder. This is needed for parallel processing on compute cluster
    result_dir = os.path.join(result_dir, os.path.splitext(os.path.basename(circ_file))[0])
    if os.path.isdir(result_dir):
        raise ValueError("Unique directory for this run does already exist!")
    else:
        os.mkdir(result_dir)
    
    
    # load circuits from file
    circ_list = utils.load_circuits_from_file([circ_file])
    
    if len(circ_list) != 1:
        raise ValueError("Detected serialized list of quantum circuits in the qpy file. This is currently not supported.")
    
    # Load calibration dictionary from YAML file
    cal_dict = utils.load_calibration_from_file(cal_file)

    # add z-basis measurements for Hellinger fidelity calculation
    for curr_circ in circ_list:
        # draw original circuit and save it to file
        curr_circ_file_plt, _ = os.path.splitext(circ_file)
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
    print("Running {} circuits with {} CPU cores".format(len(circ_list), num_cpus))
    start_time = time.time() # save start time
    smplr_result, transp_circs = utils.run_circuits_sampler(circ_list, cal_dict, result_dir=result_dir, sim_id=sim_id, num_parallel_threads=num_cpus)
    end_time = time.time() # save end time
    duration = end_time - start_time # duration of transpilation
    print(f"Total time for parallel processing: {duration:.4f} seconds")
    
    if len(circ_list) != len(transp_circs):
        raise ValueError("number of transpied circuits and observables does not match number of input circuits!")
    
    # save results
    for pub_res, transp_circ, init_circ in zip(smplr_result, transp_circs, circ_list):
        curr_file_pream = os.path.splitext(os.path.basename(circ_file))[0]

        # save transpiled circuit
        transp_circ_file = os.path.join(result_dir, curr_file_pream + "_transp.qpy")
        if os.path.exists(transp_circ_file):
            raise FileExistsError(f"File for pickled transpiled circuit {transp_circ_file} does already exist!")
        with open(transp_circ_file, "wb") as f:
            qpy.dump(transp_circ, f)

        # save drawing of the transpiled circuit
        transp_circ_file, _ = os.path.splitext(transp_circ_file)
        transp_circ_file = transp_circ_file +".pdf"
        transp_circ.draw("mpl", filename=transp_circ_file)

        # calculate the Hellinger fidelity from the bitarray result of the measurements in the classical output register
        curr_fidelity = ghz_hellinger_fidelity(pub_res.data["output"])
        curr_fidelity_std = 0.0
        
        fname_result = os.path.join(result_dir, curr_file_pream + "_result.yaml")
        if os.path.exists(fname_result):
            raise FileExistsError(f"Result file {fname_result} does already exists!")
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
    
    