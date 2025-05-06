import argparse
import os
from witness import fidelity_est_simple, fidelity_full
from sample_circuits import eval_circuit
import simulation_utils as utils
from qiskit import qpy
import pickle
import yaml

if __name__ == "__main__":
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
        cal_file = "example_calibration_sample_fidelity.yaml"
    else:
        cal_file = args.fname
    
    # make directory for this simulation run if it not already exists
    result_dir = os.path.dirname(circ_file)
    result_dir = os.path.join(result_dir, sim_id)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    
    
    # load circuits from file
    circ_list = utils.load_circuits_from_file([circ_file])
    
    if len(circ_list) != 1:
        raise ValueError("Detected serialized list of quantum circuits in the qpy file. This is currently not supported.")
    
    # Load calibration dictionary from YAML file
    cal_dict = utils.load_calibration_from_file(cal_file)

    # construct fidelity observables for each circuit
    fidelity_witness = cal_dict["fidelity_witness"]
    observables = []
    for curr_circ in circ_list:
        # Get fidelity observable
        #observalbe = SparsePauliOp(["X"*backend.num_qubits], coeffs=np.asarray([1.0])) # This is just a dummy observable used for debugging
        idle_qubits = utils.get_idle_qubits(curr_circ)
        
        nonidle_qubit_idcs = list(range(curr_circ.num_qubits))
        for q in idle_qubits:
            q_idx, _ = curr_circ.find_bit(q)
            nonidle_qubit_idcs.remove(q_idx)
        
        if fidelity_witness == "simple":
           observable = fidelity_est_simple(curr_circ.num_qubits, nonidle_qubit_idcs)
           #observalbe = fidelity_est_simple(backend.num_qubits, nonidle_qubit_idcs)
        elif fidelity_witness == "full":
            observable = fidelity_full(curr_circ.num_qubits, nonidle_qubit_idcs)
        else:
            raise ValueError("Keyword for fidelity witness {} is unkown".format(fidelity_witness))
        observables.append(observable)
    print("Constructed observables!")
    

    # run circuits 
    est_result, transp_circs_obs_pairs = utils.run_circuits(circ_list, observables, cal_dict, result_dir=result_dir, sim_id=sim_id)
    
    if len(circ_list) != len(transp_circs_obs_pairs):
        raise ValueError("number of transpied circuits and observables does not match number of input circuits!")
    
    # save results
    for pub_res, transp_circ_obs_pair, init_circ in zip(est_result, transp_circs_obs_pairs, circ_list):
        curr_file_pream = os.path.splitext(os.path.basename(circ_file))[0]
    
        # save drawing of original circuit
        init_circ_file_plt, _ = os.path.splitext(circ_file)
        init_circ_file_plt = init_circ_file_plt + ".pdf"
        init_circ.draw("mpl", filename=init_circ_file_plt)
    
        # save transpiled circuit
        transp_circ_file = os.path.join(result_dir, curr_file_pream + "_transp.qpy")
        if os.path.exists(transp_circ_file):
            raise FileExistsError(f"File for pickled transpiled circuit {transp_circ_file} does already exist!")
        with open(transp_circ_file, "wb") as f:
            qpy.dump(transp_circ_obs_pair[0], f)

        # save drawing of the transpiled circuit
        transp_circ_file, _ = os.path.splitext(transp_circ_file)
        transp_circ_file = transp_circ_file +".pdf"
        transp_circ_obs_pair[0].draw("mpl", filename=transp_circ_file)

        # save observable
        transp_obs_file = os.path.join(result_dir, curr_file_pream + "_observable.pickle")
        with open(transp_obs_file, "wb") as f:
            pickle.dump(transp_circ_obs_pair[1], f)
    
        # extract the fidelities
        # extract value from 0-d numpy array
        curr_fidelity = float(pub_res.data.evs)
        curr_fidelity_std = pub_res.data.stds
        if curr_fidelity_std is None:
            curr_fidelity_std = 0.0
        else:
            curr_fidelity_std = float(curr_fidelity_std)
        
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
            result_data = eval_circuit(transp_circ_obs_pair[0])
            # add number of qubits and fidelity to result data
            result_data["num_qubits"] = len(nonidle_qubit_idcs)
            result_data["fidelity"] = [curr_fidelity, curr_fidelity_std]
            # save results to yaml file
            with open(fname_result, "w") as f:
                yaml.dump(result_data, f)
    
    