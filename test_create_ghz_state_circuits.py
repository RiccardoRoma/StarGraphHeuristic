import numpy as np
import pickle
from Qiskit_input_graph import draw_graph, calculate_msq
import networkx as nx
from networkx import Graph
from qiskit import QuantumCircuit
from typing import Sequence, Tuple
import os
import create_ghz_state_circuit as cgsc
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
import copy

graph_dir = "/Users/as56ohop/Documents/NAS_sync/PhD/code/ghz_state_generation_in_com_networks/ghz_generation_heuristic_alg/Saved_small_random_graphs/"
graph_files=["random_graph_n10_p0.1_erdos_renyi_copy_1.pkl"]*10

fname_vec = [os.path.join(graph_dir, f) for f in graph_files]

sim_results = []
circuits = []
overlaps = []
for i in range(len(fname_vec)):
    curr_fname = fname_vec[i]

    curr_circ, curr_init_graph, curr_star_graph = cgsc.create_ghz_state_circuit(curr_fname)

    # generate usual ghz state for verification
    ghz_state_np = np.zeros(2**curr_circ.num_qubits, dtype="complex")
    ghz_state_np[0] = 1/np.sqrt(2)
    ghz_state_np[-1] = 1/np.sqrt(2)
    ghz_state = Statevector(ghz_state_np)
 
    # get state from circuit
    state_simulator = StatevectorSimulator()
    job = state_simulator.run(curr_circ)
    curr_result = job.result()
    sim_results.append(copy.deepcopy(curr_result))
    circuits.append(curr_circ.copy())
    curr_state = Statevector(job.result().get_statevector(curr_circ))
 
    # calculate overlap
    overlap_ghz = np.abs(curr_state.inner(ghz_state))
    overlaps.append(copy.deepcopy(overlap_ghz))
    print("overlap with ghz state {}".format(overlap_ghz))
    print("measurement outcome {}".format(curr_result.data()['counts']))
    # draw initial graph
    draw_graph(curr_init_graph, show=False)
 
    # draw circuit
    print(curr_circ.draw(output="mpl"))

    # draw final star state graph
    draw_graph(curr_star_graph)
    print("------------------------------")
