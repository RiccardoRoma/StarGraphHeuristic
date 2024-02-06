from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import copy

# This function should return a lists, with all leaf qubits in it
def get_leaf_qubits_from_edges(edges_in: Sequence[Tuple[int, int]],
                               star_centers: Sequence[int]) -> np.ndarray:
    leaf_list = []
    for edge in edges_in:
        e1 = edge[0]
        e2 = edge[1]

        if e1 not in star_centers:
            if e1 not in leaf_list:
                leaf_list.append(e1)

        if e2 not in star_centers:
            if e2 not in leaf_list:
                leaf_list.append(e2)
    # sort leaf list 
    leaf_list.sort()
    #leaf_list = np.asarray(leaf_list)
    return leaf_list

    

def generate_star_states(edges_in: Sequence[Tuple[int, int]],
                         star_centers: Sequence[int],
                         add_barries: bool = False) -> QuantumCircuit:
    # sort star centers
    star_centers.sort()

    # get total number of involved qubits
    leaf_list = get_leaf_qubits_from_edges(edges_in, star_centers)
    num_leafs = len(leaf_list)    
    
    # generate a quantum circuit in which all star states are created
    circ_star_states = QuantumCircuit()
    # add all star centers first
    for c in star_centers:
        # create a QuantumRegister for each star center
        c_reg = QuantumRegister(1, "q"+str(c))
        circ_star_states.add_register(c_reg)
    
    # now add all leafs
    for l in leaf_list:
        # create a QuantumRegister foreach leaf
        l_reg = QuantumRegister(1, "q"+str(l))
        circ_star_states.add_register(l_reg)

    # generate a map between qubit labeling in graph (leafs and centers) and qubits in circuit
    qubit_map = []
    qubit_map.extend(star_centers)
    qubit_map.extend(leaf_list)

    # assign the edges to the corresponding centers
    used_edges = [] # save all used edges to don't do edges twice
    center_leafs = {} # generate a dictionary which contains all connected leafs for each center
    # iterate now through all centers
    for c in star_centers:
        # generate a list with all relevant leafs for the current center
        c_leafs = []
        # iterate through all edges
        for edge in edges_in:
            # check if current edge was already used for another center
            if edge not in used_edges:
                # if the current star center is in the current edge
                if c == edge[0]:
                    # append leaf to relevant list
                    c_leafs.append(edge[1])
                    # add the edge to the used edges list
                    used_edges.append(edge)
                elif c == edge[1]:
                    # append leaf to relevant list
                    c_leafs.append(edge[0])
                    # add the edge to the used edges list
                    used_edges.append(edge)


        # save current center and its relevant edges
        center_leafs[c] = copy.deepcopy(c_leafs)

    # now implement the star state generation
    # add Hadamard to all qubits
    for i in range(len(qubit_map)):
        circ_star_states.h(i)

    for c in star_centers:
        # iterate through its leafs
        c_leafs = center_leafs[c]
        for leaf in c_leafs:
            # add CZ between leaf and center
            try:
                circ_star_states.cz(qubit_map.index(c), qubit_map.index(leaf))
            except Exception as exp:
                print("CZ gate failed for center {} and leaf {}".format(c, leaf))
                print("center index {}".format(qubit_map.index(c)))
                print("leaf index {}".format(qubit_map.index(leaf)))
        if add_barries:
            circ_star_states.barrier()

    return circ_star_states



    