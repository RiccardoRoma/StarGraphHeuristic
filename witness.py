import math
# from Qiskit_input_graph import draw_graph, calculate_msq
# import networkx as nx
# from networkx import Graph
# import create_ghz_state_circuit as cgsc
from qiskit.quantum_info import SparsePauliOp
import copy





def witness_plain(n_qbs:int, ob_qbs: list[int]):
    """
    Constructs a fidelity witness for GHZ state according to Eq. (7) of PRL 94 6 060501 (2005).
    n_qbs:int   Total number of qubits in the circuit
    ob_qbs:list[int]    list of qubits that should be considered by the fidelity witness.
    """
    # list_obs = [('I' * n_qbs,n_qbs-1), ('X'* n_qbs,-1)]
    list_obs = [('I' * n_qbs,len(ob_qbs)-1)]
    s = ['I'] * n_qbs
    for q in ob_qbs:
        s[q] = 'X'
    list_obs.append((''.join(s), -1))
    
    for i in range(len(ob_qbs)-1):
        s = ['I'] * n_qbs
        q1 = ob_qbs[i]
        q2 = ob_qbs[i+1]
        s[q1] = 'Z'
        s[q2] = 'Z'
        list_obs.append((''.join(s), -1))

    op = SparsePauliOp.from_list(list_obs)
    return op

def witness_fancy(n_qbs:int):
    ## To-Do: Consider list of qubits that should be considered by the fidelity witness.
    list_obs = [('I' * (n_qbs), 1), ('X'* n_qbs,-1)]
    
    for i in range(1, n_qbs-1):
        s = ['I'] * n_qbs
        s[i] = 'Z'
        s[i+1] = 'Z'
        list_obs.append((''.join(s), -1))

    op = 3 * SparsePauliOp('I' * n_qbs) -  (SparsePauliOp(list_obs[1]) + SparsePauliOp(list_obs[0]))
    op += - 2 * math.prod(SparsePauliOp(list_obs[1]) + SparsePauliOp(list_obs[i])/2 for i in range(2, n_qbs))
    return op

def fidelity_est_simple(n_qbs:int, ob_qbs: list[int]) -> SparsePauliOp:
    """
    Constructs a fidelity estimate for GHZ state according to Eq. (5) of PRR 4, 033162 (2022) and Eq. (7) of PRL 94 6 060501 (2005).
    n_qbs:int   Total number of qubits in the circuit
    ob_qbs:list[int]    list of qubits that should be considered by the fidelity estimate.
    """
    # construct odd generators
    list_obs = [('I' * n_qbs, 0.5)]
    s = ['I'] * n_qbs
    for q in ob_qbs:
        s[q] = 'X'
    list_obs.append((''.join(s), 0.5))
    G_o = SparsePauliOp.from_list(list_obs)

    # contruct even generators
    even_list = []
    for i in range(len(ob_qbs)-1):
        list_obs = [('I' * n_qbs, 0.5)]
        s = ['I'] * n_qbs
        q1 = ob_qbs[i]
        q2 = ob_qbs[i+1]
        s[q1] = 'Z'
        s[q2] = 'Z'
        list_obs.append((''.join(s), 0.5))
        curr_G_e = SparsePauliOp.from_list(list_obs)
        even_list.append(curr_G_e)
    # build product of all even generators
    G_e = copy.deepcopy(even_list[0])
    for i in range(1, len(even_list)):
        G_e = G_e.dot(even_list[i])

    Id = SparsePauliOp(["I"*n_qbs], [-1.0])

    fidelity_est = SparsePauliOp.sum([G_o, G_e, Id])

    return fidelity_est

def fidelity_full(n_qbs:int, ob_qbs: list[int]) -> SparsePauliOp:
    """
    Constructs the full fidelity observable for GHZ state according to Eq. (3) of PRR 4, 033162 (2022) and Eq. (5) of PRL 94 6 060501 (2005).
    n_qbs:int   Total number of qubits in the circuit
    ob_qbs:list[int]    list of qubits that should be considered by the fidelity observable.
    """
    # construct odd generators
    list_obs = [('I' * n_qbs, 0.5)]
    s = ['I'] * n_qbs
    for q in ob_qbs:
        s[q] = 'X'
    list_obs.append((''.join(s), 0.5))
    G_o = SparsePauliOp.from_list(list_obs)

    # contruct even generators
    even_list = []
    for i in range(len(ob_qbs)-1):
        list_obs = [('I' * n_qbs, 0.5)]
        s = ['I'] * n_qbs
        q1 = ob_qbs[i]
        q2 = ob_qbs[i+1]
        s[q1] = 'Z'
        s[q2] = 'Z'
        list_obs.append((''.join(s), 0.5))
        curr_G_e = SparsePauliOp.from_list(list_obs)
        even_list.append(curr_G_e)
    # build product of all even generators
    G_e = copy.deepcopy(even_list[0])
    for i in range(1, len(even_list)):
        G_e = G_e.dot(even_list[i])

    fidelity = G_o.dot(G_e)
    return fidelity
