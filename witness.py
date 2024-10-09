import math
# from Qiskit_input_graph import draw_graph, calculate_msq
# import networkx as nx
# from networkx import Graph
# import create_ghz_state_circuit as cgsc
from qiskit.quantum_info import SparsePauliOp





def witness_plain(n_qbs:int):

    list_obs = [('I' * n_qbs,n_qbs-1), (reversed('X' * ('I' * (n_qbs-1))),-1)]
    
    for i in range(1, n_qbs-1):
        s = ['I'] * n_qbs
        s[i] = 'Z'
        s[i+1] = 'Z'
        list_obs.append((''.join(s), -1))

    op = SparsePauliOp.from_list(list_obs)
    return op

def witness_fancy(n_qbs:int):

    list_obs = [('I' * (n_qbs), 1), (reversed('X' * ('I' * (n_qbs-1))),-1)]
    
    for i in range(1, n_qbs-1):
        s = ['I'] * n_qbs
        s[i] = 'Z'
        s[i+1] = 'Z'
        list_obs.append((''.join(s), -1))

    op = 3 * SparsePauliOp('I' * n_qbs) -  (SparsePauliOp(list_obs[1]) + SparsePauliOp(list_obs[0]))
    op += - 2 * math.prod(SparsePauliOp(list_obs[1]) + SparsePauliOp(list_obs[i])/2 for i in range(2, n_qbs))
    return op






