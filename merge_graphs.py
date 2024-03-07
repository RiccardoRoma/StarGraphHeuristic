from typing import Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit
import copy
from networkx import Graph
import modify_graph_objects as mgo

def merge_graphs(circ: QuantumCircuit, G1: Graph, G2: Graph, cls_bit_cnt: int) -> Tuple[QuantumCircuit, Graph, int]:
    return circ, G1, cls_bit_cnt