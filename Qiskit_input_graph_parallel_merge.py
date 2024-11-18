#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 18 15:25:38 2024

@author: siddhu

This code is for qiskit

"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from typing import Union
from collections import defaultdict, deque
from networkx.drawing.layout import circular_layout
from itertools import combinations, groupby, chain
import math
import csv
import concurrent.futures 
import time
from collections.abc import Iterable
import modify_graph_objects as mgo
from Qiskit_input_graph import calculate_msq


def generate_random_graph(n, p, use_barabasi):
    if use_barabasi:
        # Generate a random graph using the Barabasi-Albert model
        G = nx.barabasi_albert_graph(n, p, seed=None, initial_graph=None)
    else:
        # Generate a random connected graph using the Erdos-Renyi model
        edges = combinations(range(n), 2)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if p <= 0:
            return G
        if p >= 1:
            return nx.complete_graph(n, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = random.choice(node_edges)
            G.add_edge(*random_edge, weight=1)  # Adding edge weight of 1
            for e in node_edges:
                if random.random() < p:
                    G.add_edge(*e, weight=1)  # Adding edge weight of 1
        plt.figure()
        nx.draw(G, node_color='lightgreen', 
                with_labels=True, 
                node_size=500)
    return G




def generate_graph(nodes_list, edges_list, use_barabasi):
    
        # Create a graph G
        G = nx.Graph()
        
        # Add nodes to the graph
        G.add_nodes_from(nodes_list)
        
        # Add edges to the graph
        G.add_edges_from(edges_list)
        
        # plt.figure()
        # nx.draw(G, node_color='lightgreen', 
        #         with_labels=True, 
        #         node_size=500)
        return G 

def draw_graph(graph, **kwargs):
    """
    Draw the given graph using NetworkX and Matplotlib.
    """
    # Default values
    if kwargs.get("node_color", None) is None:
        kwargs["node_color"] = "yellow"
    if kwargs.get("layout", None) is None:
        kwargs["layout"] = "circular"
    if kwargs.get("show", None) is None:
        kwargs["show"] = True
    if kwargs.get("fig_size", None) is None:
        kwargs["fig_size"] = (8, 8)
    if kwargs.get("title", None) is None:
        kwargs["title"] = "Graph Visualization"
    mgo.draw_graph(graph, **kwargs)


def parallel_merge(G, msq):
    # Create Binary tree graph Bt with nodes labeled 0, 1, ..., len(msq) - 1
    Bt = nx.DiGraph()
    Bt.add_nodes_from(range(len(msq)))
    
    # Keep track of the next available node index in Bt
    next_node_label = len(msq)
    
    # Use a list to track which indices have been merged
    merged_indices = set()

    # Keep merging until Bt is connected
    #while not nx.is_connected(Bt):
    while not nx.is_weakly_connected(Bt):
        # Iterate through msq
        for i in range(len(msq) - 1):
            if i in merged_indices:
                continue  # Skip if already merged

            si = msq[i]

            for j in range(i + 1, len(msq)):
                if j in merged_indices:
                    continue  # Skip if already merged

                sj = msq[j]

                # Find common edges between si and sj
                common_edges = [e for e in G.edges if set(e).intersection(si.nodes()) and set(e).intersection(sj.nodes())]

                if common_edges:
                    # Common edge found, print the first common edge
                    print(f"Common edge found between s{i} and s{j}: {common_edges[0]}")

                    # Add a new node to Bt
                    Bt.add_node(next_node_label)
                    Bt.add_edge(i, next_node_label)
                    Bt.add_edge(j, next_node_label)
                    print(f"Added new node {next_node_label} to Bt, connecting s{i} and s{j} to it.")

                    # Merge si and sj into a new sk
                    sk = nx.Graph()
                    sk.add_nodes_from(si.nodes())
                    sk.add_nodes_from(sj.nodes())
                    sk.add_edges_from(si.edges())
                    sk.add_edges_from(sj.edges())
                    sk.add_edges_from(common_edges)

                    # Add sk to msq and increment the next available node label
                    msq.append(sk)
                    next_node_label += 1

                    # Mark si and sj as merged
                    merged_indices.add(i)
                    merged_indices.add(j)

                    break  # Exit the inner loop once a merge occurs
    
    # draw_graph(Bt)
    if not nx.utils.graphs_equal(msq[-1], G):
        raise ValueError("final graph after last merge does not coincide with initial graph!")

    return Bt, msq

def draw_binary_tree(Bt, **kwargs):
    # Default values
    if kwargs.get("layout", None) is None:
        kwargs["layout"] = "graphviz_dot"
    if kwargs.get("node_color", None) is None:
        kwargs["node_color"] = "lightblue"
    if kwargs.get("show", None) is None:
        kwargs["show"] = True
    if kwargs.get("fig_size", None) is None:
        kwargs["fig_size"] = (8, 6)
    if kwargs.get("title", None) is None:
        kwargs["title"] = "Binary Tree Representation of merging sequence"
    if kwargs.get("edge_color", None) is None:
        kwargs["edge_color"] = "gray"
    
    mgo.draw_graph(Bt, **kwargs)

def find_root(tree: nx.DiGraph) -> int:
    """Find the root node of a directed graph (tree)."""
    for node in tree.nodes:
        if tree.out_degree(node) == 0:
            return node
    raise ValueError("No root found or multiple roots detected.")

def get_nodes_by_layers(tree: nx.DiGraph) -> list[list[int]]:
    """
    Get nodes sorted by layers (levels) of the tree graph as a list of lists.
    Root is at layer 0.
    """
    root = find_root(tree)  # Find the root node
    layers = defaultdict(list)  # To store nodes by layer
    queue = deque([(root, 0)])  # BFS queue with (node, depth)

    while queue:
        node, depth = queue.popleft()
        layers[depth].append(node)
        
        # Add children to the queue
        for child in tree.predecessors(node):
            queue.append((child, depth + 1))

    # Convert to a list of lists sorted by layer
    return [layers[layer] for layer in sorted(layers.keys())]


class MergePattern:
    def __init__(self,
                 msq: list[nx.Graph],
                 bt: nx.DiGraph):
        self._subgraphs = msq
        self._pattern_graph = bt
        #self._construct_subgraphs_of_layers(msq, bt)
        # get nodes sorted according to the binary tree layers; reverse list because bt is a inverted binary tree (root at highest layer)
        self._merge_nodes_by_layer = get_nodes_by_layers(self._pattern_graph)
        self._merge_nodes_by_layer.reverse()
        self._construct_siblings_by_layer()

    @property
    def pattern_graph(self):
        return self._pattern_graph
    @pattern_graph.setter
    def pattern_graph(self, bt: nx.DiGraph):
        self._pattern_graph = bt
        self._merge_nodes_by_layer = get_nodes_by_layers(self._pattern_graph)
        self._construct_siblings_by_layer()
        #self._construct_subgraphs_of_layers(self.subgraphs, self._pattern_graph)

    @property
    def subgraphs(self):
        return self._subgraphs

    def __getitem__(self, layer) -> list[nx.Graph]:
        return [self._subgraphs[i] for i in self._merge_nodes_by_layer[layer]]
        
    
    def __len__(self) -> int:
        return len(self._merge_nodes_by_layer)
    
    ## To-Do: Implement this
    # def __iter__(self) -> Iterable[list[nx.Graph]]:
    #     return
    
    # def _construct_subgraphs_of_layers(self,
    #                              msq: list[nx.Graph],
    #                              bt: nx.DiGraph):
    #     """
    #     Construct a list of lists of all subgraphs. The index of the outer list corresponds to the
    #     layer of the merging pattern tree graph. The inner list contains all subgraphs which should
    #     be merged in this layer.
    #     """
    #     # get the nodes of bt sorted according to the layers (list of list)
    #     bt_nodes_by_layers = get_nodes_by_layers(bt)
    #     # create list of lists containing the actual subgraphs from this
    #     subgraphs_of_layers = []
    #     for layer in bt_nodes_by_layers:
    #         curr_layer_subgraphs = []
    #         for gidx in layer:
    #             curr_layer_subgraphs.append(msq[gidx])
    #         subgraphs_of_layers.append(curr_layer_subgraphs)
    #     
    #     self._subgraphs_of_layers = subgraphs_of_layers

    def _construct_siblings_by_layer(self):
        layers = self._merge_nodes_by_layer

        # List to store sibling tuples layer by layer
        sibling_layers = []
        
        # For each layer, group siblings by their parent
        
        for depth in range(len(layers)):
            sibling_groups = defaultdict(list)
            
            for node in layers[depth]:
                parent = next(self._pattern_graph.successors(node), None)
                sibling_groups[parent].append(node)
            
            # Add only sibling tuples for this layer to the result list
            sibling_layer = [tuple(siblings) for siblings in sibling_groups.values() if len(siblings) > 1]
            sibling_layers.append(sibling_layer)
        
        self._merge_siblings_by_layer = sibling_layers

    def get_merge_pairs(self, layer: int) -> list[tuple[nx.Graph, nx.Graph]]:
        """
        Returns a list of tuples containing the graphs that should be merged in the layer index
        """
        merge_pairs = []
        # list of all subgraphs
        subgraphs = self._subgraphs
        for idx1, idx2 in self._merge_siblings_by_layer[layer]:
            merge_pairs.append((subgraphs[idx1], subgraphs[idx2]))
        return merge_pairs
    
    def draw_subgraphs(self, 
                       layer: Union[int, None] = None,
                       **kwargs):
        """
        Draw the subgraphs (of a certain layer) using draw_graph function from modify_graph_objects
        """
        if kwargs.get("fig_size", None) is None:
            kwargs["fig_size"] = (8,8)

        if layer is None:
            subgraphs = self._subgraphs
            for i in range(len(subgraphs)):
                graph = subgraphs[i]
                mgo.draw_graph(graph, **kwargs)
        elif isinstance(layer, int):
            for i in self._merge_nodes_by_layer[layer]:
                graph = self._subgraphs[i]
                mgo.draw_graph(graph, **kwargs)
        else:
            raise TypeError("Unexpected type of layer keyword! Must be integer or None.")
        
    def draw_pattern_graph(self,
                           **kwargs):
        """
        Draw the binary tree representation of the merging pattern using draw_graph function from modify_graph_objects
        """
        # define default layout for binary tree graph
        if kwargs.get("fig_size", None) is None:
            kwargs["fig_size"] = (8,6)
        if kwargs.get("node_color", None) is None:
            kwargs["node_color"] = "lightblue"
        if kwargs.get("layout", None) is None:
            kwargs["layout"] = "graphviz_dot"
        if kwargs.get("title", None) is None:
            kwargs["title"] = "Binary Tree Representation of merging pattern"
        mgo.draw_graph(self._pattern_graph, **kwargs)
        




def main():
    # List of nodes
    nodes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12]

    # Edge list (each tuple represents an edge between two nodes)
    edges_list = [(0, 4), (1, 6), (2, 3), (3, 7), (4, 9), (5, 6), (6, 8), (7, 9), (8, 9), (0, 10), (10,11),(11,12)]
    
    init_graph = generate_graph(nodes_list, edges_list, use_barabasi=False)
    draw_graph(init_graph)
    # calculate_gate_ss(generate_ibm_graph(nodes_list, edges_list, use_barabasi=False))
    #_,msq,_= calculate_msq(generate_random_graph(10, 0.1, use_barabasi=False))
    _,msq,_= calculate_msq(init_graph)
    for i in range(len(msq)):   
      draw_graph(msq[i])
    
    bt,_ = parallel_merge(init_graph,msq)
    
    draw_binary_tree(bt)

if __name__ == "__main__":
    #main()
    # List of nodes
    nodes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12]

    # Edge list (each tuple represents an edge between two nodes)
    edges_list = [(0, 4), (1, 6), (2, 3), (3, 7), (4, 9), (5, 6), (6, 8), (7, 9), (8, 9), (0, 10), (10,11),(11,12)]
    
    init_graph = generate_graph(nodes_list, edges_list, use_barabasi=False)
    draw_graph(init_graph, show=False)
    # calculate_gate_ss(generate_ibm_graph(nodes_list, edges_list, use_barabasi=False))
    #_,msq,_= calculate_msq(generate_random_graph(10, 0.1, use_barabasi=False))
    _,msq,_= calculate_msq(init_graph, show_status=False)
    # for i in range(len(msq)):   
    #   draw_graph(msq[i], show=False)
    print(f"number of subgraphs is initially {len(msq)}.")
    
    bt,_ = parallel_merge(init_graph,msq)
    
    draw_binary_tree(bt)


