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
from itertools import combinations, groupby
import math
import csv
import concurrent.futures
import time
import modify_graph_objects as mgo


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
        nx.draw(G, node_color="lightgreen", with_labels=True, node_size=500)
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


# MS=[] # a list to store all small stars and merging sequence.
# Picked_Stars=[]
# MSQ=[]
# MSG=[]
# merging_edges_list=[]


def calculate_msq(G, show_status: bool = True):
    """
    Calculates the gates using the SS method on the given graph G.
    """
    MS = []  # a list to store all small stars and merging sequence.
    Picked_Stars = []
    MSQ = []
    MSG = []
    merging_edges_list = []

    MG = {}
    SG = {}
    OPSG = {}
    sid_total_gates = 0
    # total_gates = len(connected_subgraph)-2 # gives the no of nodes,
    tn = []
    edu = []

    l = 0

    MG[0] = G.copy()
    an = list(MG[0].nodes())

    while MG[l].number_of_nodes() > 0:

        if nx.is_empty(MG[l]) == True:
            if show_status:
                print("I am here")
                plt.figure(l + 100)
                nx.draw(MG[l], node_color="lightblue", with_labels=True, node_size=500)
            for node in MG[l].nodes():
                SG[l + 1] = nx.Graph()
                SG[l + 1].add_node(node)
                if show_status:
                    plt.figure(l + 150)
                    nx.draw(
                        SG[l + 1],
                        node_color="lightpink",
                        with_labels=True,
                        node_size=500,
                    )
                l += 1
            break
        else:

            a = max(dict(MG[l].degree()).items(), key=lambda x: x[1])
            b = MG[l].edges(a[0])
            SG[l] = nx.Graph(b)
            MG[l + 1] = MG[l].copy()
            # MG[l + 1].remove_node(a[0])
            MG[l + 1].remove_nodes_from(SG[l].nodes())
            if show_status:
                plt.figure(l + 50)
                # if l==0:
                #     MS.append(SG[l].copy())
                nx.draw(SG[l], node_color="lightpink", with_labels=True, node_size=500)
            tn.extend(list(SG[l].nodes()))
            edu.extend(list(SG[l].edges()))

            check = all(item in tn for item in an)
            if check is True:
                break
            else:
                l += 1
    for i in SG.keys():
        Picked_Stars = list(SG.values())
    # for j in range(len(Picked_Stars)):
    #     draw_graph(Picked_Stars[j])
    # print(SG)

    MSG = Picked_Stars.copy()
    # print("MSG is ",MSG)
    if show_status:
        print("SG is", SG)

    first_time = True

    while len(SG) > 1:
        common_edge_found = False
        common_edge_index = 0

        for i in SG.keys():
            if i == 0:
                continue
            common_edges = [
                e
                for e in G.edges
                if set(e).intersection(SG[0].nodes())
                and set(e).intersection(SG[i].nodes())
            ]
            if show_status:
                print(common_edges)
                print("I am here and loop number is ", i)

            if len(common_edges) != 0:
                if show_status:
                    print("I am here second time and loop number is ", i)
                common_edge_found = True
                common_edge_index = i

                # This is just to copy first MSG[0] to MSQ list
                if first_time:
                    MSQ.append(SG[0].copy())
                first_time = False

                MSQ.append(SG[i].copy())

                # Find a common node in SG[0], SG[i] at random
                common_edge = random.choice(common_edges)
                merging_edges_list.append(common_edge)
                if show_status:
                    print("common edge is", common_edge)

                # Assign new star center nodes to each each graph
                if SG[0].has_node(common_edge[0]):
                    cn_0 = common_edge[0]
                    cn_i = common_edge[1]
                    if show_status:
                        print("common node of cn_i is", cn_i)

                else:
                    cn_0 = common_edge[1]
                    cn_i = common_edge[0]
                    if show_status:
                        print("common node of cn_i is", cn_i)

                if (
                    cn_0 == max(dict(SG[0].degree()).items(), key=lambda x: x[1])[0]
                    or len(SG[0].nodes()) == 2
                ):
                    pass

                else:

                    # Shifting center of MSG[0]
                    SG[0].remove_edges_from(SG[0].edges())
                    # Add edges from common_node to all other nodes in SG[0] (avoid self-edges)
                    for node in SG[0].nodes():
                        if node != cn_0:
                            SG[0].add_edge(
                                cn_0, node, weight=1
                            )  # Adding edge weight of 1
                    sid_total_gates += 2  # For shifting SG[0] star (two H gates)
                    if show_status:
                        draw_graph(SG[0])

                if (
                    cn_i == max(dict(SG[i].degree()).items(), key=lambda x: x[1])[0]
                    or len(SG[i].nodes()) == 2
                ):
                    if show_status:
                        print("shift is not required for cn_i")
                    pass

                else:

                    # Shifting center of MSG[i]
                    SG[i].remove_edges_from(SG[i].edges())
                    # Add edges from common_node to all other nodes in SG[0] (avoid self-edges)
                    for node in SG[i].nodes():
                        if node != cn_i:
                            SG[i].add_edge(
                                cn_i, node, weight=1
                            )  # Adding edge weight of 1
                    sid_total_gates += 2  # For shifting SG[0] star (two H gates)
                    if show_status:
                        draw_graph(SG[i])

                # Merging the two stars
                for node in SG[i].nodes():
                    if node != cn_0:
                        SG[0].add_edge(cn_0, node, weight=1)  # Adding edge weight of 1

                sid_total_gates += 1  # For merging (one CNOT gate)
                if show_status:
                    draw_graph(SG[0])

                # Deleting the MSG[i] star as it is merged to MSG[0]
                # SG.remove(SG[i])

                break

                # print("SS gates are ", sid_total_gates)
        if common_edge_found:
            del SG[common_edge_index]  # Remove the subgraph at index i

        else:
            if show_status:
                print("no common edges found in this iteration")
            break

    # draw_graph(SG[0])
    # print("SS gates are ", sid_total_gates)

    # Adding Star formation gates
    for pq in range(len(MSQ)):
        if MSQ[pq].number_of_nodes() == 2:
            pass
        elif MSQ[pq].number_of_nodes() == 1:
            pass
        else:
            sid_total_gates += MSQ[pq].number_of_nodes() - 2

    return merging_edges_list, MSQ, sid_total_gates


def sequential_merge(G: nx.Graph, msq: list[nx.Graph], show_status: bool = False):
    # Create Binary tree graph Bt with nodes labeled 0, 1, ..., len(msq) - 1
    Bt = nx.DiGraph()
    Bt.add_nodes_from(range(len(msq)))

    # Keep track of the next available node index in Bt
    next_node_label = len(msq)

    orig_msq_len = len(msq)

    i = 0
    # Iterate through msq
    for j in range(1, orig_msq_len):
        si = msq[i]
        sj = msq[j]
        # Find common edges between si and sj
        common_edges = [
            e
            for e in G.edges
            if set(e).intersection(si.nodes()) and set(e).intersection(sj.nodes())
        ]
        if common_edges:
            # Find a common edge to use for the merging
            ## To-Do: choose edge based on error rate and for star states based on number of center shifts
            common_edge = random.choice(common_edges)  # random choice
            # Common edge found
            if show_status:
                print(f"Common edge found between s{i} and s{j}: {common_edge}")

            # Add a new node to Bt
            Bt.add_node(next_node_label)
            Bt.add_edge(i, next_node_label, weight=common_edge)  # add merging edge info
            Bt.add_edge(j, next_node_label, weight=common_edge)  # add merging edge info
            if show_status:
                print(
                    f"Added new node {next_node_label} to Bt, connecting s{i} and s{j} to it."
                )

            # Merge si and sj into a new star graph sk
            # use convention (smaller index, higher index)
            if i < j:
                if len(si) > 1:
                    sk = mgo.merge_star_graphs(
                        si, sj, common_edge, keep_center1=True, keep_center2=True
                    )
                else:
                    sk = mgo.merge_star_graphs(
                        sj, si, common_edge, keep_center1=True, keep_center2=True
                    )

            elif j < i:
                if len(sj) > 1:
                    sk = mgo.merge_star_graphs(
                        sj, si, common_edge, keep_center1=True, keep_center2=True
                    )
                else:
                    sk = mgo.merge_star_graphs(
                        si, sj, common_edge, keep_center1=True, keep_center2=True
                    )
            else:
                raise ValueError("subgraph indices coincide!")

            # Add sk to msq and increment the next available node label
            msq.append(sk)
            i = next_node_label  # last element in msq is now part of next merge
            next_node_label += 1

    return Bt, msq


def check_merge_possibility(subgraph1, subgraph2, G, merged_edges):
    """
    Check if there is a direct edge connecting any node in subgraph1 to any node in subgraph2,
    excluding edges that involve already-merged nodes.

    Parameters:
    subgraph1 (networkx.Graph): The first subgraph.
    subgraph2 (networkx.Graph): The second subgraph.
    G (networkx.Graph): The main graph from which the subgraphs are derived.
    merged_edges (list): List of edges (as ordered pairs) already used for merging.

    Returns:
    tuple or None: A randomly chosen edge connecting the two subgraphs if one exists and satisfies the constraints; else None.
    """
    # Create a set of nodes already involved in merged edges
    merged_nodes = set(node for edge in merged_edges for node in edge)

    # Find all edges in G that connect nodes in subgraph1 to nodes in subgraph2
    connecting_edges = [
        edge
        for edge in G.edges
        if (edge[0] in subgraph1.nodes and edge[1] in subgraph2.nodes)
        or (edge[1] in subgraph1.nodes and edge[0] in subgraph2.nodes)
    ]

    # Exclude edges involving any node in merged_nodes
    valid_edges = [
        edge
        for edge in connecting_edges
        if edge[0] not in merged_nodes and edge[1] not in merged_nodes
    ]

    # If valid connecting edges are found, randomly pick one
    if valid_edges:
        return random.choice(valid_edges)
    else:
        return None


def merging_many_stars(subgraphs, G, merged_edges):
    """
    Merge a list of subgraphs into a single graph, including additional edges.

    Parameters:
    subgraphs (list of networkx.Graph): A list of subgraphs to merge.
    G (networkx.Graph): The main graph from which the subgraphs are derived.
    merge_edges (list of tuples): A list of edges to add to the merged graph.

    Returns:
    networkx.Graph: The merged graph containing all nodes and edges from the subgraphs
                    and the additional edges from merge_edges.
    """
    # Create an empty graph for the merged result
    merged_graph = nx.Graph()

    # Add nodes and edges from each subgraph to the merged graph
    for subgraph in subgraphs:
        merged_graph.add_nodes_from(subgraph.nodes)
        merged_graph.add_edges_from(subgraph.edges)

    # Add additional edges from the merge_edges list
    for edge in merged_edges:
        if G.has_edge(*edge):  # Ensure the edge exists in the main graph
            merged_graph.add_edge(*edge)
        else:
            print(f"Warning: Edge {edge} not in main graph G and was not added.")

    return merged_graph


def find_merging_stars_with_digraph(msq, G):
    """
    Find merging stars and track the merging process using a directed graph (DiGraph).

    Parameters:
    msq (list): A list of subgraphs (stars) to merge.
    G (networkx.Graph): The main graph to check merge possibilities.

    Returns:
    tuple: merged_stars, merged_edges, merge_associations, and the initial DiGraph.
    """
    # Initialize variables
    msq_original = msq[:]  # Keep the original msq intact
    merged_stars = []  # List to track merged stars
    merged_edges = []  # List to track edges used for merging
    merge_associations = []  # Track which edge merges which stars

    # julius.wallnoefer@uibk.ac.at

    # Initialize a directed graph (DiGraph)
    D = nx.DiGraph()
    D.add_nodes_from(range(len(msq_original)))  # Nodes are indices of msq elements

    # Loop until the DiGraph is strongly connected
    # while len(list(nx.strongly_connected_components(D))) > 1:
    for sp in range(1):
        # Iterate through all pairs of stars
        for i, si in enumerate(msq):
            if si in merged_stars:  # Skip the stars that are already merged
                continue
            for j, sj in enumerate(msq):
                if i == j or sj in merged_stars:  # Skip self-checks and merged stars
                    continue

                # Check merge possibility between si and sj
                edge = check_merge_possibility(si, sj, G, merged_edges)
                if edge is not None:
                    # Add edge to merged_edges and update associations
                    merged_edges.append(edge)
                    merge_associations.append((edge, si, sj, i, j))

                    # Add si and sj to merged_stars (ensure uniqueness)
                    if si not in merged_stars:
                        merged_stars.append(si)
                    if sj not in merged_stars:
                        merged_stars.append(sj)

        # After processing all merges, check if the graph is connected
        # if nx.is_strongly_connected(D):
        #     break
        for p, subgraph in enumerate(merged_stars):
            print(f"Drawing subgraph {i + 1}")
            draw_graph(subgraph, node_color="lightblue")

        print("merged stars are ", merged_stars)
        print("merged edges are ", merged_edges)
        print("merge_assosiations ", merge_associations)
        
        for k in merge_associations:
            print("mergee_element",k)


        draw_graph(D)

        # Initialize a dictionary to track which node each merged star is associated with
        

        for edge, si, sj, i, j in merge_associations:
            # Check if there are edges from i or j
            edges_i = list(D.out_edges(i))  # Get outgoing edges of i
            edges_j = list(D.out_edges(j))  # Get outgoing edges of j
            
            # Check if any edges exist for i
            if edges_i:
                connected_node = edges_i[0][1]  # Get the node connected to i (the destination of the edge)
            elif edges_j:
                connected_node = edges_j[0][1]  # Get the node connected to j (the destination of the edge)
            else:
                connected_node = None  # No edges from j
            
            # If no connection found, create a new node
            if connected_node is None:
                # Get the next node number by checking the current number of nodes in the graph
                new_node = len(D.nodes)
                D.add_node(new_node)  # Add the new node to the graph
                
                # Add directed edges from i and j to the new node
                D.add_edge(i, new_node)
                D.add_edge(j, new_node)

                sk = merging_many_stars([si, sj], G, [edge])
                # Add the new merged star to msq
                msq.append(sk)
                
                # connected_node = new_node  # Update connected_node to the new node number
            else:

                if i == connected_node:
                    D.add_edge(j, connected_node)

                    sn = msq[connected_node]  # Get the subgraph at index connected_node

                    sk = merging_many_stars([sj, sn], G, [edge])
                    msq.append(sk)
                else:
                    D.add_edge(i, connected_node)

                    sn = msq[connected_node]  # Get the subgraph at index connected_node

                    sk = merging_many_stars([si, sn], G, [edge])
                    msq.append(sk)



                # If connected_node is already there, join i and j to that node
                D.add_edge(i, connected_node)
                D.add_edge(j, connected_node)

                sk = merging_many_stars([si, sj], G, [edge])
                msq.append(sk)
        
        

        merged_edges.clear()
        connected_node = None
        print(msq[-2].nodes())
            

        draw_graph(D,node_color="pink")



    return merged_stars, merged_edges, merge_associations


# def parallel_merge(G: nx.Graph, msq: list[nx.Graph], show_status: bool = False):
#     # Create Binary tree graph Bt with nodes labeled 0, 1, ..., len(msq) - 1
#     Bt = nx.DiGraph()
#     Bt.add_nodes_from(range(len(msq)))

#     # Keep track of the next available node index in Bt
#     next_node_label = len(msq)

#     # Use a list to track which indices have been merged
#     merged_indices = set()

#     # Keep merging until Bt is connected
#     # while not nx.is_connected(Bt):
#     while not nx.is_weakly_connected(Bt):
#         # Iterate through msq
#         for i in range(len(msq) - 1):
#             if i in merged_indices:
#                 continue  # Skip if already merged

#             si = msq[i]

#             for j in range(i + 1, len(msq)):
#                 if j in merged_indices:
#                     continue  # Skip if already merged

#                 sj = msq[j]

#                 # Find common edges between si and sj
#                 common_edges = [
#                     e
#                     for e in G.edges
#                     if set(e).intersection(si.nodes())
#                     and set(e).intersection(sj.nodes())
#                 ]

#                 if common_edges:
#                     # Find a common edge to use for the merging
#                     ## To-Do: choose edge based on error rate and for star states based on number of center shifts
#                     common_edge = random.choice(common_edges)  # random choice
#                     # Common edge found
#                     if show_status:
#                         print(f"Common edge found between s{i} and s{j}: {common_edge}")

#                     # Add a new node to Bt
#                     Bt.add_node(next_node_label)
#                     Bt.add_edge(
#                         i, next_node_label, weight=common_edge
#                     )  # add merging edge info
#                     Bt.add_edge(
#                         j, next_node_label, weight=common_edge
#                     )  # add merging edge info
#                     if show_status:
#                         print(
#                             f"Added new node {next_node_label} to Bt, connecting s{i} and s{j} to it."
#                         )

#                     # Add sk to msq and increment the next available node label
#                     msq.append(sk)
#                     next_node_label += 1

#                     # Mark si and sj as merged
#                     merged_indices.add(i)
#                     merged_indices.add(j)

#                     break  # Exit the inner loop once a merge occurs

#     return Bt, msq


# def draw_binary_tree(Bt, **kwargs):
#     # Default values
#     if kwargs.get("layout", None) is None:
#         kwargs["layout"] = "graphviz_dot"
#     if kwargs.get("node_color", None) is None:
#         kwargs["node_color"] = "lightblue"
#     if kwargs.get("show", None) is None:
#         kwargs["show"] = True
#     if kwargs.get("fig_size", None) is None:
#         kwargs["fig_size"] = (8, 6)
#     if kwargs.get("title", None) is None:
#         kwargs["title"] = "Binary Tree Representation of merging sequence"
#     if kwargs.get("edge_color", None) is None:
#         kwargs["edge_color"] = "gray"

#     mgo.draw_graph(Bt, **kwargs)


# def find_root(tree: nx.DiGraph) -> int:
#     """Find the root node of a directed graph (tree)."""
#     for node in tree.nodes:
#         if tree.out_degree(node) == 0:
#             return node
#     raise ValueError("No root found or multiple roots detected.")


# def get_nodes_by_layers(tree: nx.DiGraph) -> list[list[int]]:
#     """
#     Get nodes sorted by layers (levels) of the tree graph as a list of lists.
#     Root is at layer 0.
#     """
#     root = find_root(tree)  # Find the root node
#     layers = defaultdict(list)  # To store nodes by layer
#     queue = deque([(root, 0)])  # BFS queue with (node, depth)

#     while queue:
#         node, depth = queue.popleft()
#         layers[depth].append(node)

#         # Add children to the queue
#         for child in tree.predecessors(node):
#             queue.append((child, depth + 1))

#     # Convert to a list of lists sorted by layer
#     return [layers[layer] for layer in sorted(layers.keys())]


# class MergePattern:
#     def __init__(self, init_graph: nx.Graph, msq: list[nx.Graph], bt: nx.DiGraph):
#         self._initial_graph = init_graph
#         self._subgraphs = msq
#         self._pattern_graph = bt
#         # self._construct_subgraphs_of_layers(msq, bt)
#         # get nodes sorted according to the binary tree layers; reverse list because bt is a inverted binary tree (root at highest layer)
#         self._merge_nodes_by_layer = get_nodes_by_layers(self._pattern_graph)
#         self._merge_nodes_by_layer.reverse()
#         self._construct_siblings_by_layer()

#     @classmethod
#     def from_graph_sequential(cls, init_graph: nx.Graph):
#         _, msq, _ = calculate_msq(init_graph, show_status=False)
#         bt, _ = sequential_merge(init_graph, msq)
#         return cls(init_graph, msq, bt)

#     @classmethod
#     def from_graph_parallel(cls, init_graph: nx.Graph):
#         _, msq, _ = calculate_msq(init_graph, show_status=False)
#         bt, _ = parallel_merge(init_graph, msq)
#         return cls(init_graph, msq, bt)

#     @property
#     def initial_graph(self):
#         return self._initial_graph

#     @property
#     def pattern_graph(self):
#         return self._pattern_graph

#     @pattern_graph.setter
#     def pattern_graph(self, bt: nx.DiGraph):
#         self._pattern_graph = bt
#         self._merge_nodes_by_layer = get_nodes_by_layers(self._pattern_graph)
#         self._construct_siblings_by_layer()
#         # self._construct_subgraphs_of_layers(self.subgraphs, self._pattern_graph)

#     @property
#     def subgraphs(self):
#         return self._subgraphs

#     def __getitem__(self, layer) -> list[nx.Graph]:
#         return [self._subgraphs[i] for i in self._merge_nodes_by_layer[layer]]

#     def __len__(self) -> int:
#         return len(self._merge_nodes_by_layer)

#     ## To-Do: Implement this
#     # def __iter__(self) -> Iterable[list[nx.Graph]]:
#     #     return

#     # def _construct_subgraphs_of_layers(self,
#     #                              msq: list[nx.Graph],
#     #                              bt: nx.DiGraph):
#     #     """
#     #     Construct a list of lists of all subgraphs. The index of the outer list corresponds to the
#     #     layer of the merging pattern tree graph. The inner list contains all subgraphs which should
#     #     be merged in this layer.
#     #     """
#     #     # get the nodes of bt sorted according to the layers (list of list)
#     #     bt_nodes_by_layers = get_nodes_by_layers(bt)
#     #     # create list of lists containing the actual subgraphs from this
#     #     subgraphs_of_layers = []
#     #     for layer in bt_nodes_by_layers:
#     #         curr_layer_subgraphs = []
#     #         for gidx in layer:
#     #             curr_layer_subgraphs.append(msq[gidx])
#     #         subgraphs_of_layers.append(curr_layer_subgraphs)
#     #
#     #     self._subgraphs_of_layers = subgraphs_of_layers

#     def _construct_siblings_by_layer(self):
#         layers = self._merge_nodes_by_layer

#         # List to store sibling tuples layer by layer
#         sibling_layers = []

#         # For each layer, group siblings by their parent

#         for depth in range(len(layers)):
#             sibling_groups = defaultdict(list)

#             for node in layers[depth]:
#                 parent = next(self._pattern_graph.successors(node), None)
#                 sibling_groups[parent].append(node)

#             # Add only sibling tuples for this layer to the result list
#             sibling_layer = [
#                 tuple(siblings)
#                 for siblings in sibling_groups.values()
#                 if len(siblings) > 1
#             ]
#             sibling_layers.append(sibling_layer)

#         self._merge_siblings_by_layer = sibling_layers

#     def get_initial_subgraphs(self) -> list[nx.Graph]:
#         pattern_graph_leafs = [
#             node
#             for node in self.pattern_graph.nodes
#             if self.pattern_graph.in_degree(node) == 0
#         ]

#         subgraphs = []
#         for idx in pattern_graph_leafs:
#             subgraphs.append(self._subgraphs[idx])

#         return subgraphs

#     def get_merge_pairs(
#         self, layer: int
#     ) -> list[tuple[nx.Graph, nx.Graph, tuple[int, int]]]:
#         """
#         Returns a list of tuples containing the graphs that should be merged in the layer index and the merging edge
#         """
#         merge_pairs = []
#         # list of all subgraphs
#         subgraphs = self._subgraphs
#         for idx1, idx2 in self._merge_siblings_by_layer[layer]:
#             parent_idx = next(self._pattern_graph.successors(idx1))
#             idx_pair = (idx1, parent_idx)  # successor implies that edge is of this form

#             merging_edge = self._pattern_graph[idx1][parent_idx].get("weight", None)
#             if not merging_edge:
#                 raise ValueError(
#                     "Unable to retrieve merging edge at pattern graph indices ({}, {})!".format(
#                         idx1, parent_idx
#                     )
#                 )
#             # follow convention that the subgraph with the smallest index comes first (relevant for merging order, i.e., for graph states which center is kept)
#             # if this convention is changed also the convention in parallel_merge and sequential merge function must be changed
#             if idx1 < idx2:
#                 merge_pairs.append((subgraphs[idx1], subgraphs[idx2], merging_edge))
#             elif idx2 < idx1:
#                 merge_pairs.append((subgraphs[idx2], subgraphs[idx1], merging_edge))
#             else:
#                 raise ValueError("merge sibling indices coincide!")
#         return merge_pairs

#     def draw_subgraphs(self, layer: Union[int, None] = None, **kwargs):
#         """
#         Draw the subgraphs (of a certain layer) using draw_graph function from modify_graph_objects
#         """
#         # Default values
#         if kwargs.get("node_color", None) is None:
#             kwargs["node_color"] = "yellow"
#         if kwargs.get("layout", None) is None:
#             kwargs["layout"] = "circular"
#         show_plots_at_end = False
#         if kwargs.get("show", None) is None:
#             show_plots_at_end = True
#             kwargs["show"] = False
#         if kwargs.get("fig_size", None) is None:
#             kwargs["fig_size"] = (8, 8)
#         if kwargs.get("title", None) is None:
#             kwargs["title"] = "Graph Visualization"

#         if layer is None:
#             subgraphs = self._subgraphs
#             for i in range(len(subgraphs)):
#                 if i == len(subgraphs) - 1:
#                     if show_plots_at_end:
#                         kwargs["show"] = True
#                 graph = subgraphs[i]
#                 mgo.draw_graph(graph, **kwargs)
#         elif isinstance(layer, int):
#             for i in self._merge_nodes_by_layer[layer]:
#                 if i == self._merge_nodes_by_layer[layer][-1]:
#                     if show_plots_at_end:
#                         kwargs["show"] = True
#                 graph = self._subgraphs[i]
#                 mgo.draw_graph(graph, **kwargs)
#         else:
#             raise TypeError(
#                 "Unexpected type of layer keyword! Must be integer or None."
#             )

#     def draw_pattern_graph(self, **kwargs):
#         """
#         Draw the binary tree representation of the merging pattern using draw_graph function from modify_graph_objects
#         """
#         # define default layout for binary tree graph
#         if kwargs.get("layout", None) is None:
#             kwargs["layout"] = "graphviz_dot"
#         if kwargs.get("node_color", None) is None:
#             kwargs["node_color"] = "lightblue"
#         if kwargs.get("show", None) is None:
#             kwargs["show"] = True
#         if kwargs.get("fig_size", None) is None:
#             kwargs["fig_size"] = (8, 6)
#         if kwargs.get("title", None) is None:
#             kwargs["title"] = "Binary Tree Representation of merging pattern"
#         if kwargs.get("edge_color", None) is None:
#             kwargs["edge_color"] = "gray"

#         mgo.draw_graph(self._pattern_graph, **kwargs)


if __name__ == "__main__":
    # main()
    # List of nodes
    nodes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Edge list (each tuple represents an edge between two nodes)
    edges_list = [
        (0, 4),
        (1, 6),
        (2, 3),
        (3, 7),
        (4, 9),
        (5, 6),
        (5, 13),
        (6, 8),
        (7, 9),
        (8, 9),
        (0, 10),
        (10, 11),
        (11, 12),
    ]

    init_graph = generate_graph(nodes_list, edges_list, use_barabasi=False)
    # draw_graph(init_graph, show=False)
    # calculate_gate_ss(generate_ibm_graph(nodes_list, edges_list, use_barabasi=False))
    # _,msq,_= calculate_msq(generate_random_graph(10, 0.1, use_barabasi=False))
    _, msq1, _ = calculate_msq(init_graph, show_status=False)
    # for i in range(len(msq)):
    #   draw_graph(msq[i], show=False)
    print(f"number of subgraphs is initially {len(msq1)}.")

    find_merging_stars_with_digraph(msq1, init_graph)
    draw_graph(init_graph)

    # bt1, _ = parallel_merge(init_graph, msq1)

    # draw_graph(bt1, show=True)

    # # draw_binary_tree(bt1, show=False)
    # # Define MergePattern instance for parallel merge
    # ptrn1 = MergePattern(init_graph, msq1, bt1)

    # _, msq2, _ = calculate_msq(init_graph, show_status=False)
    # # for i in range(len(msq)):
    # #   draw_graph(msq[i], show=False)
    # print(f"number of subgraphs is initially {len(msq2)}.")

    # bt2, _ = sequential_merge(init_graph, msq2)

    # # draw_binary_tree(bt2, show=True)
    # # Define MergePattern instance for sequential merge
    # ptrn2 = MergePattern(init_graph, msq2, bt2)

    # # print all merging pairs
    # print("merge pairs for parallel merge:")
    # for layer in range(len(ptrn1)):
    #     curr_pairs_graph = ptrn1.get_merge_pairs(layer)
    #     curr_pairs = ptrn1._merge_siblings_by_layer[layer]
    #     print(f"layer {layer}: merge pairs {curr_pairs}")
    #     for graph_pair_idcs, graph_pair in zip(curr_pairs, curr_pairs_graph):
    #         print(f"    graph {graph_pair_idcs[0]}: nodes={graph_pair[0].nodes}")
    #         print(f"    graph {graph_pair_idcs[1]}: nodes={graph_pair[1].nodes}")
    #         print(f"    merging edge {graph_pair[2]}")
    # print()
    # print("merge pairs for sequential merge:")
    # for layer in range(len(ptrn2)):
    #     curr_pairs_graph = ptrn2.get_merge_pairs(layer)
    #     curr_pairs = ptrn2._merge_siblings_by_layer[layer]
    #     print(f"layer {layer}: merge pairs {curr_pairs}")
    #     for graph_pair_idcs, graph_pair in zip(curr_pairs, curr_pairs_graph):
    #         print(f"    graph {graph_pair_idcs[0]}: nodes={graph_pair[0].nodes}")
    #         print(f"    graph {graph_pair_idcs[1]}: nodes={graph_pair[1].nodes}")
    #         print(f"    merging edge {graph_pair[2]}")

    # # draw the binary tree representing the merge pattern
    # ptrn1.draw_pattern_graph(show_weights=True, show=False)
    # ptrn2.draw_pattern_graph(show_weights=True, show=False)

    # # draw all subgraphs
    # # ptrn1.draw_subgraphs(show=False)
    # # ptrn2.draw_subgraphs()

    # ptrn1.draw_subgraphs(layer=-1, show=False)
    # ptrn2.draw_subgraphs(layer=-1)
