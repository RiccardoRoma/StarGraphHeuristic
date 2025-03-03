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
from networkx.algorithms.isomorphism import is_isomorphic
from collections import deque



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

def find_sink_and_longest_leaf(G):
    # Find the sink node
    for node in G.nodes:
        if G.out_degree(node) == 0 and G.in_degree(node) > 0:
            sink_node = node
            break
    else:
        return None, None  # No sink node found

    # Reverse the graph to follow incoming edges
    reversed_G = G.reverse()
    
    # Perform BFS to find the longest path from the sink node
    visited = set()
    queue = deque([(sink_node, 0)])  # (node, distance)
    visited.add(sink_node)
    
    longest_leaf = None
    max_distance = -1
    
    while queue:
        node, distance = queue.popleft()
        
        # If the node is a leaf node and has the longest distance, update
        if reversed_G.out_degree(node) == 0:
            if distance > max_distance:
                max_distance = distance
                longest_leaf = node
        
        for neighbor in reversed_G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return sink_node, (longest_leaf, max_distance)

def find_merging_tree(msq, G):
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

    # Initialize a directed graph (DiGraph)
    D = nx.DiGraph()
    D.add_nodes_from(range(len(msq_original)))  # Nodes are indices of msq elements

    # Loop until the DiGraph is strongly connected
    while not set(G.nodes) == set(msq[-1].nodes):

    # for sp in range(3):
        
        for i, si in enumerate(msq):
            if si in merged_stars:  # Skip the stars that are already merged
                continue
            for j, sj in enumerate(msq):
                if i == j or sj in merged_stars:  # Skip self-checks and merged stars
                    continue

                edge = check_merge_possibility(si, sj, G, merged_edges)
                if edge is not None:
                    # Add edge to merged_edges and update associations
                    merged_edges.append(edge)
                    merge_associations.append((edge, si, sj, i, j))
                    # if sp == 1:
                    #     print("edge is ", edge, si.nodes(),sj.nodes(), i, j)


                    # Add si and sj to merged_stars (ensure uniqueness)
                    if si not in merged_stars:
                        merged_stars.append(si)
                    if sj not in merged_stars:
                        merged_stars.append(sj)

        # Initialize variables to check if it is new merge or not
        edi = False
        edj = False
        

        for edge, si, sj, i, j in merge_associations:
            # print("edge is ", edge, i,j)
            # Check if there are edges from i or j
            edges_i = list(D.out_edges(i))  # Get outgoing edges of i
            edges_j = list(D.out_edges(j))  # Get outgoing edges of j
            
            # Check if any edges exist for i
            if edges_i:
                connected_node = edges_i[0][1]  # Get the node connected to i (the destination of the edge)
                edi = True
            elif edges_j:
                connected_node = edges_j[0][1]  # Get the node connected to j (the destination of the edge)
                edj = True
            else:
                connected_node = None  # No edges from j
            
            # If no connection found, create a new node
            if connected_node is None:
                # Get the next node number by checking the current number of nodes in the graph
                new_node = len(D.nodes)
                D.add_node(new_node)  # Add the new node to the graph
                
                # Add directed edges from i and j to the new node
                
                # if i != new_node:
                #     D.add_edge(i, new_node)
                # if j != new_node:
                #     D.add_edge(j, new_node)
                D.add_edge(i, new_node, weight=edge)
                D.add_edge(j, new_node, weight=edge)

                sk = merging_many_stars([si, sj], G, [edge])
                # Add the new merged star to msq
                msq.append(sk)
                
            else:

                if edi == True:
                    
                    # if j != new_node:
                    #     D.add_edge(j, new_node)
                    if j != connected_node:
                        D.add_edge(j, connected_node, weight=edge)

                    sn = msq[connected_node]  # Get the subgraph at index connected_node

                    sk = merging_many_stars([sj, sn], G, [edge])
                    msq[connected_node] = sk
                    
                elif edj == True:

                    # if i != new_node:
                    #     D.add_edge(i, new_node)
                    if i != connected_node:
                        D.add_edge(i, connected_node, weight=edge)

                    sn = msq[connected_node]  # Get the subgraph at index connected_node

                    sk = merging_many_stars([si, sn], G, [edge])
                    msq[connected_node] = sk

        merged_edges.clear()
        merge_associations.clear()
        connected_node = None
        edi = False
        edj = False

        
        # for subgraph in msq:
        #     print("msq stars are ",subgraph.nodes())

        # for k in merge_associations:
        #     print("mergee_element",k)

        # for g in merged_stars:
        #     print("merged stars are ", g.nodes())
        draw_graph(D,node_color="pink", show=False, layout="graphviz_dot", show_weights=True)
       

    return D, msq

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

    draw_graph(init_graph, show=False)
    D,_=find_merging_tree(msq1, init_graph)
    for i, si in enumerate(msq1):
        print(f"subgraph {i}: nodes={si.nodes}")
    sink_node, (longest_leaf, max_distance) = find_sink_and_longest_leaf(D)
    print("sink node is ", sink_node, "longest leaf is ", longest_leaf, "max distance is ", max_distance)
    
    plt.show()