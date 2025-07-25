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
import copy


def generate_random_graph(n, p, use_barabasi, show_plot=True):
    """Create a connected random graph according to the Barabasi-Albert model or the Edros-Renyi model.

    Args:
        n: Number of nodes in the graph, i.e., range(n) = list of the nodes.
        p: Measure of connectedness in both models.
        use_barabasi: Bool flag to use Barabasi-Albert model. 
        show_plot: plot generated random graph

    Returns:
        Random graphs as a networkx graph object
    """
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
        if show_plot:
            plt.figure()
            draw_graph(G, node_color='lightgreen', 
                       with_labels=True, 
                       node_size=500,
                       show=show_plot)
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
    MS=[] # a list to store all small stars and merging sequence.  
    Picked_Stars=[]
    MSQ=[]
    MSG=[]
    merging_edges_list=[]
    
    MG = {}
    SG = {}
    OPSG = {}
    sid_total_gates=0
    #total_gates = len(connected_subgraph)-2 # gives the no of nodes, 
    tn = []
    edu = []
    
    l = 0

   

    MG[0] = G.copy()
    an = list(MG[0].nodes())

    while MG[l].number_of_nodes() > 0:
        
        if nx.is_empty(MG[l])==True:
            if show_status:
                print("I am here")
                plt.figure(l + 100)
                nx.draw(MG[l], 
                        node_color='lightblue', 
                        with_labels=True, 
                        node_size=500)
            for node in MG[l].nodes():
                SG[l + 1] = nx.Graph()
                SG[l + 1].add_node(node)
                if show_status:
                    plt.figure(l + 150)
                    nx.draw(SG[l + 1], 
                            node_color='lightpink', 
                            with_labels=True, 
                            node_size=500)
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
                plt.figure(l+50)
                # if l==0:
                #     MS.append(SG[l].copy())
                nx.draw(SG[l], 
                        node_color='lightpink', 
                        with_labels=True, 
                        node_size=500)
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
        print("SG is",SG)
    
    

        
    first_time=True        
    
    if len(SG) == 1:
        # This is just to copy first MSG[0] to MSQ list
        MSQ.append(SG[0].copy())
    else:
        while len(SG) > 1:
            common_edge_found = False
            common_edge_index = 0
            
            
            for i in SG.keys():
                if i == 0:
                    continue
                common_edges = [e for e in G.edges if set(e).intersection(SG[0].nodes()) and set(e).intersection(SG[i].nodes())]
                if show_status:
                    print(common_edges)
                    print("I am here and loop number is ",i)
                
                                  
                if len(common_edges)!=0:
                    if show_status:
                        print("I am here second time and loop number is ",i)
                    common_edge_found = True
                    common_edge_index = i
                    
                    # This is just to copy first MSG[0] to MSQ list
                    if first_time:
                        MSQ.append(SG[0].copy())
                    first_time=False
                    
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
                        
                    
                    if cn_0 == max(dict(SG[0].degree()).items(), key=lambda x: x[1])[0] or len(SG[0].nodes()) == 2:
                        pass
                   
                    else:
                                   
                        # Shifting center of MSG[0]
                        SG[0].remove_edges_from(SG[0].edges())      
                        # Add edges from common_node to all other nodes in SG[0] (avoid self-edges)
                        for node in SG[0].nodes():
                            if node != cn_0:
                                SG[0].add_edge(cn_0, node, weight=1)  # Adding edge weight of 1
                        sid_total_gates += 2 # For shifting SG[0] star (two H gates)
                        if show_status:
                            draw_graph(SG[0])
                        
                
                    if cn_i == max(dict(SG[i].degree()).items(), key=lambda x: x[1])[0] or len(SG[i].nodes()) == 2:
                        if show_status:
                            print("shift is not required for cn_i")
                        pass
                   
                    else:
                
                        # Shifting center of MSG[i]
                        SG[i].remove_edges_from(SG[i].edges())      
                        # Add edges from common_node to all other nodes in SG[0] (avoid self-edges)
                        for node in SG[i].nodes():
                            if node != cn_i:
                                SG[i].add_edge(cn_i, node, weight=1)  # Adding edge weight of 1
                        sid_total_gates += 2 # For shifting SG[0] star (two H gates)
                        if show_status:
                            draw_graph(SG[i])
                        
                    
                    # Merging the two stars                   
                    for node in SG[i].nodes():
                        if node != cn_0:
                            SG[0].add_edge(cn_0, node, weight=1)  # Adding edge weight of 1
                    
                    sid_total_gates += 1 # For merging (one CNOT gate)
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
        if MSQ[pq].number_of_nodes()==2:
            pass
        elif MSQ[pq].number_of_nodes()==1: 
            pass
        else:
            sid_total_gates += MSQ[pq].number_of_nodes()-2
        
    
    return merging_edges_list, MSQ, sid_total_gates

def compute_scaling_factor(G, target):
    """
    Compute the scaling factor given a graph G and a target degree.

    Parameters:
        G (networkx.Graph): The input graph.
        target (int): The desired target degree.

    Returns:
        float: The computed scaling factor.
    """
    if not isinstance(target, int):
        raise ValueError(f"Target center node degree is not an integer: {target}!")
    if target < 0:
        raise ValueError(f"Target center node degree must be a positive integer: {target}")
    
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    if avg_degree == 0:
        return 0.0  # Avoid division by zero

    scaling_factor = target / avg_degree
    return scaling_factor


def calculate_msq_avg_degree(G, scaling_factor: float = 1.0, show_status: bool = True) -> tuple[list[nx.Graph], int]:
    """Calculates the to-be-merged star subgraphs using the SS method on the given graph G.
    Modified to pick stars of (roughly) consistent size based on the average degree.
    The star graph size relates to the degree of the center node.

    Args:
        G: initial Graph
        scaling_factor: Gives the percentage of the to-achieve subgraph center node degree in relation to the average degree in the initial graph. Defaults to 1.0, which means the center node degree should be equal to the average degree.
        show_status: Bool flag to show intermediate plots of the function. Defaults to True.

    Returns:
        list of star subgraphs that should be merged, target center node degree of the star subgraphs
    """

    Picked_Stars = []
    MSQ = []
    MG = {}
    SG = {}
    tn = []
    edu = []
    l = 0

    # Compute target degree based on the average degree of the original graph
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    # scaling_factor = 1.0  # adjust this factor as needed
    target = round(scaling_factor * avg_degree)

    # target = round(sum(dict(G.degree()).values()) / G.number_of_nodes())

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
            # Instead of always picking the highest degree node,
            # choose a node whose degree is as close as possible to the target.
            degree_dict = dict(MG[l].degree())
            # Prefer nodes with degree >= target.
            candidates = [
                (node, deg) for node, deg in degree_dict.items() if deg >= target
            ]
            if candidates:
                chosen = min(candidates, key=lambda x: x[1] - target)
            else:
                # If no candidate meets the target, choose one closest to target by absolute difference.
                chosen = min(degree_dict.items(), key=lambda x: abs(x[1] - target))
            center = chosen[0]
            neighbors = list(MG[l].neighbors(center))
            # To enforce consistency in star sizes, if the node has more neighbors than the target,
            # select only a random subset of 'target' neighbors.
            if len(neighbors) > target:
                chosen_neighbors = random.sample(neighbors, target)
            else:
                chosen_neighbors = neighbors

            # Build the star subgraph from the center to the selected neighbors.
            edges = [(center, nb) for nb in chosen_neighbors]
            SG[l] = nx.Graph()
            SG[l].add_edges_from(edges)

            MG[l + 1] = MG[l].copy()
            MG[l + 1].remove_nodes_from(SG[l].nodes())
            if show_status:
                plt.figure(l + 50)
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
    MSQ = Picked_Stars.copy()

    return MSQ, target


def sequential_merge(G: nx.Graph, msq_in: list[nx.Graph], show_status: bool = False):
    msq = copy.deepcopy(msq_in) # Keep the original msq intact
    # Create Binary tree graph Bt with nodes labeled 0, 1, ..., len(msq) - 1
    Bt = nx.DiGraph()
    Bt.add_nodes_from(range(len(msq)))

    # Condition for single step generation (msq has just one element)
    if len(msq) == 1:
        return Bt, msq
    
    # Keep track of the next available node index in Bt
    next_node_label = len(msq)

    orig_msq_len = len(msq)
    
    i = 0
    # Iterate through msq
    for j in range(1,orig_msq_len):
        si = msq[i]
        sj = msq[j]
        # Find common edges between si and sj
        common_edges = [e for e in G.edges if set(e).intersection(si.nodes()) and set(e).intersection(sj.nodes())]
        if common_edges:
            # Find a common edge to use for the merging
            ## To-Do: choose edge based on error rate and for star states based on number of center shifts
            common_edge = random.choice(common_edges) # random choice
            # Common edge found
            if show_status:
                print(f"Common edge found between s{i} and s{j}: {common_edge}")

            # Add a new node to Bt
            Bt.add_node(next_node_label)
            Bt.add_edge(i, next_node_label, weight=common_edge) # add merging edge info
            Bt.add_edge(j, next_node_label, weight=common_edge) # add merging edge info
            if show_status:
                print(f"Added new node {next_node_label} to Bt, connecting s{i} and s{j} to it.")

            # Merge si and sj into a new star graph sk
            # use convention (smaller index, higher index)
            if i < j:
                if len(si) > 1:
                    sk = mgo.merge_star_graphs(si, sj, common_edge, keep_center1=True, keep_center2=True)
                else:
                    sk = mgo.merge_star_graphs(sj, si, common_edge, keep_center1=True, keep_center2=True)
                
            elif j < i:
                if len(sj) > 1:
                    sk = mgo.merge_star_graphs(sj, si, common_edge, keep_center1=True, keep_center2=True)
                else:
                    sk = mgo.merge_star_graphs(si, sj, common_edge, keep_center1=True, keep_center2=True)
            else:
                raise ValueError("subgraph indices coincide!")

            # Add sk to msq and increment the next available node label
            msq.append(sk)
            i = next_node_label # last element in msq is now part of next merge
            next_node_label += 1
    
    return Bt, msq


def check_merge_possibility(subgraph1: nx.Graph, subgraph2: nx.Graph, G: nx.Graph, merged_edges: list[tuple[int]]) -> Union[tuple, None]:
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
        ## To-Do: choose edge based on error rate and for star states based on number of center shifts
        return random.choice(valid_edges)
    else:
        return None



def find_sink_and_longest_leaf(G: nx.DiGraph) -> tuple[int, tuple[int, int]]:
    """Finding the sink node (node with out-degree 0 and in-degree > 0) and the longest leaf node in a directed graph.

    Args:
        G: A directed graph (nx.DiGraph).

    Returns:
        tuple: The sink node and the longest leaf node (with the longest distance from the sink node).
    """
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

def find_merging_tree(G: nx.Graph, msq_in: list[nx.Graph]) -> tuple[nx.DiGraph, list[nx.Graph]]:
    """
    Find merging stars and track the merging process using a directed graph (DiGraph).

    Parameters:
    G (networkx.Graph): The main graph to check merge possibilities.
    msq (list): A list of subgraphs (stars) to merge.

    Returns:
    tuple: merging tree graph (DiGraph) and the updated list of subgraphs (msq).
    """
    # Initialize variables
    #msq_original = msq[:]  
    msq = copy.deepcopy(msq_in) # Keep the original msq intact
    merged_stars = []  # List to track merged stars
    merged_edges = []  # List to track edges used for merging
    merge_associations = []  # Track which edge merges which stars

    # Initialize a directed graph (DiGraph)
    D = nx.DiGraph()
    D.add_nodes_from(range(len(msq_in)))  # Nodes are indices of msq elements

    # Condition for single step generation (msq has just one element)
    if len(msq_in) == 1:
        return D, msq

    # Loop until the DiGraph is strongly connected
    while not set(G.nodes) == set(msq[-1].nodes):        
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
                
                D.add_edge(i, new_node, weight=edge)
                D.add_edge(j, new_node, weight=edge)

                if i < j:
                    sk = mgo.merge_star_graphs(si, sj, edge, keep_center1=True, keep_center2=True)
                else:
                    sk = mgo.merge_star_graphs(sj, si, edge, keep_center1=True, keep_center2=True)
                # Add the new merged star to msq
                msq.append(sk)
                
            else:

                if edi == True:
                    if j != connected_node:
                        D.add_edge(j, connected_node, weight=edge)

                    sn = msq[connected_node]  # Get the subgraph at index connected_node

                    if j < connected_node:
                        sk = mgo.merge_star_graphs(sj, sn, edge, keep_center1=True, keep_center2=True)
                    else:
                        sk = mgo.merge_star_graphs(sn, sj, edge, keep_center1=True, keep_center2=True)
                    msq[connected_node] = sk
                    
                elif edj == True:
                    if i != connected_node:
                        D.add_edge(i, connected_node, weight=edge)

                    sn = msq[connected_node]  # Get the subgraph at index connected_node

                    if i < connected_node:
                        sk = mgo.merge_star_graphs(si, sn, edge, keep_center1=True, keep_center2=True)
                    else:
                        sk = mgo.merge_star_graphs(sn, si, edge, keep_center1=True, keep_center2=True)
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
            
        # draw_graph(D,node_color="pink")
       

    return D, msq


def parallel_merge(G: nx.Graph, msq_in: list[nx.Graph], show_status: bool = False) -> tuple[nx.DiGraph, list[nx.Graph]]:
    """Perform binary, parallel merging of stars in a graph.

    Args:
        G: Initial graph to merge stars from.
        msq: List of subgraphs (stars) to merge.
        show_status: Bool flag to show current status. Defaults to False.

    Raises:
        ValueError: If subgraph indices coincide.

    Returns:
        tuple: Binary tree graph (DiGraph) and the updated list of subgraphs (msq).
    """
    msq = copy.deepcopy(msq_in) # Keep the original msq intact
    # Create Binary tree graph Bt with nodes labeled 0, 1, ..., len(msq) - 1
    Bt = nx.DiGraph()
    Bt.add_nodes_from(range(len(msq)))

    # Condition for single step generation (msq has just one element)
    if len(msq) == 1:
        return Bt, msq
    
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
                    # Find a common edge to use for the merging
                    ## To-Do: choose edge based on error rate and for star states based on number of center shifts
                    common_edge = random.choice(common_edges) # random choice
                    # Common edge found
                    if show_status:
                        print(f"Common edge found between s{i} and s{j}: {common_edge}")

                    # Add a new node to Bt
                    Bt.add_node(next_node_label)
                    Bt.add_edge(i, next_node_label, weight=common_edge) # add merging edge info
                    Bt.add_edge(j, next_node_label, weight=common_edge) # add merging edge info
                    if show_status:
                        print(f"Added new node {next_node_label} to Bt, connecting s{i} and s{j} to it.")

                    # Merge si and sj into a new star graph sk
                    # use convention (smaller index, higher index)
                    if i < j:
                        if len(si) > 1:
                            sk = mgo.merge_star_graphs(si, sj, common_edge, keep_center1=True, keep_center2=True)
                        else:
                            sk = mgo.merge_star_graphs(sj, si, common_edge, keep_center1=True, keep_center2=True)
                        
                    elif j < i:
                        if len(sj) > 1:
                            sk = mgo.merge_star_graphs(sj, si, common_edge, keep_center1=True, keep_center2=True)
                        else:
                            sk = mgo.merge_star_graphs(si, sj, common_edge, keep_center1=True, keep_center2=True)
                    else:
                        raise ValueError("subgraph indices coincide!")

                    # Add sk to msq and increment the next available node label
                    msq.append(sk)
                    next_node_label += 1

                    # Mark si and sj as merged
                    merged_indices.add(i)
                    merged_indices.add(j)

                    break  # Exit the inner loop once a merge occurs
    
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
    layers = defaultdict(list)  # Init a dict with lists as valules to store nodes by layer
    queue = deque([(root, 0)])  # BFS queue with (node, depth)

    while queue:
        node, depth = queue.popleft()
        layers[depth].append(node)
        
        # Add children to the queue
        for child in tree.predecessors(node):
            queue.append((child, depth + 1))

    # Convert to a list of lists sorted by layer
    return [layers[layer] for layer in sorted(layers.keys())]

def check_binary_tree(bt: nx.DiGraph) -> bool:
    """This function checks a directed (inverted) tree graph if it's a binary inverted tree, i.e., if all nodes have at most two predecessors.

    Args:
        bt: Inverted tree graph to be checked.

    Returns:
        Bool flag that is True if the tree graph is a binary tree
    """
    is_binary = True
    # binary inverted tree is a tree which has at most two children

    for node in bt:
        if len(list(bt.predecessors(node))) > 2:
            is_binary = False

    return is_binary

class MergePattern:
    def __init__(self,
                 init_graph: nx.Graph,
                 msq: list[nx.Graph],
                 bt: nx.DiGraph):
        self._initial_graph = init_graph
        if len(msq) != len(bt):
            raise ValueError("length of subgraph list does not equal the number of nodes in the merging tree!")
        self._subgraphs = msq
        self._pattern_graph = bt
        self._is_binary = check_binary_tree(bt)
        #self._construct_subgraphs_of_layers(msq, bt)
        # get nodes sorted according to the binary tree layers; reverse list because bt is a inverted binary tree (root at highest layer)
        self._merge_nodes_by_layer = get_nodes_by_layers(self._pattern_graph)
        self._merge_nodes_by_layer.reverse()
        self._construct_siblings_by_layer()

    @classmethod
    def create_msq_and_merge_tree(cls, init_graph: nx.Graph, substate_size_fac: Union[float, None] = None, substate_size: Union[int, None] = None, parallel: bool = True, binary_merge: bool = False) -> tuple[list[nx.Graph], nx.DiGraph, float, int]:
        """Creates the subgraph list msq and the corresponding merging tree for creating a GHZ state from the given initial graph.

        Args:
            init_graph: Initial graph to create a GHZ state from
            substate_size_fac: Percentage factor between the target center node degree of each subgraph (star graphs) and the average degree in the initial graph. If this is not None this factor is used for the subgraph creation. Defaults to None, which means the algorithm picks always the highest degree node if substate_size is also None.
            substate_size: Target size of each initial subgraph that is created in msq list. Defaults to None.
            parallel: Bool flag for parallel merging. Defaults to True. If false then the merging is done sequentially.
            binary_merge: Bool flag if parallel merge should be performed according to a binary merge tree or a non-binary merge tree. Defaults to False. Note that if parallel flag is False, the binary merge flag has to be true because sequential merge is always binary.

        Raises:
            ValueError: If substate_size_fac and substate_size are both not None. Only one should be given or both should be None.
            ValueError: If sequential merging is chosen (parallel flag is False) and binary_merge flag is False. Sequential merge is always binary!

        Returns:
            subgraph list msq, merge tree, substate size factor, substate size
        """
        if substate_size_fac:
            if substate_size:
                raise ValueError("Substate size and substate size factor are both not None. Use just one of them and set the other to None.")
        else:
            if substate_size:
                # if substate size is given calculate the corresponding scaling factor
                target_degree = substate_size - 1 # for star graphs the center node degree is the size - 1
                substate_size_fac = compute_scaling_factor(init_graph, target_degree)

        if substate_size_fac is None:
            # if no substate_size is given, choose the highest degree star graphs
            _,msq,_ = calculate_msq(init_graph, show_status=False)
            substate_size = None
        else:
            # target_center_degree = substate_size - 1 # subgraphs are star graphs so the size is center node degree + 1
            # scaling_factor = compute_scaling_factor(init_graph, target_center_degree)
            msq, target_degree = calculate_msq_avg_degree(init_graph, scaling_factor=substate_size_fac, show_status=False)
            substate_size = target_degree + 1 # target degree of star graph center is size - 1!

        if not parallel and not binary_merge:
            raise ValueError("If sequential merging is chosen, the binary merge flag must be true!")
        
        if parallel:
            if binary_merge:
                bt, msq = parallel_merge(init_graph, msq)
            else:
                bt, msq = find_merging_tree(init_graph, msq)
        else:
            bt, msq = sequential_merge(init_graph,msq)

        return msq, bt, substate_size_fac, substate_size

    @classmethod
    def from_graph_sequential(cls, init_graph: nx.Graph, substate_size_fac: Union[float, None] = None):
        msq, bt, _, _ = cls.create_msq_and_merge_tree(init_graph, substate_size_fac=substate_size_fac, parallel=False, binary_merge=True)
        return cls(init_graph, msq, bt)
    
    @classmethod
    def from_graph_parallel(cls, init_graph: nx.Graph, binary_merge: bool = True, substate_size_fac: Union[float, None] = None):
        msq, bt, _, _ = cls.create_msq_and_merge_tree(init_graph, substate_size_fac=substate_size_fac, parallel=True, binary_merge=binary_merge)
        return cls(init_graph, msq, bt)


    @property
    def initial_graph(self):
        return self._initial_graph
    @property
    def pattern_graph(self):
        return self._pattern_graph
    @pattern_graph.setter
    def pattern_graph(self, bt: nx.DiGraph):
        self._pattern_graph = bt
        self._is_binary = check_binary_tree(bt)
        self._merge_nodes_by_layer = get_nodes_by_layers(self._pattern_graph)
        self._construct_siblings_by_layer()
        #self._construct_subgraphs_of_layers(self.subgraphs, self._pattern_graph)

    @property
    def subgraphs(self):
        return self._subgraphs
    
    @property
    def is_binary(self):
        return self._is_binary

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

    def get_initial_subgraphs(self) -> list[nx.Graph]:
        pattern_graph_leafs = [node for node in self.pattern_graph.nodes if self.pattern_graph.in_degree(node) == 0]

        subgraphs = []
        for idx in pattern_graph_leafs:
            subgraphs.append(self._subgraphs[idx])

        return subgraphs

    def get_merge_sets(self, layer: int) -> list[tuple[list[nx.Graph], list[tuple[int, int]]]]:
        """Creates a list of tuples containing the subgraphs that should be merged in the layer index and the merging edges.

        Args:
            layer: The layer index for which to retrieve the merge sets.

        Raises:
            ValueError: If the merging edge cannot be retrieved from the pattern graph.

        Returns:
            A list of tuples containing the subgraphs to merge and the merging edges
        """
        merge_sets = []

        # handle trivial merge pattern
        if len(self._merge_siblings_by_layer[layer]) == 0:
            return merge_sets

        # list of all subgraphs
        subgraphs = self._subgraphs
        for idcs in self._merge_siblings_by_layer[layer]:
            parent_idx = next(self._pattern_graph.successors(idcs[0]))
            
            # follow convention that the subgraph with the smallest index comes first (relevant for merging order, i.e., for graph states which center is kept)
            # if this convention is changed also the convention in find_merging_tree, parallel_merge and sequential merge function must be changed
            curr_merge_set_graphs = [subgraphs[idx] for idx in sorted(idcs)]
            curr_merge_edges = []
            for idx in sorted(idcs):
                curr_edge = self._pattern_graph[idx][parent_idx].get("weight", None)
                if not curr_edge:
                    raise ValueError("Unable to retrieve merging edge at pattern graph indices ({}, {})!".format(idx, parent_idx))
                curr_merge_edges.append(curr_edge)
            
            merge_sets.append((curr_merge_set_graphs, curr_merge_edges))

        return merge_sets

    def find_subgraph(self, subgraph: nx.Graph) -> Union[None, int]:
        """Tries to find the given subgraph in the subgraph list of this MergePattern instance. 
        It there for compares the nodes and edges of the input subgraph with each subgraph in the MergePattern instance.

        Args:
            subgraph: Subgraph that is searched for in the subgraph list of the MergePattern instance

        Returns:
            If the subgraph is found it returns the index of the matching subgraph in the subgraph list of the MergePattern instance. 
            If the subgraph is not found it returns None.
        """
        graph_idx = None
        for idx, graph in enumerate(self._subgraphs):
            curr_nodes_match = nx.utils.nodes_equal(graph.nodes, subgraph.nodes)
            curr_edges_match = nx.utils.edges_equal(graph.edges, subgraph.edges)
            if curr_nodes_match and curr_edges_match:
                graph_idx = idx
                break
        return graph_idx

    
    def draw_subgraphs(self, 
                       layer: Union[int, None] = None,
                       **kwargs):
        """
        Draw the subgraphs (of a certain layer) using draw_graph function from modify_graph_objects
        """
        # Default values
        if kwargs.get("node_color", None) is None:
            kwargs["node_color"] = "yellow"
        if kwargs.get("layout", None) is None:
            kwargs["layout"] = "circular"
        show_plots_at_end = False
        if kwargs.get("show", None) is None:
            show_plots_at_end = True
            kwargs["show"] = False
        if kwargs.get("fig_size", None) is None:
            kwargs["fig_size"] = (8, 8)
        if kwargs.get("title", None) is None:
            kwargs["title"] = "Graph Visualization"


        if layer is None:
            subgraphs = self._subgraphs
            for i in range(len(subgraphs)):
                if i == len(subgraphs)-1:
                    if show_plots_at_end:
                        kwargs["show"] = True
                graph = subgraphs[i]
                mgo.draw_graph(graph, **kwargs)
        elif isinstance(layer, int):
            for i in self._merge_nodes_by_layer[layer]:
                if i == self._merge_nodes_by_layer[layer][-1]:
                    if show_plots_at_end:
                        kwargs["show"] = True
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
        if kwargs.get("layout", None) is None:
            kwargs["layout"] = "graphviz_dot"
        if kwargs.get("node_color", None) is None:
            kwargs["node_color"] = "lightblue"
        if kwargs.get("show", None) is None:
            kwargs["show"] = True
        if kwargs.get("fig_size", None) is None:
            kwargs["fig_size"] = (8, 6)
        if kwargs.get("title", None) is None:
            kwargs["title"] = "Binary Tree Representation of merging pattern"
        if kwargs.get("edge_color", None) is None:
            kwargs["edge_color"] = "gray"

        mgo.draw_graph(self._pattern_graph, **kwargs)
        

if __name__ == "__main__":
    #main()
    # List of nodes
    nodes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Edge list (each tuple represents an edge between two nodes)
    edges_list = [(0, 1), (0, 4), (1, 6), (2, 3), (3, 7), (4, 9), (5, 6), (6, 8), (7, 9), (8, 9), (0, 10), (10,11),(11,12)]
    
    init_graph = generate_graph(nodes_list, edges_list, use_barabasi=False)
    draw_graph(init_graph, show=False)
    # calculate_gate_ss(generate_ibm_graph(nodes_list, edges_list, use_barabasi=False))
    #_,msq,_= calculate_msq(generate_random_graph(10, 0.1, use_barabasi=False))
    _,msq1,_= calculate_msq(init_graph, show_status=False)
    # for i in range(len(msq)):   
    #   draw_graph(msq[i], show=False)
    print(f"number of subgraphs is initially {len(msq1)}.")
    
    bt1,_ = parallel_merge(init_graph,msq1)
    
    #draw_binary_tree(bt1, show=False)
    # Define MergePattern instance for parallel merge
    ptrn1 = MergePattern(init_graph, msq1, bt1)

    _,msq2,_= calculate_msq(init_graph, show_status=False)
    # for i in range(len(msq)):   
    #   draw_graph(msq[i], show=False)
    print(f"number of subgraphs is initially {len(msq2)}.")
    
    bt2,_ = sequential_merge(init_graph,msq2)

    #draw_binary_tree(bt2, show=True)
    # Define MergePattern instance for sequential merge
    ptrn2 = MergePattern(init_graph, msq2, bt2)

    # Define MergePattern instance for non-binary parallel merge
    ptrn3 = MergePattern.from_graph_parallel(init_graph, binary_merge=False)

    # print all merging pairs
    print("merge pairs for parallel merge:")
    for layer in range(len(ptrn1)):
        curr_pairs_graph = ptrn1.get_merge_pairs(layer)
        curr_pairs = ptrn1._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge pairs {curr_pairs}")
        for graph_pair_idcs, graph_pair in zip(curr_pairs,curr_pairs_graph):
            print(f"    graph {graph_pair_idcs[0]}: nodes={graph_pair[0].nodes}")
            print(f"    graph {graph_pair_idcs[1]}: nodes={graph_pair[1].nodes}")
            print(f"    merging edge {graph_pair[2]}")
    print(f"merge tree is binary {ptrn1.is_binary}")
    print()
    print("merge pairs for sequential merge:")
    for layer in range(len(ptrn2)):
        curr_pairs_graph = ptrn2.get_merge_pairs(layer)
        curr_pairs = ptrn2._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge pairs {curr_pairs}")
        for graph_pair_idcs, graph_pair in zip(curr_pairs,curr_pairs_graph):
            print(f"    graph {graph_pair_idcs[0]}: nodes={graph_pair[0].nodes}")
            print(f"    graph {graph_pair_idcs[1]}: nodes={graph_pair[1].nodes}")
            print(f"    merging edge {graph_pair[2]}")
    print(f"merge tree is binary {ptrn2.is_binary}")
    print()
    print("merge sets for parallel merge:")
    for layer in range(len(ptrn3)):
        curr_merge_set = ptrn3.get_merge_sets(layer)
        curr_set = ptrn3._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge sets {curr_set}")
        for graph_set_idcs, graph_set in zip(curr_set,curr_merge_set):
            if len(graph_set_idcs) != len(graph_set[0]):
                raise ValueError("graph set indices and graph set do not have the same length!")
            print(f"    merging set {graph_set_idcs}:")
            for graph_idx, graph, edge in zip(graph_set_idcs, graph_set[0], graph_set[1]):
                print(f"    graph {graph_idx}: nodes={graph.nodes()}, merging edge {edge}")
            print()
    print(f"merge tree is binary {ptrn3.is_binary}")

    # draw the binary tree representing the merge pattern
    ptrn1.draw_pattern_graph(show_weights=True, show=False)
    ptrn2.draw_pattern_graph(show_weights=True, show=False)
    ptrn3.draw_pattern_graph(show_weights=True, show=False)

    # draw all subgraphs 
    #ptrn1.draw_subgraphs(show=False)
    #ptrn2.draw_subgraphs()

    # check if the nodes in the last subgraph is the same as the nodes in the initial graph
    print("initial graph nodes: ", init_graph.nodes())
    print("last subgraph nodes in parallel merge: ", sorted(ptrn1[-1][0].nodes()))
    print("last subgraph nodes in sequential merge: ", sorted(ptrn2[-1][0].nodes()))
    print("last subgraph nodes in non-bin. parallel merge: ", sorted(ptrn3[-1][0].nodes()))
    print("initial graph nodes == last subgraph nodes in parallel merge: ", init_graph.nodes() == ptrn1[-1][0].nodes())
    print("initial graph nodes == last subgraph nodes in sequential merge: ", init_graph.nodes() == ptrn2[-1][0].nodes())
    print("initial graph nodes == last subgraph nodes in parallel merge: ", init_graph.nodes() == ptrn3[-1][0].nodes())

    ptrn1.draw_subgraphs(layer=-1, show=False)
    ptrn2.draw_subgraphs(layer=-1, show=False)
    ptrn3.draw_subgraphs(layer=-1)

    # print merging sets
    print("merge sets for parallel merge:")
    for layer in range(len(ptrn1)):
        curr_merge_set = ptrn1.get_merge_sets(layer)
        curr_set = ptrn1._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge sets {curr_set}")
        for graph_set_idcs, graph_set in zip(curr_set,curr_merge_set):
            if len(graph_set_idcs) != len(graph_set[0]):
                raise ValueError("graph set indices and graph set do not have the same length!")
            print(f"    merging set {graph_set_idcs}:")
            for graph_idx, graph, edge in zip(graph_set_idcs, graph_set[0], graph_set[1]):
                print(f"    graph {graph_idx}: nodes={graph.nodes()}, merging edge {edge}")
            print()
    print()
    print("merge sets for sequential merge:")
    for layer in range(len(ptrn2)):
        curr_merge_set = ptrn2.get_merge_sets(layer)
        curr_set = ptrn2._merge_siblings_by_layer[layer]
        print(f"layer {layer}: merge sets {curr_set}")
        for graph_set_idcs, graph_set in zip(curr_set,curr_merge_set):
            if len(graph_set_idcs) != len(graph_set[0]):
                raise ValueError("graph set indices and graph set do not have the same length!")
            print(f"    merging set {graph_set_idcs}:")
            for graph_idx, graph, edge in zip(graph_set_idcs, graph_set[0], graph_set[1]):
                print(f"    graph {graph_idx}: nodes={graph.nodes()}, merging edge {edge}")
            print()
    
    ## Test for handeling fully connected graphs
    # edges = [(0,1), (0,2), (0, 3), (1,2), (1,3), (2,3)]
    # graph = nx.Graph(edges)
    # mgo.draw_graph(graph, show=False)
# 
    # _, msq, _ = calculate_msq(graph, show_status=False)
    # #msq, _ = calculate_msq_avg_degree(graph, 1.0, show_status=False)
    # #bt, msq = sequential_merge(graph, msq)
    # #bt, msq = parallel_merge(graph, msq)
    # bt, msq = find_merging_tree(graph, msq)
# 
    # print(f"length of msq: {len(msq)}")
    # if len(msq) >= 1:
    #     mgo.draw_graph(msq[0], title="first element of msq", show = False)
# 
    # mgo.draw_graph(bt, layout="graphviz_dot", show=False)
# 
    # pattern = MergePattern(graph, msq, bt)
# 
    # print(f"pattern length: {len(pattern)}")
# 
    # subgraphs = pattern.get_initial_subgraphs()
    # print(f"length pattern subgraphs: {len(subgraphs)}")
    # if len(subgraphs) >= 1:
    #     mgo.draw_graph(subgraphs[0], title="first element of pattern subgraphs", show = True)
# 
    # # check if this works
    # last_subgraph = pattern.subgraphs[-1]
# 
# 
    # merge_sets = pattern.get_merge_sets(0)
    # print(f"merge_sets: {merge_sets}")




