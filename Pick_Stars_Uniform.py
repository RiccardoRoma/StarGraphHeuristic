#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 18, 2025 15:25:38 2024

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


def compute_scaling_factor(G, target):
    """
    Compute the scaling factor given a graph G and a target degree.

    Parameters:
        G (networkx.Graph): The input graph.
        target (int): The desired target degree.

    Returns:
        float: The computed scaling factor.
    """
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    if avg_degree == 0:
        return 0.0  # Avoid division by zero

    scaling_factor = target / avg_degree
    return (scaling_factor, 1.0)


def calculate_msq(G, scaling_factor: float = 1.0, show_status: bool = True):
    """
    Calculates the gates using the SS method on the given graph G.
    Modified to pick stars of (roughly) consistent size based on the average degree.
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
    init_graph = generate_random_graph(20, 0.2, use_barabasi=False)
    # init_graph = generate_graph(nodes_list, edges_list, use_barabasi=False)
    draw_graph(init_graph, show=True)
    # calculate_gate_ss(generate_ibm_graph(nodes_list, edges_list, use_barabasi=False))
    # _,msq,_= calculate_msq(generate_random_graph(10, 0.1, use_barabasi=False))
    msq1, target_size = calculate_msq(init_graph, 1.0, show_status=False)

    for i in range(len(msq1)):
        draw_graph(msq1[i], node_color="pink", show=True)
