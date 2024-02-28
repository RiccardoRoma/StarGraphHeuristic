#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:49:50 2024

@author: siddhu

The below code is to produce random graphs according to Erdos-Renyi and Barabasi Albert models and save them.
One can also uncomment the last part of the code, import the saved graphs and plot them. 
Written for the project with prof. Althaus in Mainz
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations, groupby
import pickle
import os

def generate_random_graph(n, p, use_barabasi):
    if use_barabasi:
        G = nx.barabasi_albert_graph(n, p, seed=None, initial_graph=None)
    else:
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
            G.add_edge(*random_edge, weight=1)
            for e in node_edges:
                if random.random() < p:
                    G.add_edge(*e, weight=1)
    return G

def save_random_graph(graph, n, p, use_barabasi, save_path='/Users/thaslimjaglurbasheer/Documents/Siddhu/graphs_generation/Saved_random_graphs'):
    graph_type = "barabasi" if use_barabasi else "erdos_renyi"
    filename = os.path.join(save_path, f'random_graph_n{n}_p{p:.1f}_{graph_type}.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)

# Generate and save random graphs
use_barabasi=False
save_directory = '/Users/thaslimjaglurbasheer/Documents/Siddhu/graphs_generation/Saved_random_graphs'
for n in range(10, 1001, 10):
    for p in [0.1 * i for i in range(1, 11)]: # Change this for Barabasi, p is m
            random_graph = generate_random_graph(n, p, use_barabasi)
            save_random_graph(random_graph, n, p, use_barabasi, save_directory)

# Read the saved file and plot the graph

# def read_random_graph(filename):
#     with open(filename, 'rb') as file:
#         loaded_graph = pickle.load(file)
#     return loaded_graph

# # Specify the filename of the saved graph you want to read
# filename_to_read = '/Users/thaslimjaglurbasheer/Documents/Siddhu/graphs_generation/Saved_random_graphs/random_graph_n20_p0.4_erdos_renyi.pkl'  # Adjust the filename as needed

# # Read the saved graph
# loaded_graph = read_random_graph(filename_to_read)

# # Plot the loaded graph
# nx.draw(loaded_graph, with_labels=True, font_weight='bold')
# plt.show()