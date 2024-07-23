#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:28:24 2024

@author: siddhu
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from collections import defaultdict, deque
from networkx.drawing.layout import circular_layout
from itertools import combinations, groupby
import math
import csv
import concurrent.futures 
import time

# def generate_ibm_graph(nodes_list, edges_list, use_barabasi):
#     
#         # Create a graph G
#         G = nx.Graph()
#         
#         # Add nodes to the graph
#         G.add_nodes_from(nodes_list)
#         
#         # Add edges to the graph
#         G.add_edges_from(edges_list)
#         
#         plt.figure()
#         nx.draw(G, node_color='lightgreen', 
#                 with_labels=True, 
#                 node_size=500)
#         return G

def draw_graph(graph, show_weights=False, node_color='yellow', layout="circular", title="", fname=""):
    """
    Draw the given graph using NetworkX and Matplotlib.
    """
    if layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "planar":
        pos = nx.planar_layout(graph)
    elif layout == "multipartite":
        pos = nx.multipartite_layout(graph)
    elif layout == "graphviz_dot":
        pos = nx.nx_agraph.pygraphviz_layout(graph, prog="dot")
    else:
        print("Unknown layout string, use spring layout")
        pos = nx.spring_layout(graph)  # Default to spring layout if layout does not match any implemented layout

    plt.figure(figsize=(16, 9))
    if show_weights:
        nx.draw(graph, pos, with_labels=False, node_color=node_color, node_size=500, font_size=10, font_color='black', font_weight='bold')
        # Draw node labels (including weights)
        node_labels = {node: f"{node}\n({data.get('weight', 0.0)})" for node, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=node_labels)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(graph, 'weight', default=0.0)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    else:
        nx.draw(graph, pos, with_labels=True, node_color=node_color, node_size=500, font_size=10, font_color='black', font_weight='bold')


    if len(title) == 0:
        plt.title("Graph Visualization")
    else:
        plt.title(title)
    if len(fname) > 0:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def update_graph_center(G,new_center):
    """
    shifts the center of the the graph
    """
    G.remove_edges_from(G.edges())
    for node in G.nodes():
        if node != new_center:
            G.add_edge(new_center, node, weight=1) # Adding edge weight of 1
    return G
            
def merging_two_graphs(G1,G2,t:tuple):
    """
    merges two graphs G1 and G2 and new center will ALWAYS be G1_center
    """
    G3 = G1.copy()
    for node in G2.nodes():
        if node != t[0]:
            G3.add_edge(t[0], node, weight=1)  # Adding edge weight of 1      
    return G3

def get_graph_center(G):
    """
    Gives the center of the graph G, if it is a bell pair, it would randomly give you one node. 
    """   
    degrees = dict(G.degree())
    
    if not degrees:
        return None  # Return None for empty graph
    
    max_degree_node = max(degrees, key=degrees.get)
    max_degree = degrees[max_degree_node]
    
    return max_degree_node
    
    


# # nodes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# #     # Edge list (each tuple represents an edge between two nodes)
# # edges_list = [(0, 4), (1, 6), (2, 3), (3, 7), (4, 9), (5, 6), (6, 8), (7, 9), (8, 9)]

# nodes_list = [1, 2, 3, 4]

# edges_list=[(1,2)]
   
# edges_list = [(1, 2), (1, 3), (1, 4)]

# nodes_list1 = [5, 6, 7, 8]
   
# edges_list1 = [(5, 6), (5, 7), (5, 8)]

# G1=generate_ibm_graph(nodes_list, edges_list, use_barabasi=False)
# ab=get_graph_center(G1)
# G2=generate_ibm_graph(nodes_list1, edges_list1, use_barabasi=False)
# update_center_graph(G1, 3)
# draw_graph(G1)
# t=(3,5)
# G4=merging_two_graphs(G1, G2, t)
# draw_graph(G4,"pink")


      