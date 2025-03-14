#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:28:24 2024

@author: siddhu
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy
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

def draw_graph(graph: nx.Graph, show_weights: bool=False, fig_size: tuple[int, int]=(16,9), node_color: str='yellow', layout: str="circular", title: str="", fname: str="", show: bool=True, **kwargs):
    """
    Draw the given graph using NetworkX and Matplotlib.
    """
    # Default values 
    if kwargs.get("node_size", None) is None:
        kwargs["node_size"] = 500
    if kwargs.get("font_size", None) is None:
        kwargs["font_size"] = 10
    if kwargs.get("font_color", None) is None:
        kwargs["font_color"] = "black"
    if kwargs.get("font_weight", None) is None:
        kwargs["font_weight"] = "bold"
    


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

    plt.figure(figsize=fig_size)
    if show_weights:
        # handle show node label keyword
        if kwargs.get("with_labels", None) is not None:
            show_node_labels = kwargs.pop("with_labels")
        else:
            show_node_labels = True

        #nx.draw(graph, pos, with_labels=False, node_color=node_color, node_size=500, font_size=10, font_color='black', font_weight='bold')
        nx.draw(graph, pos, node_color=node_color, with_labels=False, **kwargs)
        # Draw node labels (including weights)
        if show_node_labels:
            node_labels = {node: f"{node}\n({data.get('weight', 0.0)})" for node, data in graph.nodes(data=True)}
        else:
            node_labels = {node: f"({data.get('weight', 0.0)})" for node, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=node_labels)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(graph, 'weight', default=0.0)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    else:
        if kwargs.get("with_labels", None) is None:
            kwargs["with_labels"] = True
        #nx.draw(graph, pos, with_labels=True, node_color=node_color, node_size=500, font_size=10, font_color='black', font_weight='bold')
        nx.draw(graph, pos, node_color=node_color, **kwargs)


    if len(title) == 0:
        plt.title("Graph Visualization")
    else:
        plt.title(title)
    if len(fname) > 0:
        plt.savefig(fname, bbox_inches="tight")

    if show:
        plt.show()


def update_graph_center(G: nx.Graph, new_center: int) -> nx.Graph:
    """
    shifts the center of the the graph
    """
    G.remove_edges_from(G.edges())
    for node in G.nodes():
        if node != new_center:
            G.add_edge(new_center, node)
    return G
            
def merging_two_graphs(G1: nx.Graph, G2: nx.Graph, t:tuple) -> nx.Graph:
    """
    merges two graphs G1 and G2 and new center will ALWAYS be G1_center
    """
    G3 = copy.deepcopy(G1)
    G3.add_nodes_from(G2.nodes())
    for node in G2.nodes():
        if node != t[0]:
            G3.add_edge(t[0], node)

    return G3

def merge_star_graphs(graph1_in: nx.Graph,
                      graph2_in: nx.Graph,
                      merging_edge: tuple[int, int],
                      keep_center1: bool = True,
                      keep_center2: bool = True) -> nx.Graph:
    """This function merges two star graphs into a single star graph.

    Args:
        graph1_in: First star graph to merge.
        graph2_in: Second star graph to merge.
        merging_edge: Edge to merge the two star graphs along.
        keep_center1: Bool flag to keep the center of the first star graph in the merged graph. Defaults to True.
        keep_center2: Bool flag to keep the center of the second star graph in the merged graph. Defaults to True.

    Raises:
        ValueError: If the merging edge does not contain nodes of the respective subgraphs.
        ValueError: If both centers are removed from the merged graph.

    Returns:
        Merged star graph.
    """
    # create copies which can be modified without changing the initial graphs
    graph1 = copy.deepcopy(graph1_in)
    graph2 = copy.deepcopy(graph2_in)

    # merging edges list tuples are not ordered after (graph1, graph2) but (smaller value, higher value). Consider this here
    if merging_edge[0] in graph1.nodes:
        new_center1 = merging_edge[0]
    elif merging_edge[1] in graph1.nodes:
        new_center1 = merging_edge[1]
    else:
        raise ValueError("merging edge {} don't contain a node of subgraph 1 {}".format(merging_edge, graph1.nodes))
    if merging_edge[0] in graph2.nodes:
        new_center2 = merging_edge[0]
    elif merging_edge[1] in graph2.nodes:
        new_center2 = merging_edge[1]
    else:
        raise ValueError("merging edge {} don't contain a node of subgraph 2 {}".format(merging_edge, graph2.nodes))
    
    # update center of graphs
    graph1 = update_graph_center(graph1, new_center1)
    graph2 = update_graph_center(graph2, new_center2)

    # merge graph obejects accordingly
    if keep_center1:
        graph_merged = merging_two_graphs(graph1, graph2, (new_center1, new_center2))
        if not keep_center2:
            graph_merged.remove_node(new_center2)
    elif keep_center2:
        graph_merged = merging_two_graphs(graph2, graph1, (new_center2, new_center1))
        if not keep_center1:
            graph_merged.remove_node(new_center1)
    else:
        raise ValueError("Cannot remove both initial graph centers. One needs to be kept.")

    return graph_merged

def get_graph_center(G: nx.Graph) -> int:
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


      