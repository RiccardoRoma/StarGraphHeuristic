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
from collections import defaultdict, deque
from networkx.drawing.layout import circular_layout
from itertools import combinations, groupby
import math
import csv
import concurrent.futures 
import time


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




# def generate_ibm_graph(nodes_list, edges_list, use_barabasi):
    
#         # Create a graph G
#         G = nx.Graph()
        
#         # Add nodes to the graph
#         G.add_nodes_from(nodes_list)
        
#         # Add edges to the graph
#         G.add_edges_from(edges_list)
        
#         plt.figure()
#         nx.draw(G, node_color='lightgreen', 
#                 with_labels=True, 
#                 node_size=500)
#         return G

def draw_graph(graph, node_color='yellow', layout="circular"):
    """
    Draw the given graph using NetworkX and Matplotlib.
    """
    if layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph)  # Default to spring layout if layout is not circular

    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos, with_labels=True, node_color=node_color, node_size=500, font_size=10, font_color='black', font_weight='bold')
    plt.title("Graph Visualization")
    plt.show()



MS=[] # a list to store all small stars and merging sequence.  
Picked_Stars=[]
MSQ=[]
MSG=[]
merging_edges_list=[]

def calculate_msq(G):
    """
    Calculates the gates using the SS method on the given graph G.
    """
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
            print("I am here")
            plt.figure(l + 100)
            nx.draw(MG[l], node_color='lightblue', 
                    with_labels=True, 
                    node_size=500)
            for node in MG[l].nodes():
                SG[l + 1] = nx.Graph()
                SG[l + 1].add_node(node)
                plt.figure(l + 150)
                nx.draw(SG[l + 1], node_color='lightpink', 
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
            plt.figure(l+50)
            # if l==0:
            #     MS.append(SG[l].copy())
            nx.draw(SG[l], node_color='lightpink', 
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
    print("SG is",SG)
    
    

        
    first_time=True        
    
    while len(SG) > 1:
        common_edge_found = False
        common_edge_index = 0
        
        
        for i in SG.keys():
            if i == 0:
                continue
            common_edges = [e for e in G.edges if set(e).intersection(SG[0].nodes()) and set(e).intersection(SG[i].nodes())]
            print(common_edges)
            print("I am here and loop number is ",i)
            
                              
            if len(common_edges)!=0:
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
                print("common edge is", common_edge)
                
            
                # Assign new star center nodes to each each graph
                if SG[0].has_node(common_edge[0]):
                    cn_0 = common_edge[0]
                    cn_i = common_edge[1]
                    print("common node of cn_i is", cn_i)
        
                else:
                    cn_0 = common_edge[1]
                    cn_i = common_edge[0]
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
                    draw_graph(SG[0])
                    
            
                if cn_i == max(dict(SG[i].degree()).items(), key=lambda x: x[1])[0] or len(SG[i].nodes()) == 2:
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
                    draw_graph(SG[i])
                    
                
                # Merging the two stars                   
                for node in SG[i].nodes():
                    if node != cn_0:
                        SG[0].add_edge(cn_0, node, weight=1)  # Adding edge weight of 1
                
                sid_total_gates += 1 # For merging (one CNOT gate)
                draw_graph(SG[0])
                
                # Deleting the MSG[i] star as it is merged to MSG[0]
                # SG.remove(SG[i])
        
                break
                
                # print("SS gates are ", sid_total_gates)
        if common_edge_found:
            del SG[common_edge_index]  # Remove the subgraph at index i
            
          
        else:
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


def main():
    # List of nodes
    nodes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Edge list (each tuple represents an edge between two nodes)
    edges_list = [(0, 4), (1, 6), (2, 3), (3, 7), (4, 9), (5, 6), (6, 8), (7, 9), (8, 9)]
    
    # calculate_gate_ss(generate_ibm_graph(nodes_list, edges_list, use_barabasi=False))
    calculate_msq(generate_random_graph(10, 0.1, use_barabasi=False))
    # for i in range(len(MS)):   
    #     draw_graph(MS[i])

if __name__ == "__main__":
    main()


