# Star Heuristic graph
## modules (heuristic setup)

### qiskit_input_graph
**generate_random_graph (n, p, use_barabasi)** : generates a graph using the Barabasi-Albert model or Erdos-Renyi model \
n :  number of nodes \
p : number of edges to attach from a new node to existing nodes \
use_barabasi : uses Barabasi-Albert model \
returns graph G \
\
**draw_graph (graph, node_color, layout)**  : Draws the graph \
graph : input graph \
node_color : color for the nodes (choose any, yellow by default) \
layout : circular or spring \
\
**calculate_msq (G, show_status)** : calculates the gates using the SS method on the given graph or calculates the list of subgraphs in sequential merging order (MSQ) and at what edges one should merge the graphs (merging_edges_list) \
G : input \
show_status : boolean True/False \
returns merging_edges_list 

### modify_graph_objects
**generate_ibm_graph (node_list, edge_list)** : draws a graph from edge list and node list.\
node_list : List of all the nodes \
edge_list: list of all the edges \
return figure of  a graph \
\
**draw_graph( graph, node_color, layout)** : Draw the given graph using NetworkX and Matplotlib \
return figure of  a graph \
\
**update_graph_center ( graph, new_center)**  : shifts the center of the graph\
returns new graph \
\
**merging_two_graphs (G1, G2, t)** : merges two graphs G1 and G2 and new center will always be G1 center \
G1, G2 : input graphs \
t : tuple of the centers (C1, C2) \
returns the new merged graph \
\
**get_graph_center (graph)** : gives the center of the graph G, if it is a Bell pair, it would randomly give you one node \
returns the center of a graph 

### save_random_graphs
produces random graphs according to Erdos-Renyi and Barabasi Albert models and save them.\
\
**generate_random_graph (n, p, use_barabasi)** : generates a graph using the Barabasi-Albert model or Erdos-Renyi model \
n :  number of nodes \
p : number of edges to attach from a new node to existing nodes \
use_barabasi : uses Barabasi-Albert model \
returns graph G \
\
**save_random_graph ( graph, n, p, use_barabasi, copy_number, save_path)** : used to save all the random graphs produced in the last function \
\
**read_random_graph (n , p, copy_number, use_barabasi, save_path)**: read the saved random graphs and plot them. 

## modules for quantum circuit

### generate_star_states

**get_leaf_qubits_from_edges (edges_in, star_centers)**: returns a list with all the leaf qubits in it \
edges_in :  list of all the edges \
star_center :  center of a star \
returns the list of leaf qubits 

**generate_star_state (graph, circ)** : returns quantum circuit for a graph \
graph : target graph \
circ : empty quantum circuit with number of qubits as the number of nodes in the initial graph \
return a quantum circuit for a graph 

### shift_center
**shift_center ( circ, graph, center_in, center_fin)** : shifts the center of  a graph ( in the quantum circuit version ) 

### merge_graphs
**merge_graphs (circ, G1, G2, cls_bit_cnt, edge_list_G1)** : merges the graphs G1 and G2 with centers C1 and C2 respectively. \
circ : input quantum circuit \
G1, G2 : graphs (subgraphs) G1 and G2 \
cls_bit_cnt: index of the classical register used \
edge_list_G1: list of all the edges of graph (subgraph) G1 

### create_ghz_state_circuit
**create_ghz_state_circuit ( graph_file)** : returns the final GHZ state for given graph.


## Steps for GHZ state circuit:
- load initial graph from pickle file
- get merging sequence with calculate_msq function
- iterate through the list of merging sequence
- iterations: generate star state for each subgraph in the merging sequence list
- in each iteration, two graphs are center shifted and then merged
- shift center of first subgraph, shift center of second subgraph, and then merge them
- compose the circuit

