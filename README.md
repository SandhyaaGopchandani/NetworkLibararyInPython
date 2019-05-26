# Network Libarary In Python

This repository contains a network library (nets.py) for handling undirected and directed graphs. 

The library contains following methods:

- read_edgelist , write_edgelist: a method to read from and write into network edge list format
- read_adjlist , write_adjlist: a method to read from and write into network adjacency list format
- read_adjmat , write_adjmat: a method to read from and write into network adjacency matrix format

- nodes , edges: methods to return a list of all nodes, and a list of all edges
- add_node: method to insert a new node into the graph
- add_nodes_from: insert multiple nodes from a list

- remove_node , remove_nodes_from: remove a single node, multiple nodes from a list

- add_edge, add_edges_from: insert new edge into graph, creating nodes as needed, insert multiple edges from list of node tuples

- remove_edge , remove_edges_from: likewise, remove edge(s)

- has_node , has_edge: return True or False if node (edge) is present or not

- neighbors: given a node as input, return set of neighbors of node

- number_of_nodes , number_of_edges: get counts of nodes and edge currently in graph

- degree: return dict mapping node to number of neighbors, for all nodes or an optional list of nodes

- clustering_coefficient: return the clustering coefﬁcient(s) of one or more nodes

- subgraph: return a new graph from a given list of nodes, consisting of those nodes and the links between those nodes in the original graph.

- size_gcc: fraction of nodes within the giant (largest) connected component of the graph

- info - a nicely formatted summary “printout” of the state of the graph, built from the other methods