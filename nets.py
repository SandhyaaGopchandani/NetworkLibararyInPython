#!/usr/bin/env python

# nets.py
# Sandhya
# Last Modified: 03/28/2018

"""
This is a network tool for creation, manipulation of complex networks from different graph representation.
It features data structure for graphs and digraphs. It includes basic as well as extensive graph functionalities like clustering coefficient,
degrees, modularity and shortest path.
"""

import sys, os
import collections
import itertools
import csv, json


def is_number(value):
    """Return true if string can be converted to number."""
    try:
        float(value)
        return True
    except ValueError:
        return False
    
def write_edgelist(graph, filename, delimiter="\t"):
    """This method takes a graph and writes it to an empty file in edge list format"""
    with open(filename, 'w') as file:
        for key, value in graph.items():
            for item in value:
                file.write('{0} {1}\n'.format(key, item))

def read_edgelist(filename, delimiter=" ", nodetype=None):
    """This method reads graph info from filename in edge list format and stores in a dictionary 
    where key is node and value is the list of neighbors"""
    graph = {}
    with open(filename, "r") as f: 
        for line in f:
            line = line.rstrip('\n')
            line = line.split(delimiter)
            key, value = line[0], line[1]
            if(is_number(key) and is_number(value)):
                key = int(key)
                value = int(value)
            if key not in graph:
                graph[key] = [value]
            else:
                graph[key].append(value)
    return graph
    

def write_adjlist(graph, filename, delimiter="\t"):
    """This method takes a graph and writes it to an empty file in adjacency list format"""
    with open(filename, 'w') as file:
        for key, value in graph.items():
            value = ' '.join(str(e) for e in value)
            file.write('{0} {1}\n'.format(key, value))


def read_adjlist(filename, delimiter=" ", nodetype=None):
    """This method reads graph info from filename in adjacency list format and stores in a dictionary 
    where key is node and value is the list of neighbors"""
    
    graph = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.split(delimiter)
            key, value = line[0] , line[1:]
            if(is_number(key)):
                key = int(key)
                value = list(map(int, value))
            if key not in graph:
                graph[key] = value
            else:
                graph[key].append(value)
    return graph


def write_adjmat(graph, filename, delimiter="\t"):
    """This method takes a graph and writes it to an empty file in adjacency matrix format"""
    size = len(graph)
    matrix = []
    matrix = [[0] * size for i in range(size)]

    for key, value in graph.items():
        for item in value:
            matrix[key][item] = 1
    with open(filename, 'w') as file:
        for row in matrix:
            value = ' '.join(str(e) for e in row)
            file.write(value + '\n')
    #print(matrix)
    
def read_adjmat(filename, delimiter=" ", nodetype=None):
    """This method reads graph info from filename in adjacency matrix format and stores in a dictionary 
    where key is node and value is the list of neighbors"""
    graph = {}
    with open(filename, "r") as f:
        for num, line in enumerate(f):
            line = line.rstrip('\n')
            line = line.split(delimiter)
            line = list(map(int,line))
            for i, j in enumerate(line):
                if j == 1:
                    if num not in graph:
                        graph[num] = [i]
                    else:
                        graph[num].append(i)
    return graph

#read_edgelist("data/karate.edgelist", ' ')
#read_edgelist("data/NCAA_2005.edgelist_small", "\" \"")
#read_edgelist("data/polblogs.edgelist",' ')
#read_edgelist("data/NCAA_divI_confs.txt",'\t')
#read_adjlist("data/cond-mat.adjlist_small", ' ')
#read_adjlist("data/NCAA_divI_confs.txt", '\t')
#read_adjmat("data/small_graph.adjmat.txt",' ')
#write_adjlist(a,"data/write_adjlist.txt")
#write_edgelist(a,"data/write_edgelist.txt")
#write_adjmat(a,"data/write_adjMat.txt")
#read_adjlist("data/write_adjMat.txt", ' ')


class Node():
    """The Node class defines the node structure and operations that can be done at node level.
    The class separates the node from the graph and treats as a subcomponent. The graph class uses
    this class to perform node operation"""
    
    def __init__(self, id=None, neighbors=None):
        """This is a constructor of Node class. 
        Node has an id and set of neighbors. When id and neighbors not provided, it is set to None"""
        self.id = id
        if neighbors is None:
            self.neighbors = set()
        else:
            self.neighbors = neighbors


    def add_neighbor(self, other_node, dual):
        """This method adds other_node as neighbor to this class node. The dual check is for 
        directed and undirected graph. If dual is true, it adds this node as neighbor to other_node.
        """
        self.neighbors.add(other_node)
        if dual:
            other_node.neighbors.add(self)

        
    def remove_neighbor(self, neighbor):
        """In this function, a node can remove the neighbor node called neighbor"""
        self.neighbors.remove(neighbor)
        
    def is_neighbor(self, other_node):
        """This function checks whether other_node is a neighbor to this node.It uses the
        intersection property of sets and returns true if intersected set is not empty."""
        neighbor_list = self.get_neighbors()
        if other_node in neighbor_list:
            return True
        else:
            return False
  
    def get_neighbors(self):
        """This function returns the list of neighbor to the node"""
        neighbor_list = []
        for item in self.neighbors:
            neighbor_list.append(item.id)
        return sorted(neighbor_list)
    
    def to_string(self):
        """This function prints the node and its neighbors."""
        print_str = str(self.id)+"----> "
        temp_list = []
        for item in self.neighbors:
            temp_list.append(item.id)
        temp_list = sorted(temp_list)
        for item in temp_list:
            print_str += str(item)+" "
        return print_str
    
    def node_info(self):
        node = {}
        node[self.id] = []
        temp_list = []
        for item in self.neighbors:
            temp_list.append(item.id)
        temp_list = sorted(temp_list)
        node[self.id] = temp_list
        
        return node
  
class Graph(object):
    """The graph is a set of Nodes. This Graph classes uses Node class methods to make a 
    complete graph and contains all the graph related methods."""
    
    def __init__(self, adjmat=None):
        """This is a constructor of Graph class and it sets up the graph by calling read_adjmat function
        """
        self.Nodes = {}
        if(adjmat is not None):
            self.read_adjmat(adjmat)
        
    def read_adjmat(self, adjMatrix):
        """This function takes user graph dictionary and reads the graph into dictionary of Nodes.
        where key is the node and value is list of neighbor nodes. Each Node contains an id and set of 
        neighbors."""
        for node_id, neighbors in adjMatrix.items():
            self.add_node(node_id)
            curr_node = self.get_node(node_id)
            for neighbor_id in neighbors:
                self.add_node(neighbor_id)
                neighbor_node = self.get_node(neighbor_id)
                curr_node.add_neighbor(neighbor_node, True)
                
    def nodes(self, graph=None):
        """This function returns the list of nodes."""
        if graph is None:
            return list(self.Nodes.keys())
        else:
            return list(graph.keys())
    

    def edges(self, graph=None):
        """This function returns the list of all unique edges."""
        edge_list = []
        if graph is None:
            for node_id, node in self.Nodes.items():
                neighbors = node.get_neighbors()
                for neighbor in neighbors:
                    reverse = (neighbor,node_id)
                    tup = (node_id,neighbor)
                    if reverse not in edge_list and tup not in edge_list:   
                        edge_list.append((node_id,neighbor))
            #return sorted(edge_list)
        else:
            for node_id, neigh in graph.items():
                node = self.get_node(node_id)
                neighbors = node.get_neighbors()
                for neighbor in neighbors:
                    reverse = (neighbor,node_id)
                    tup = (node_id,neighbor)
                    if reverse not in edge_list and tup not in edge_list:   
                        edge_list.append((node_id,neighbor))
        return sorted(edge_list)
            
    
    def add_node(self, node):
        """This function is used to add a node to the graph. It only adds the node 
        if it does not already exist in the graph."""
        if(not self.has_node(node)):
            self.Nodes[node] = Node(node)
    
    def add_nodes_from(self, nodes):
        """This function takes in the list of nodes and adds all nodes if they do not
        exist already in the graph."""
        for node in nodes:
            if(not self.has_node(node)):
                self.Nodes[node] = Node(node)
    
    
    def remove_node(self, node_id):
        """This function remove all instances of given node from the graph"""
        node = self.get_node(node_id)
        neighbors = node.get_neighbors()
        
        for neighbor_id in neighbors:
            curr_neighbor = self.get_node(neighbor_id)
            curr_neighbor.remove_neighbor(node)
            
        del self.Nodes[node_id]
        
    
    def remove_nodes_from(self, nodes):
        """This function remove all instances of given list of nodes from the graph"""
        for node_id in nodes:
            node = self.get_node(node_id)
            neighbors = node.get_neighbors()
            for neighbor_id in neighbors:
                curr_neighbor = self.get_node(neighbor_id)
                curr_neighbor.remove_neighbor(node)
            
            del self.Nodes[node_id]
        
    
    def add_edge(self, nodei, nodej):
        """This function takes in a node tuple as source and destination and adds an edge between them.
        This function checks all the conditions where source or destination or both do not already
        exist in the graph."""

        if self.has_node(nodei):
            source_node = self.get_node(nodei)
            if self.has_node(nodej):
                dest_node = self.get_node(nodej)
                source_node.add_neighbor(dest_node, True)
            else:
                self.add_node(nodej)
                dest_node = self.get_node(nodej)
                source_node.add_neighbor(dest_node, True)
        else:
            self.add_node(nodei) 
            source_node = self.get_node(nodei)
            if self.has_node(nodej):
                dest_node = self.get_node(nodej)
                source_node.add_neighbor(dest_node, True)
            else:
                self.add_node(nodej)
                dest_node = self.get_node(nodej)
                source_node.add_neighbor(dest_node, True)
    
    
    def add_edge_from(self, edges):
        """This function takes in a list of nodetuple as source and destination and adds an edge between the tuple.
        This function checks all the conditions where source or destination or both do not already
        exist in the graph."""
        
        for source_id, dest_id in edges:
            if self.has_node(source_id):
                source_node = self.get_node(source_id)
                if self.has_node(dest_id):
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, True)
                else:
                    self.add_node(dest_id)
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, True)
            else:
                self.add_node(source_id) 
                source_node = self.get_node(source_id)
                if self.has_node(dest_id):
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, True)
                else:
                    self.add_node(dest_id)
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, True)
            
    
    def has_node(self, node_id):
        """This function returns true if node is present in the graph"""
        return node_id in self.Nodes
    
    def has_edge(self,nodei, nodej):
        """This function returns true if an edge is present in the graph"""
        source_id = nodei
        dest_id = nodej
        edge = False
        if self.has_node(source_id):
            source_node = self.get_node(source_id)
            for neighbor in source_node.neighbors:
                if dest_id == neighbor.id:
                    edge = True
        return edge
    
    def remove_edge(self,nodei, nodej):
        if self.has_edge(nodei, nodej):
            curr_i = self.get_node(nodei)
            curr_j = self.get_node(nodej)
            curr_i.remove_neighbor(curr_j)
            
    def remove_edges_from(self, edges):
        for nodei, nodej in edges:
            if(self.has_edge(nodei,nodej)):
                curr_i = self.get_node(nodei)
                curr_j = self.get_node(nodej)
                curr_i.remove_neighbor(curr_j)
                
                
    def neighbors(self, node):
        """"This function returns a list of neighbors of a given node."""
        if node in self.Nodes.keys():
            node_obj = self.get_node(node)
            return sorted(node_obj.get_neighbors())
    
    def number_of_nodes(self):
        """This function returns number of nodes in the graph"""
        return len(self.Nodes)
        
    def number_of_edges(self):
        """"This function returns number of edges in the graph. Since this is undirected graph. So,
        the edges are counted twice hence divided by 2."""
        num = 0
        for node_id, node in self.Nodes.items():
            neighbors = node.get_neighbors()
            num+= len(neighbors)
        edge_num = num//2
        return edge_num
            
    
    def degree(self, node=None):
        """"This function returns the degree of all nodes - that is number of neighbors each node has.
        and depending on input, it also returns the degree of node passed""" 
        if node is None:
            node2degree = set()
            node2neighbor = []
            for node_id, node in self.Nodes.items():
                degree = len(node.get_neighbors())
                node2degree.add((node_id, degree))
                node2neighbor.append((node_id,sorted(node.get_neighbors())))

            return sorted(node2degree), sorted(node2neighbor)
        else:
            if node in self.Nodes.keys():
                node_obj = self.Nodes[node] 
                degree = len(node_obj.get_neighbors())
                return degree
    
    
    def clustering_coefficient(self,node_id=None):
        """this function returns clustering coefficient of given node. If no node is given, it
        calculates the clustering coefficient of all nodes and returns a clustering coefficient
        for the network"""
        c_i = 0
        if node_id is not None:
            if node_id in self.Nodes.keys():
                k_i = self.degree(node_id)
                if k_i < 2:
                    return 0.0
                T_i = 0
                node = self.get_node(node_id)
                neighbors = node.get_neighbors()
                for j1,j2 in itertools.combinations(neighbors,2): # make sure not to have
                    if self.has_edge(j1,j2):
                        T_i +=1
                c_i = (2.0 * T_i) / (k_i * (k_i-1))
                return c_i
            else:
                return None
        else:
            g_ci = 0
            for node_id, node in self.Nodes.items():
                k_i = self.degree(node_id)
                if k_i < 2:
                    c_i =  0.0
                else:
                    T_i = 0
                    node = self.get_node(node_id)
                    neighbors = node.get_neighbors()
                    for j1,j2 in itertools.combinations(neighbors,2): # make sure not to have
                        if self.has_edge(j1,j2):
                            T_i +=1
                    node_ci = round((2.0 * T_i) / (k_i * (k_i-1)),3)
                    #print("node_ci", node_ci)
                    c_i = c_i + node_ci
                    g_ci = c_i
            g_ci = round(g_ci/ self.number_of_nodes(),4)
            
            return g_ci
    
    def subgraph(self, nodes):
        """This function returns a subgraph comprising only of the nodes provided
        along with their neighbor nodes."""
        subgraph = {}
        for node_id in nodes:
            node = self.get_node(node_id)
            neighbors = node.get_neighbors()
            subgraph[node_id] = []
            for neighbor in neighbors:
                if neighbor in nodes:
                    subgraph[node_id].append(neighbor)
        return subgraph
    
    
    def size_gcc(self):
        """This function uses BFS method to find the maximum connected component from the graph 
        and returns the fraction of biggest connected node in the whole graph."""
        total_nodes = self.number_of_nodes()
        gcc_list = []
        for node_id, node in self.Nodes.items():
            size_gcc, connected_nodes = self.BFS(node_id)
            gcc_list.append(size_gcc)
        return max(gcc_list)/total_nodes
    
    
    def BFS(self, node_id):
        """This method implements BFS to find the connected component from given source node to
        all connected nodes. The method returns the connected nodes to the given node and also 
        a number of connected component."""
        if node_id in self.Nodes.keys():
            explored = []
            queue = [node_id]
            while queue:
                curr = queue.pop(0)
                if curr not in explored:
                    explored.append(curr)
                    node = self.get_node(curr)
                    neighbors = node.get_neighbors()
                    for neighbor in neighbors:
                        queue.append(neighbor)
            return len(explored), set(sorted(explored))
        else:
            return None
        
    def info(self):
        
        self.print_graph()
        print("=========================================")
        print("No.Nodes",self.number_of_nodes(), sep=" | ")
        print("No.Edges", self.number_of_edges(),sep=" | ")
        deg_num , deg_list = self.degree()
        print("=========================================")
        print("Node","Degree", sep="     ")
        cluster_coeff = self.clustering_coefficient()
        for index,row in enumerate(deg_num):
            #print(*deg_num[index],sep='    |    ')
            print(format(deg_num[index][0], '02d') ,format(deg_num[index][1], '02d'), sep="   |   ")

        
        #print("clustering coefficient",self.clustering_coefficient())
        print("Network Clustering Coefficient", cluster_coeff)
        print("GCC",self.size_gcc())
  
    
    def get_node(self, node_id):
        """This function returns the node mapped to provided node id."""
        return self.Nodes[node_id]
    
    def print_graph(self,graph=None):
        """This function is used to print the graph."""
        if graph is None:
            for node_id, node in self.Nodes.items():
                print(node.to_string())

        else:
            for node_id, node in graph.items():
                print(node.to_string())
                



class DiGraph(Graph):
    # FILL ME IN
    def __init__(self, adjmat=None):
        """This is a constructor of DiGraph class and it sets up the graph by calling read_adjmat function
        """
        self.Nodes = {}
        if(adjmat is not None):
            self.read_adjmat(adjmat)
            
    def read_adjmat(self, adjMatrix):
        """This function takes user graph dictionary and reads the graph into dictionary of Nodes.
        where key is the node and value is list of neighbor nodes. Each Node contains an id and set of 
        neighbors."""
        for node_id, neighbors in adjMatrix.items():
            self.add_node(node_id)
            curr_node = self.get_node(node_id)
            for neighbor_id in neighbors:
                self.add_node(neighbor_id)
                neighbor_node = self.get_node(neighbor_id)
                curr_node.add_neighbor(neighbor_node, False)
    
    def add_edge(self, nodei, nodej):
        """This function takes in a node tuple as source and destination and adds an edge between them.
        This function checks all the conditions where source or destination or both do not already
        exist in the graph."""
        source_id = nodei
        dest_id = nodej
        if self.has_node(source_id):
            source_node = self.get_node(source_id)
            if self.has_node(dest_id):
                dest_node = self.get_node(dest_id)
                source_node.add_neighbor(dest_node, False)
            else:
                self.add_node(dest_id)
                dest_node = self.get_node(dest_id)
                source_node.add_neighbor(dest_node, False)
        else:
            self.add_node(source_id) 
            source_node = self.get_node(source_id)
            if self.has_node(dest_id):
                dest_node = self.get_node(dest_id)
                source_node.add_neighbor(dest_node, False)
            else:
                self.add_node(dest_id)
                dest_node = self.get_node(dest_id)
                source_node.add_neighbor(dest_node, False)
    
    
    def add_edge_from(self, edges):
        """This function takes in a list of nodetuple as source and destination and adds an edge between the tuple.
        This function checks all the conditions where source or destination or both do not already
        exist in the graph."""
        
        for source_id, dest_id in edges:
            if self.has_node(source_id):
                source_node = self.get_node(source_id)
                if self.has_node(dest_id):
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, False)
                else:
                    self.add_node(dest_id)
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, False)
            else:
                self.add_node(source_id) 
                source_node = self.get_node(source_id)
                if self.has_node(dest_id):
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, False)
                else:
                    self.add_node(dest_id)
                    dest_node = self.get_node(dest_id)
                    source_node.add_neighbor(dest_node, False)
    
    
    def clustering_coefficient(self,node_id=None):
        """this function returns clustering coefficient of given node. If no node is given, it
        calculates the clustering coefficient of all nodes and returns a list of tuples where 
        each tuple represent node with its clustering coefficient."""
        c_i = 0.0
        if node_id is not None:
            if node_id in self.Nodes.keys():
                k_i = self.degree(node_id)
                if k_i < 2:
                    return 0.0
                T_i = 0
                node = self.get_node(node_id)
                neighbors = node.get_neighbors()
                for j1,j2 in itertools.combinations(neighbors,2): # make sure not to have
                    if self.has_edge(j1,j2):
                        T_i +=1
                c_i = (T_i) / (k_i * (k_i-1))
                return c_i
            else:
                return None
        else:
            g_ci = 0
            for node_id, node in self.Nodes.items():
                k_i = self.degree(node_id)
                if k_i < 2:
                    c_i =  0.0
                else:
                    T_i = 0
                    node = self.get_node(node_id)
                    neighbors = node.get_neighbors()
                    for j1,j2 in itertools.combinations(neighbors,2): # make sure not to have
                        if self.has_edge(j1,j2):
                            T_i +=1
                    node_ci = (T_i) / (k_i * (k_i-1))
                    c_i = c_i + node_ci
                    g_ci = c_i
            g_ci = round(g_ci/ self.number_of_nodes(),4)
            
            return g_ci
        
    def in_degree(self,node=None):
        """This function returns the list of tuples (node,in_degree) for all nodes or return the single
        indegree value for specific node passed."""
        node_list = self.nodes()
        edge_list= self.edges()
        if node is None:
            in_degree_list = []
            for node in node_list:
                indegree = 0
                for nodei, nodej in edge_list:
                    if node == nodej:
                        indegree+=1
                in_degree_list.append((node,indegree)) 
            return in_degree_list
        else:
            indegree = 0
            for nodei, nodej in edge_list:
                if node == nodej:
                    indegree+=1
            return indegree
        
    def out_degree(self,node=None):
        """This function returns the list of tuples (node,out_degree) for all nodes or return the single
        outdegree value for specific node passed."""
        node_list = self.nodes()
        edge_list= self.edges()
        if node is None:
            out_degree_list = []
            for node in node_list:
                outdegree = 0
                for nodei, nodej in edge_list:
                    if node == nodei:
                        outdegree+=1
                out_degree_list.append((node,outdegree)) 
            return out_degree_list
        else:
            outdegree = 0
            for nodei, nodej in edge_list:
                if node == nodei:
                    outdegree+=1
            return outdegree
        
    def info(self):
        
        self.print_graph()
        print("=========================================")
        print("No.Nodes",self.number_of_nodes(), sep=" | ")
        print("No.Edges", self.number_of_edges(),sep=" | ")
        in_deg_num = self.in_degree()
        out_deg_num = self.out_degree()
        print("=========================================")
        print("Node","InDegree","OutDegree",sep="  ")
        cluster_coeff = self.clustering_coefficient()
        for index,row in enumerate(in_deg_num):
            print(format(in_deg_num[index][0], '02d') ,format(in_deg_num[index][1], '02d'),format(out_deg_num[index][1], '02d'), sep="   |   ")
        #print("clustering coefficient",self.clustering_coefficient())
        print("Clustering Coeff:", cluster_coeff)
        print("GCC",self.size_gcc())
    
        
        
    
def connected_nodes(graph, source):
    """This function calls BFS function from the graph class and returns the list of
    connected nodes of the given source node."""
    undirected = Graph(graph)
    num, connected_nodes = undirected.BFS(source)
    return connected_nodes


def degree_distribution(graph,norm=False, plot=False):
    """This method computes the probability distribution of a chosen node
    to have exactly k neighbors"""
    undirected = Graph(graph)
    node_num = undirected.number_of_nodes()
    #print("node_num", node_num)
    deg_list = []
    deg_dist_list = []
    for node_id, node in undirected.Nodes.items():
        deg = undirected.degree(node_id)
        deg_list.append(deg)
        
    deg2freq=dict(collections.Counter(deg_list))
    for deg, freq in deg2freq.items():
        deg_dist_list.append((deg,freq/node_num))
        #deg2dist[deg] = freq/node_num
    return deg_dist_list


def is_same_community(i,j):
    if i == j:
        return True
    else:
        return False
    
def modularity(graph, partition):
    undirected = Graph(graph)
    M = undirected.number_of_edges()
    degree_list, degree_len = undirected.degree()
    node_list = undirected.nodes()
    summation = 0

    for node_i in node_list:
        for node_j in node_list:
            if(is_same_community(partition[node_i], partition[node_j])):
                #print("Same", node_i, node_j)
                k_i = undirected.degree(node_i)
                k_j = undirected.degree(node_j)
                part1 = (k_i*k_j)/(2*M)
                
                if(undirected.has_edge(node_i, node_j)):    
                    summation+= (1-part1)
                else:
                    summation+= (0-part1)
   
    modularity = round(summation/(2*M),4)
    return modularity



def relabel_nodes(graph, mapping=None, copy=False):
    """This method takes a graph and node mapping. It renames the node names to the new nodes name provided"""
    
    if mapping is not None:
        if copy:
            relabeled = Graph(graph)
            for node_id, new_node_id in mapping.items():
                if node_id in relabeled.Nodes.keys():
                    node = relabeled.Nodes[node_id]
                    for item in node.neighbors:
                        for val in item.neighbors:
                            if val.id == node_id:
                                val.id = new_node_id
                                if item.id == node_id:
                                    item.id = new_node_id
        return relabeled




def extract_backbone(graph, weights,alpha):
    backbone_graph = {}
    undirected = Graph(graph)
    for node_id, node in undirected.Nodes.items():
        degree = undirected.degree(node_id)
        if degree > 1:
            deg_sum = 0
            neighbor_list= undirected.neighbors(node_id)
            for neighbor in neighbor_list:
                deg_sum+= weights[(node_id,neighbor)]
            for neighbor in neighbor_list:
                edge_weight = weights[(node_id,neighbor)]
                try:
                    pij = float(edge_weight)/deg_sum
                except ZeroDivisionError:
                    pij = 0.0
                if (1-pij)**(degree-1) < alpha:
                    backbone_graph[(node_id, neighbor)] = edge_weight
    return backbone_graph



#relabel_nodes(graph,{1:44,2:55}, True)

print("==============UNDIRECTED GRAPH INFO=============")
#a  = {0: [1,2,3], 1: [0,2], 2: [1,0], 3: [4,5],4: [3,5], 5: [3,4]}
data= read_adjmat("data/small_graph.adjmat.txt",' ')
undirected = Graph(data)
undirected.add_node(50)
undirected.add_nodes_from([51,50])
undirected.add_edge(7,11)
undirected.add_edge_from([(7,11),(50,1)])
undirected.remove_node(51)
undirected.info()



print("==============DIRECTED GRAPH INFO=============")
#data = {0: [1,2,3], 1: [0,2], 2: [1,0], 3: [4,5],4: [3,5], 5: [3,4]}
directed = DiGraph(data)
directed.add_node(7)
directed.add_nodes_from([9,7])
directed.add_edge(7,3)
directed.add_edge_from([(7,5),(7,1)])
directed.remove_node(9)
directed.info()

print("===========Test Functions: Sample Graphs==========")
graph = {5: [2,3], 2: [5,3], 3:[5,2]}
alpha = 0.005
weights = {(5, 2): 9, (2, 3): -3, (2, 5): -8, (5, 3): 9, (3, 2): 0, (3, 5): 5}
edges = [(5,2), (2,3), (2,5),(5,3),(3,2),(3,5)]

backbone = extract_backbone(graph, weights, alpha)
for key, value in backbone.items() :
    print("Extract backbone:", key, "-->", value)

module_graph = {1:[2,3], 2:[1,5], 3:[1,4],4:[3,5], 5:[2,4]}
partition = {1:1,2:2,3:1,4:1,5:2}
print("modularity:", modularity(module_graph,partition))
