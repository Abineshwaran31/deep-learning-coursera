#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import networkx as nx
import pandas as pd


# In[8]:


def nearest_neighbour_initialization(g,closed_tour=False):
    curr_node = np.random.choice(g.nodes) 
    path = [curr_node]
    not_visited = set(g.nodes)-{curr_node}
    while not_visited:
        not_visited_neighbours = not_visited&set(g.neighbors(curr_node))
        key =lambda x: g[curr_node][x]["weight"]
        curr_node = min(not_visited_neighbours,key = key) 
        path.append(curr_node)
        not_visited.remove(curr_node)
    if closed_tour:
        path.append(path[0])
    return path


# In[9]:


nearest_neighbour_initialization(g)


# In[10]:


from collections import defaultdict
def has_cycle(g):
    try:
        nx.find_cycle(g)
    except nx.NetworkXNoCycle:
        return False
    return True

def get_path_from_edges(edges,closed_tour=False):
    path_graph = nx.Graph(edges)
    # if it is an open tour start from a node with a single degree
    curr = min(path_graph.nodes,key=path_graph.degree) 
    path,visited = [curr],{curr}
    while len(path)<len(path_graph):
        curr = (set(path_graph.neighbors(curr))-visited).pop()
        visited.add(curr)
        path.append(curr)
    if closed_tour:
        path.append(path[0])
    return path
def shortest_edge_initialization(g,method="greedy", closed_tour = False):
    edge_list = set(g.edges)
    times_visited  = defaultdict(int)
    tour = set() 
    max_tour_len = len(g) if closed_tour else len(g)-1
    key = nx.get_edge_attributes(g,"weight").get
    while len(tour)<max_tour_len:
       u,v = min(edge_list, key=key)
       times_visited[u]+=1
       times_visited[v]+=1
       tour.add((u,v))
       edge_list.remove((u,v))
       for u,v in set(edge_list):
            if (
                (has_cycle(nx.Graph(tour|{(u,v)})) and len(tour) != len(g)-1)
                or times_visited[u] ==2 or times_visited[v] ==2

            ):
                edge_list.remove((u,v))

    return get_path_from_edges(tour,closed_tour=closed_tour)
shortest_edge_initialization(g)


# In[11]:


def partially_matched_crossover(p1,p2):
    pt = np.random.randint(1,len(p1)-1) # crossover point
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
    m1=set(p1)-set(c1)
    m2=set(p2)-set(c2)
    
    if m1 or m2:
        c1=pd.Series(c1)
        c1[c1.duplicated()]=list(m1)
        c2=pd.Series(c2)
        c2[c2.duplicated()]=list(m2)
        c1=c1.to_list()
        c2=c2.to_list()
        
    return c1,c2
parents=[[1, 2, 0, 4, 6, 5, 3], [0, 1, 4, 2, 6, 5, 3]]
print(partially_matched_crossover(*parents))


# In[12]:


def order_crossover(p1,p2):
    start = np.random.randint(0,len(p1)-1)
    end = np.random.randint(start+1,len(p1) if start !=0 else len(p1)-1)
    def fill_blanks(p1,p2,s,e):
        
        unvisited_nodes = p2.copy()
        for node in p1[s:e]:
            unvisited_nodes.remove(node)

        c = p1.copy()
        for i in range(len(p1)):
            if i<s or i>=e:
                c[i] = unvisited_nodes.pop(0)
        return c

    c1 = fill_blanks(p1,p2,start,end)
    c2 = fill_blanks(p2,p1,start,end)
    return c1,c2
order_crossover(*parents)


# In[13]:


def inversion_mutation(p):
    start = np.random.randint(0,len(p)-1)
    end = np.random.randint(start+1,len(p)+1)
    subtour = p[start:end]
    c = p.copy()
    for i in range(start,end):
        c[i] = subtour.pop()
    return c
p=[2, 0, 4, 1, 3, 6, 5]
inversion_mutation(p)


# In[14]:


def insertion_mutation(p):
    i = np.random.randint(1,len(p))
    k = np.random.randint(0,len(p)-1)
    c = p.copy()
    print(c,i,k)
    c.insert(k,c.pop(i))
    return c
insertion_mutation(p)


# In[ ]:




