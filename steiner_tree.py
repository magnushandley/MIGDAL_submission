# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx.algorithms.approximation import steiner_tree
from networkx.algorithms import dag_longest_path

"""
Implementation of the steiner tree (end of section 5.1 in report) - loads a file containing an array of coordinates from the degeneracy breaking ridgefinder
"""

coords = np.loadtxt("/Users/magnus/Documents/MIGDAL/stripreader/nx_test_3.txt")

def distance(point1,point2):
    difference = np.subtract(point1,point2)
    square = difference.dot(difference)
    return(np.sqrt(square))

def xtostrip(x,offset = 15,stripwidth = 10/120):
    return(int(x/stripwidth)+15)


nodes = coords.T



G = nx.Graph()

for i in range(len(nodes)):
    G.add_node(i)

index = 0
zprev = False
verticies = []

for i in range(len(coords[2])):
    for j in range(i+1,len(coords[2])):
        G.add_edge(i,j,weight=distance(nodes[i],nodes[j]))

pos = nodes

#create nodes and edges
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

#Plot all nodes and edges - very messy
#fig = plt.figure()
#ax = fig.add_subplot(111, projection="3d")
#
#ax.scatter(*node_xyz.T)
#
#for vizedge in edge_xyz:
#    ax.plot(*vizedge.T, color="tab:gray")
#
#plt.show()



#
#Use Melhorns algorithm to create steiner tree
st = steiner_tree(G,range(len(nodes)))
# longest_path = dag_longest_path(st)

pos = nodes

#Extract nodes from computed steiner tree
node_xyz = np.array([pos[v] for v in sorted(st)])
print(node_xyz)
edge_xyz = np.array([(pos[u], pos[v]) for u, v in st.edges()])

#Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(*node_xyz.T)

for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")

ax.set_xlabel("x[cm]")
ax.set_ylabel("y[cm]")
ax.set_zlabel("z[cm]")            
plt.show()


