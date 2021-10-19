#Currently small scale, only accepts ~32*32*32 cube as transposing full simulation to graph is computationally infeasible

import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting 

from nbodykit.lab import *
from nbodykit import style, setup_logging
from pmesh.pm import ParticleMesh
from sklearn.neighbors import NearestNeighbors

import jraph
import jax.numpy as jnp


#Take particle catalogue and return a pygsp nearest neighbours graph
def cat_to_graph(cat,n):
    
    #extract particle positions from catalogue
    mesh = cat.to_mesh(resampler='tsc')
    pos = np.array(mesh.Position)
    
    
    
    #Take small section of the cube for interpretability

    size = 40
    mask = (pos[:,0]<size)&(pos[:,1]<size)&(pos[:,2]<size)
    smallcube = pos[mask,:]
    
    #NN algorithm
    nbrs = NearestNeighbors(n_neighbors = n,algorithm = 'auto').fit(smallcube)
    distances, indices = nbrs.kneighbors(pos)
    

    #Convert to adjacency matrix and return results
    mat = nbrs.kneighbors_graph(smallcube).toarray()
    np.fill_diagonal(mat,0)
    G = graphs.Graph(mat)
    
    return G,distances,indices

    
    
#Create a nearest neighbours jraph GraphTuple using the same input
def cat_to_graphtuple(cat, n):
    
    mesh = cat.to_mesh(resampler='tsc')
    pos = np.array(mesh.Position)
    
   #Take small section of the cube for interpretability

    size = 40
    mask = (pos[:,0]<size)&(pos[:,1]<size)&(pos[:,2]<size)
    smallcube = pos[mask,:]
    
    #NN algorithm
    nbrs = NearestNeighbors(n_neighbors = n,algorithm = 'auto').fit(smallcube)
    distances, indices = nbrs.kneighbors(pos)
    
    
    
    
    node_features = smallcube #Will break when smallcube removed, remember to replace with pos later
    n_nodes = smallcube.shape[0]
    senders = []
    receivers = []
    edges = []

    for i in range(indices.shape[0]):
            for j in range(indices.shape[1]-1):
                senders.append(indices[i][0])
                receivers.append(indices[i][j+1])
                edges.append([distances[i][j+1]])

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)
    edges = jnp.array(edges)
    
    n_node = jnp.array([smallcube.shape[0]])
    n_edge = jnp.array([distances.shape[0] * (distances.shape[1] - 1)]) 
    global_context = jnp.array([[1]])
    return jraph.GraphsTuple(nodes  = node_features, senders = senders, receivers = receivers,
                         edges = edges, n_node = n_node, n_edge = n_edge, globals = global_context)