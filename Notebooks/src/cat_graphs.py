
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting 

from nbodykit.lab import *
from nbodykit import style, setup_logging
from pmesh.pm import ParticleMesh
from sklearn.neighbors import NearestNeighbors

from src.tidal_vectors import *

import jraph
import jax.numpy as jnp



#Take particle catalogue and return a pygsp nearest neighbours graph
#Have put in defining the wanted box as part of the algorithm, to change on the fly

def cat_to_graph(cat,mesh,n_neighbours,mask_size):

    pos = np.array(mesh.Position)

    mask = (pos[:,0]<mask_size)&(pos[:,1]<mask_size)&(pos[:,2]<mask_size)
    smallcube = pos[mask,:]
    
    #NN algorithm
    nbrs = NearestNeighbors(n_neighbors = n_neighbours, algorithm = 'auto').fit(smallcube)
    distances, indices = nbrs.kneighbors(smallcube)
    

    #Convert to adjacency matrix and return results
    mat = nbrs.kneighbors_graph(smallcube).toarray()
    np.fill_diagonal(mat,0)
    G = graphs.Graph(mat)
    
    return G,distances,indices
    
    
#Create a nearest neighbours jraph GraphTuple using the same input
#Add additional parameters to the graph. Want to embed additional information into it.
def cat_to_graphtuple(cat,n,n_neighbours,mask_size):
    
    #mesh = cat.to_mesh(resampler='tsc') #Redoing the mesh is wasteful, ask before as you probably already have it
    pos = np.array(mesh.Position)   
    #size = 40
    #mask = (pos[:,0]<size)&(pos[:,1]<size)&(pos[:,2]<size)
    #smallcube = pos[mask,:]
   
    #NN algorithm
    nbrs = NearestNeighbors(n_neighbors = n,algorithm = 'auto').fit(pos)
    distances, indices = nbrs.kneighbors(pos)
   
    
    node_features = pos #This is where to embed additional information
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


def mesh_to_graphtuple(mesh,n_neighbours,mask_size):
    
    #mesh = cat.to_mesh(resampler='tsc') #Redoing the mesh is wasteful, ask before as you probably already have it
    tidal_results = calculate_tidal_vecs(mesh)

    pos = np.array(mesh.Position)   
    size = mask_size
    mask = (pos[:,0]<size)&(pos[:,1]<size)&(pos[:,2]<size)
    smallcube = pos[mask,:]
    
    #Nearest Neighbour algorithm
    nbrs = NearestNeighbors(n_neighbors = n_neighbours,algorithm = 'auto').fit(smallcube)
    distances, indices = nbrs.kneighbors(smallcube)
   
    
    node_features = smallcube #This is where to embed additional information
    real = mesh.to_real_field()
    
    vals = tidal_results[1].reshape([256,256,256,3])
    vects = tidal_results[2].reshape([256,256,256,3,3])
    rhos = tidal_results[3]
    #round node location to nearest int to find corresponding location withing tidal vector results
    #Yes this will induce issues, but it should work FOR NOW
    for node in node_features:
        x = int(round(node[0]))
        y = int(round(node[1]))
        z = int(round(node[2]))
        node = np.append(node,vals[x,y,z])
        node = np.append(node,vects[x,y,z])
        node = np.append(node,rhos[x,y,z])
        
        
                  
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
