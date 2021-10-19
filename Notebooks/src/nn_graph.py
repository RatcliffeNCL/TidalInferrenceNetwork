#Imports
#Currently small scale, only accepts 32*32*32 cube as transposing full simulation to graph is computationally infeasible


import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting 

from nbodykit.lab import *
from nbodykit import style, setup_logging
from pmesh.pm import ParticleMesh
from sklearn.neighbors import NearestNeighbors


#Take particle catalogue and return a nearest neighbours graph
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

    