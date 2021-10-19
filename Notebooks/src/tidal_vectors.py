from nbodykit.lab import *
from nbodykit import style, setup_logging
from pmesh.pm import ParticleMesh
import sys
import os
from mpi4py import MPI
import numpy as np
from pmesh.pm import ParticleMesh
from absl import app
from absl import flags
from astropy.io import fits

   
#Parallel communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size() 
    
    

def lowpass_transfer(r):
  """
  Filter for smoothing the field
  """
  def filter(k, v):
    k2 = sum(ki ** 2 for ki in k)
    return np.exp(-0.5 * k2 * r**2) * v
  return filter

def tidal_transfer(d1, d2):
  """
  Filter to compute the tidal tensor
  """
  def filter(k, v):
    k2 = sum(ki ** 2 for ki in k)
    k2[k2 == 0] = 1.0
    C1 = (v.BoxSize / v.Nmesh)[d1]
    w1 = k[d1] * C1

    C2 = (v.BoxSize / v.Nmesh)[d2]
    w2 = k[d2] * C2
    return w1 * w2 / k2 * v
  return filter

def calculate_tidal_vecs(mesh):
  BoxSize = 1380
  # Instantiate mesh
  pm =  ParticleMesh(BoxSize = BoxSize, # assuming cube
                     Nmesh=[256]*3,
                     dtype='f8',
                     comm=comm)
  # Extract particle positions of original mesh
  pos = np.array(mesh.Position)

  # Create domain decomposition for the particles that matches the mesh
  layout = pm.decompose(pos) #For parallel computations

  # Create a mesh
  rho = pm.create('real')

  # Paint the particles on the mesh
  rho.paint(pos, layout=layout, hold=False)
  print('Density painted')

  rho1 = rho
  # Compute density and forward FFT
  N = pm.comm.allreduce(len(pos))
  fac = 1.0 * pm.Nmesh.prod() / N
  rho[...] *= fac
  rhok = rho.r2c()
  rhok = rhok.apply(lowpass_transfer(r=1.0))
  print('Rho computed')
  rho2 = rhok.c2r()
  # Create mock galaxy positions
  x = np.linspace(0, BoxSize, 256)
  y = np.linspace(0, BoxSize, 256)
  z = np.linspace(0,BoxSize, 256)
    
  xv, yv, zv = np.meshgrid(x, y, z)
  gal_pos = np.stack([xv.reshape((-1,)),yv.reshape((-1,)),zv.reshape((-1,))], axis=-1)

  # Computing the distribution of galaxies in the cube
  layout_gal = pm.decompose(gal_pos)
  gal_pos = layout_gal.exchange(gal_pos)

  # Retrieve local density on each galaxy
  density = rhok.c2r().readout(gal_pos)
  density = layout_gal.gather(density, mode='all')
 # if rank == 0:
 #   fits.writeto(FLAGS.output_dir+'/density_247.fits', density, overwrite=True)

  #Do tidal calculations  
  tidal_tensors = []
  for i in range(3):
    tidal_tensors.append(np.stack([rhok.apply(tidal_transfer(j, i)).c2r().readout(gal_pos) for j in range(3)], axis=-1))
  tidal_tensors = np.stack(tidal_tensors, axis=-1)

  # At this point, tidal_tensor in each rank contains the tensor for the local
  # galaxies
  # Now computing diagonalization
  vals, vects = np.linalg.eigh(tidal_tensors)

  # Retrieving the computed values
  vals = layout_gal.gather(vals, mode='all')
  vects = layout_gal.gather(vects, mode='all')

  return density, vals, vects, rho1, rho2
  #if rank == 0:
  #  fits.writeto(FLAGS.output_dir+'/tidal_val_247.fits', vals, overwrite=True)
  #  fits.writeto(FLAGS.output_dir+'/tidal_vects_247.fits', vects, overwrite=True)

