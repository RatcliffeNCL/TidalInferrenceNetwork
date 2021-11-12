#This script will make the entire test dataset. Probably want to run it on Rocket
#Create test graph from nbodykit

from nbodykit.lab import *
from nbodykit import style, setup_logging
from pmesh.pm import ParticleMesh
from tidal_vectors import *
from numpy import asarray
from numpy import savez_compressed



redshift = 0.55
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

BoxSize=1380
nbar=3e-3
bias=1.0
nmesh = 256


seeds = []
for i in range(1,3):
    seed = i * 13
    cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=nmesh, bias=bias, seed=seed)

    #extract particle positions from catalogue
    meshes = cat.to_mesh(resampler='tsc')

    seeds.append(calculate_tidal_vecs(meshes))

savez_compressed('../data/nbodydata.npz', seeds)