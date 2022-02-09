import numpy as np
import tensorflow as tf
import galflow
lp = galflow.lightprofiles

import tensorflow_probability as tfp
tfd = tfp.distributions

import galsim

"""
This code aims at reproducing the simulations described in Sheldon & Huff (2017)
https://arxiv.org/pdf/1702.02601.pdf
"""

# Test parameters
batch_size = 4
im_size = 512
stamp_size = 64
pix_scale = 0.05

# Get Disk+Bulge parametric profiles from galsim
cosmos_cat = galsim.COSMOSCatalog()
indices = np.arange(batch_size)
param_gal_list = cosmos_cat.makeGalaxy(indices, gal_type='parametric', chromatic=False)

bulgefit = cosmos_cat.getParametricRecord(indices)['bulgefit']

flux_disk = bulgefit[:,0] #TODO: need to convert this intensity to flux
n_disk = bulgefit[:,2]
hlr_disk = bulgefit[:,1]

flux_bulge = bulgefit[:,0+8] #TODO: need to convert this intensity to flux
n_bulge = bulgefit[:,2+8]
hlr_bulge = bulgefit[:,1+8]

# Light profiles
disk = lp.sersic(n=n_disk, half_light_radius=hlr_bulge, flux=flux_disk, nx=im_size, scale=pix_scale)
bulge = lp.sersic(n=n_bulge, half_light_radius=hlr_bulge, flux=flux_bulge, nx=im_size, scale=pix_scale)

# galaxy ellipticity

sigma_e_disk = .2
sigma_e_bulge = .1

e_disk = tfd.Normal(loc=tf.zeros(2,), scale=sigma_e_disk).sample(batch_size)
e_bulge = tfd.Normal(loc=tf.zeros(2,), scale=sigma_e_bulge).sample(batch_size)

disk = tf.expand_dims(disk, -1)  
disk = galflow.shear(disk, e_disk[:,0], e_disk[:,1])[...,0]

bulge = tf.expand_dims(bulge, -1)  
bulge = galflow.shear(bulge, e_bulge[:,0], e_bulge[:,1])[...,0]

# Prior on $e$ should be like:
# Sheldon & Huff (2017)
# P(e) \propto [1-(e)^2]^2 \exp(-e^2/2\sigma^2)
# 
# - [Bernstein et al. (2014)]
# P(e) \propto e^2[1-(e)^2]^2 \exp(-e^2/2\sigma^2)
# 
# but let's start with a Normal distribution

sigma_e_disk = .2
sigma_e_bulge = .1

e_disk = tfd.Normal(loc=tf.zeros(2,), scale=sigma_e_disk).sample(batch_size)
e_bulge = tfd.Normal(loc=tf.zeros(2,), scale=sigma_e_bulge).sample(batch_size)

disk = tf.expand_dims(disk, -1)  
disk = galflow.shear(disk, e_disk[:,0], e_disk[:,1])[...,0]

bulge = tf.expand_dims(bulge, -1)  
bulge = galflow.shear(bulge, e_bulge[:,0], e_bulge[:,1])[...,0]

bulge_frac = 0.3
gal = bulge_frac * bulge + (1-bulge_frac) * disk

# Need to do something for the flux here

# drawing galaxies on the stamp
left = im_size//2-stamp_size//2
right = im_size//2+stamp_size//2
disk = disk[:,left:right, left:right]
bulge = bulge[:,left:right, left:right]

## Adding Knots

# See: 
# - https://galsim-developers.github.io/GalSim/_build/html/gal.html?highlight=randomknots#galsim.RandomKnots
# - https://galsim-developers.github.io/GalSim/_build/html/tutorials.html#demo-4

# Covolution with the PSF
beta = 3.5
fwhm = .9
psf = galsim.Moffat(beta=beta, fwhm=.9)

psf_q = 0.025
psf_shape = galsim.Shear(q=psf_q, beta=0.*galsim.degrees)
psf.shear(psf_shape)

psf_kimage = psf.drawKImage(scale=0.05).array

psf_kimage = tf.reshape(tf.convert_to_tensor(psf_kimage, tf.complex64), (1,psf_kimage.shape[0], psf_kimage.shape[1], 1))
gal = tf.expand_dims(gal, -1)

# conv = galflow.convolve(gal, psf_kimage)