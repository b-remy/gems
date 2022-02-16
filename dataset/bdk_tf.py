import numpy as np
import tensorflow as tf
import galflow
lp = galflow.lightprofiles
import math

import tensorflow_probability as tfp
tfd = tfp.distributions

import galsim

import matplotlib.pyplot as plt

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

# Get fit parameters.  For 'sersicfit', the result is an array of 8 numbers for each
# galaxy:
#     SERSICFIT[0]: intensity of light profile at the half-light radius.
#     SERSICFIT[1]: half-light radius measured along the major axis, in units of pixels
#                   in the COSMOS lensing data reductions (0.03 arcsec).
#     SERSICFIT[2]: Sersic n.
#     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
#     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are all
#                   elliptical.
#     SERSICFIT[5]: x0, the central x position in pixels.
#     SERSICFIT[6]: y0, the central y position in pixels.
#     SERSICFIT[7]: phi, the position angle in radians.  If phi=0, the major axis is
#                   lined up with the x axis of the image.
# For 'bulgefit', the result is an array of 16 parameters that comes from doing a
# 2-component sersic fit.  The first 8 are the parameters for the disk, with n=1, and
# the last 8 are for the bulge, with n=4.

cosmos_cat = galsim.COSMOSCatalog()
indices = np.arange(batch_size)
param_gal_list = cosmos_cat.makeGalaxy(indices, gal_type='parametric', chromatic=False)

bulgefit = cosmos_cat.getParametricRecord(indices)['bulgefit']

i_disk = bulgefit[:,0]
n_disk = 1.*tf.ones(batch_size)
hlr_disk = bulgefit[:,1]

i_bulge = bulgefit[:,0+8] #TODO: need to convert this intensity to flux
n_bulge = 4.*tf.ones(batch_size)
hlr_bulge = bulgefit[:,1+8]

q_disk = bulgefit[:,3]
phi_disk = bulgefit[:,7]

q_bulge = bulgefit[:,3+8]
phi_bulge = bulgefit[:,7+8]

b_1 = 1.67834699
b_4 = 7.66924944

def get_flux(hlr, n, q, i, b_n):
  return 2*math.pi*tf.math.exp(tf.math.lgamma(2*n))*tf.math.exp(b_n)*q*hlr*hlr*i/tf.math.pow(b_n, 2*n)

flux_disk = get_flux(hlr_disk, n_disk, q_disk, i_disk, b_1)
flux_bulge = get_flux(hlr_bulge, n_bulge, q_bulge, i_bulge, b_4)

# Light profiles
disk = lp.sersic(n=n_disk, half_light_radius=hlr_bulge, flux=flux_disk, nx=im_size, scale=pix_scale)
bulge = lp.sersic(n=n_bulge, half_light_radius=hlr_bulge, flux=flux_bulge, nx=im_size, scale=pix_scale)

# galaxy ellipticity

# reduced shear notation
# $$\epsilon_1 + i\epsilon_2 = \frac{1-q}{1+q}\exp{\left(2i\phi\right)}$$
def get_e(q, phi):
  e1 = (1-q)/(1+q) * tf.math.cos(2*phi)
  e2 = (1-q)/(1+q) * tf.math.sin(2*phi)
  return tf.cast(tf.stack([e1,e2], axis=-1), tf.float32)

e_disk = get_e(q_disk, phi_disk)
e_bulge = get_e(q_bulge, phi_bulge)

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

# plt.figure(figsize=(10,5))
plt.subplot(331)
plt.title('disk')
plt.imshow(disk[0,...], cmap='gist_stern')

plt.subplot(332)
plt.title('bulge')
plt.imshow(bulge[0,...], cmap='gist_stern')

plt.subplot(333)
plt.title('gal')
plt.imshow(gal[0,...], cmap='gist_stern')
plt.colorbar()
#---
plt.subplot(334)
plt.title('disk')
plt.imshow(disk[1,...], cmap='gist_stern')

plt.subplot(335)
plt.title('bulge')
plt.imshow(bulge[1,...], cmap='gist_stern')

plt.subplot(336)
plt.title('gal')
plt.imshow(gal[1,...], cmap='gist_stern')
plt.colorbar()
#---
plt.subplot(337)
plt.title('disk')
plt.imshow(disk[2,...], cmap='gist_stern')

plt.subplot(338)
plt.title('bulge')
plt.imshow(bulge[2,...], cmap='gist_stern')

plt.subplot(339)
plt.title('gal')
plt.imshow(gal[2,...], cmap='gist_stern')
plt.colorbar()
#---
plt.show()

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