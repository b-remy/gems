import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import galflow
lp = galflow.lightprofiles
import timeit

from psf import get_gaussian_psf
from shear import shear_map

import tensorflow_addons as tfa

_log10 = tf.math.log(10.)
_pi = np.pi

### Forward models

def sersic_model(batch_size=1, num_gal=25, stamp_size=64, scale=0.03):
  """PGM:
  - Sersic light profiles
  - Varying intrinsic ellipticity
  - Constant shear
  """
  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_e = 0.003

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68*tf.ones((batch_size, num_gal)), scale=.3, name="hlr")
  hlr = tf.math.exp(log_l_hlr * _log10)
  hlr = tf.reshape(hlr, [-1])

  # prior on Sersic index n
  log_l_n = ed.Normal(loc=.1*tf.ones((batch_size, num_gal)), scale=.39, name="n")
  n = tf.math.exp(log_l_n * _log10)
  n = tf.reshape(n, [-1])

  # Flux
  F = 16.693710205567005 * tf.ones((batch_size, num_gal))
  F = tf.reshape(F, [-1])

  # Generate light profile
  profile = lp.sersic(n=n, half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=scale)

  # prior on intrinsic galaxy ellipticity
  e = ed.Normal(loc=tf.zeros((batch_size, num_gal, 2)), scale=.2, name="e")
  e = e + 0. # fixes evalutation with tf.Variable()
  e = tf.reshape(e, [batch_size*num_gal, 2])

  # Apply intrinsic ellipticity on profiles the image
  ims = tf.expand_dims(profile, -1)  
  ims = galflow.shear(ims, e[:,0], e[:,1])

  # Constant shear in the field
  gamma = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=0.05, name="gamma")

  # Apply same shear on all images
  ims = tf.reshape(ims, [batch_size, num_gal, nx, ny])
  ims = tf.transpose(ims, perm=[0, 2, 3, 1])
  ims = galflow.shear(ims, 
                      gamma[:,0],
                      gamma[:,1])
  
  ims = tf.transpose(ims, perm=[0, 3, 1, 2])
  ims = tf.reshape(ims, [batch_size*num_gal, nx, ny, 1])

  # Convolve the image with the PSF
  interp_factor = 1
  padding_factor = 1
  kpsf = get_gaussian_psf(scale, stamp_size, interp_factor, padding_factor)

  profile = galflow.convolve(ims, kpsf,
                      zero_padding_factor=padding_factor,
                      interp_factor=interp_factor)[...,0]
  
  profile = tf.reshape(profile, [batch_size, num_gal, nx, ny])

  # Returns likelihood
  return  ed.Normal(loc=profile, scale=sigma_e, name="obs")


def gaussian_model(batch_size=1, num_gal=25, stamp_size=64, scale=0.03, fixed_flux=False):
  """PGM:
  - Gaussian light profiles
  - Varying intrinsic ellipticity
  - Constant shear
  """
  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  # sigma_e = 0.003
  sigma_e = 0.003

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68*tf.ones((batch_size, num_gal)), scale=.3, name="hlr")
  hlr = tf.math.exp(log_l_hlr * _log10)
  hlr = tf.reshape(hlr, [-1])
  
  # Flux
  # log_l_F = ed.Normal(loc=0.74, scale=.60, name="F")
  # F = tf.math.exp(log_l_F * _log10) * tf.ones(batch_size)
  if fixed_flux:
    F = 16.693710205567005 * tf.ones((batch_size, num_gal))
  else:
    F = ed.Uniform(0.*tf.ones((batch_size, num_gal)), 50.*tf.ones((batch_size, num_gal)), name="F")
  F = tf.reshape(F, [-1])

  # Generate light profile
  profile = lp.gaussian(half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=scale)

  # prior on intrinsic galaxy ellipticity
  e = ed.Normal(loc=tf.zeros((batch_size, num_gal, 2)), scale=.2, name="e")
  e = e + 0. # fixes evalutation with tf.Variable()
  e = tf.reshape(e, [batch_size*num_gal, 2])

  # Apply intrinsic ellipticity on profiles the image
  ims = tf.expand_dims(profile, -1)  
  ims = galflow.shear(ims, e[:,0], e[:,1])
    
  # Constant shear in the field
  gamma = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=0.05, name="gamma")

  # Apply same shear on all images
  ims = tf.reshape(ims, [batch_size, num_gal, nx, ny])
  ims = tf.transpose(ims, perm=[0, 2, 3, 1])
  ims = galflow.shear(ims, 
                      gamma[:,0],
                      gamma[:,1])
  
  ims = tf.transpose(ims, perm=[0, 3, 1, 2])
  ims = tf.reshape(ims, [batch_size*num_gal, nx, ny, 1])

  # Convolve the image with the PSF
  interp_factor = 1
  padding_factor = 1
  kpsf = get_gaussian_psf(scale, stamp_size, interp_factor, padding_factor)

  profile = galflow.convolve(ims, kpsf, 
                        zero_padding_factor=padding_factor,
                        interp_factor=interp_factor)[...,0]

  # ims = tf.reshape(ims, [batch_size, num_gal, nx, ny])
  
  profile = tf.reshape(profile, [batch_size, num_gal, nx, ny])

  # Returns likelihood
  return ed.Normal(loc=profile, scale=sigma_e, name="obs")

def varying_shear_gaussian_model(batch_size=1, num_gal=8*8, stamp_size=64, scale=0.03, fixed_flux=False, shear_map_width=16, resolution=5.):
  """PGM:
  - Gaussian light profiles
  - Varying intrinsic ellipticity
    - shear_map_width in pixels
    - resolution in arcmin/pixel
  - Constant shear
  """
  # shear_map_width = 16 # pixels
  # resolution = 5. # arcmin/pixel
  # num_gal_x = 8
  num_gal_x = int(np.sqrt(num_gal))
  # num_gal = num_gal_x**2
  # Galaxy positions
  # galaxies on a grid
  x = np.linspace(0., shear_map_width-1, num_gal_x)
  xx, yy = np.meshgrid(x, x)
  # pos_x = tf.reshape(xx, -1)
  # pos_y = tf.reshape(yy, -1)
  gal_pos = tf.cast(tf.expand_dims(tf.stack([xx, yy], axis=-1), 0), tf.float32)

  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  # sigma_e = 0.003
  sigma_e = 0.003

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68*tf.ones((batch_size, num_gal)), scale=.3, name="hlr")
  hlr = tf.math.exp(log_l_hlr * _log10)
  hlr = tf.reshape(hlr, [-1])
  
  # Flux
  # log_l_F = ed.Normal(loc=0.74, scale=.60, name="F")
  # F = tf.math.exp(log_l_F * _log10) * tf.ones(batch_size)
  if fixed_flux:
    F = 16.693710205567005 * tf.ones((batch_size, num_gal))
  else:
    F = ed.Uniform(0.*tf.ones((batch_size, num_gal)), 50.*tf.ones((batch_size, num_gal)), name="F")
  F = tf.reshape(F, [-1])

  # Generate light profile
  profile = lp.gaussian(half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=scale)

  # prior on intrinsic galaxy ellipticity
  e = ed.Normal(loc=tf.zeros((batch_size, num_gal, 2)), scale=.1, name="e")
  e = e + 0. # fixes evalutation with tf.Variable()
  e = tf.reshape(e, [batch_size*num_gal, 2])

  # Apply intrinsic ellipticity on profiles the image
  ims = tf.expand_dims(profile, -1)  
  ims = galflow.shear(ims, e[:,0], e[:,1])

  # # Constant shear in the field
  # gamma = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=0.05, name="gamma")
  # Gaussian Random Field
  # Shear gridpositions
  x = np.linspace(0., shear_map_width, shear_map_width)
  xx, yy = np.meshgrid(x, x)
  pos_shear_x = xx
  pos_shear_y = yy

  shear_map_p = shear_map(batch_size=batch_size, map_width=shear_map_width, resolution=resolution, name="latent_shear")
  # print('galaxy positions', gal_pos.shape)
  # print('shear map shape', shear_map_p.shape)
  
  gamma_values = tfa.image.resampler(shear_map_p, gal_pos)
  # print('interp shear shape', gamma_values.shape)
  
  gamma = tf.reshape(gamma_values, [batch_size*num_gal, 2])
  # Apply same shear on all images
  ims = tf.reshape(ims, [batch_size*num_gal, nx, ny, 1])
  # ims = tf.transpose(ims, perm=[0, 2, 3, 1])
  ims = galflow.shear(ims, 
                      gamma[:,0],
                      gamma[:,1])
  
  # ims = tf.transpose(ims, perm=[0, 3, 1, 2])
  # ims = tf.reshape(ims, [batch_size*num_gal, nx, ny, 1])

  # Convolve the image with the PSF
  interp_factor = 1
  padding_factor = 1
  kpsf = get_gaussian_psf(scale, stamp_size, interp_factor, padding_factor)

  profile = galflow.convolve(ims, kpsf, 
                        zero_padding_factor=padding_factor,
                        interp_factor=interp_factor)[...,0]

  # ims = tf.reshape(ims, [batch_size, num_gal, nx, ny])
  
  profile = tf.reshape(profile, [batch_size, num_gal, nx, ny])

  # Returns likelihood
  return ed.Normal(loc=profile, scale=sigma_e, name="obs")