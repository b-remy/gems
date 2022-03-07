import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import galsim
import galflow
lp = galflow.lightprofiles
import timeit

from psf import get_gaussian_psf

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
  # gamma = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=0.05, name="gamma")
  gamma = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=0.005, name="gamma")

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