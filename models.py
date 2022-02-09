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

def sersic_model(batch_size=16, stamp_size=64, scale=0.03):
  """PGM:
  - Sersic light profiles
  - Varying intrinsic ellipticity
  - Constant shear
  """
  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_e = 0.0003

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68*tf.ones(batch_size), scale=.3, name="hlr")
  hlr = tf.math.exp(log_l_hlr * _log10)

  # prior on Sersic index n
  log_l_n = ed.Normal(loc=.1*tf.ones(batch_size), scale=.39, name="n")
  n = tf.math.exp(log_l_n * _log10)

  # Flux
  F = 16.693710205567005 * tf.ones(batch_size)

  # Generate light profile
  profile = lp.sersic(n=n, half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=scale)

  # prior on intrinsic galaxy ellipticity
  e = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=.2, name="e")

  # Constant shear in the field
  gamma = ed.Normal(loc=tf.zeros((2,)), scale=0.05, name="gamma")

  # Apply intrinsic ellipticity on profiles the image
  ims = tf.expand_dims(profile, -1)  
  ims = galflow.shear(ims, e[:,0], e[:,1])
    
  # Apply same shear on all images
  ims = galflow.shear(ims, 
                      gamma[0]*tf.ones(batch_size),
                      gamma[1]*tf.ones(batch_size))

  # Convolve the image with the PSF
  interp_factor = 1
  padding_factor = 1
  kpsf = get_gaussian_psf(scale, stamp_size, interp_factor, padding_factor)

  profile = galflow.convolve(ims, kpsf,
                      zero_padding_factor=padding_factor,
                      interp_factor=interp_factor)[...,0]

  # Returns likelihood
  return  ed.Normal(loc=profile, scale=sigma_e, name="obs")




def gaussian_model(batch_size=25, stamp_size=64, scale=0.03):
  """PGM:
  - Gaussian light profiles
  - Varying intrinsic ellipticity
  - Constant shear
  """
  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_e = 0.0003

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68*tf.ones(batch_size), scale=.3, name="hlr")
  hlr = tf.math.exp(log_l_hlr * _log10)

  # Flux
  F = 16.693710205567005 * tf.ones(batch_size)

  # Generate light profile
  profile = lp.gaussian(half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=scale)

  # prior on intrinsic galaxy ellipticity
  e = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=.2, name="e")

  # Constant shear in the field
  gamma = ed.Normal(loc=tf.zeros((2,)), scale=0.05, name="gamma")

  # Apply intrinsic ellipticity on profiles the image
  ims = tf.expand_dims(profile, -1)  
  ims = galflow.shear(ims, e[:,0], e[:,1])
    
  # Apply same shear on all images
  ims = galflow.shear(ims, 
                      gamma[0]*tf.ones(batch_size),
                      gamma[1]*tf.ones(batch_size))

  # Convolve the image with the PSF
  interp_factor = 1
  padding_factor = 1
  kpsf = get_gaussian_psf(scale, stamp_size, interp_factor, padding_factor)

  profile = galflow.convolve(ims, kpsf, 
                        zero_padding_factor=padding_factor,
                        interp_factor=interp_factor)[...,0]

  # Returns likelihood
  return  ed.Normal(loc=profile, scale=sigma_e, name="obs")

