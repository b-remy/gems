import edward2 as ed
import tensorflow as tf

import numpy as np

import os
import fnmatch
from absl import app
from absl import flags

import galsim
import galflow
lp = galflow.lightprofiles

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi

def main(_):

  y = tf.convert_to_tensor(np.load("data/stamp.npy"), dtype=tf.float32)
  print(y.shape)

  # INPUTS
  stamp_size = y.shape[0]
  # N = 5

  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_e = 0.003
  # noise = ed.Normal(loc=tf.zeros((nx, ny)), scale=sigma_e, name="noise")

  # PSF model from galsim COSMOS catalog
  cat = galsim.COSMOSCatalog()
  psf = cat.makeGalaxy(2,  gal_type='real', noise_pad_size=0).original_psf

  interp_factor=2
  padding_factor=2
  Nk = stamp_size*interp_factor*padding_factor
  from galsim.bounds import _BoundsI
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                          scale=2.*_pi/(stamp_size*padding_factor*_scale),
                          recenter=False)

  kpsf = tf.cast(np.fft.fftshift(imkpsf.array.reshape(1, Nk, Nk//2+1), axes=1), tf.complex64)

  # Flux
  F = 16.693710205567005
    
  def model(target):
    # prior on Sersic index n
    log_l_n = ed.Normal(loc=.1, scale=.39, name="n")
    n = tf.math.exp(log_l_n * _log10)

    # prior on Sersic size half light radius
    log_l_hlr = ed.Normal(loc=-.68, scale=.3, name="hlr")
    hlr = tf.math.exp(log_l_hlr * _log10)

    # prior on shear
    gamma = ed.Normal(loc=tf.zeros((2)), scale=.09, name="shear")

    # Generate light profile
    profile = lp.sersic(n, scale_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)

    # Shear the image
    tfg1 = tf.reshape(tf.convert_to_tensor(gamma[0], tf.float32), (1))
    tfg2 = tf.reshape(tf.convert_to_tensor(gamma[1], tf.float32), (1))
    ims = tf.cast(tf.reshape(profile, (1,stamp_size,stamp_size,1)), tf.float32)
    ims = galflow.shear(ims, tfg1, tfg2)

    # Convolve the image with the PSF
    profile = galflow.convolve(ims, kpsf,
                        zero_padding_factor=padding_factor,
                        interp_factor=interp_factor)[0,...,0]

    # Evaluate likelihood
    # image = profile + noise
    Z = tf.math.pow(tf.math.sqrt(2*_pi) * sigma_e, stamp_size**2)
    l = tf.math.exp(tf.reduce_sum((target - profile)*(target - profile)) / (2*sigma_e*sigma_e)) / Z

    return l

  # Target log-probability function
  log_joint = ed.make_log_joint_fn(model)

  def target_log_prob_fn(n, hlr, gamma):
    return log_joint(n=n, hlr=hlr, shear=gamma, target=y)

  n = 1.
  hlr = 1.
  gamma = tf.zeros(2)
  print(target_log_prob_fn(n=n, hlr=hlr, gamma=gamma))

if __name__ == "__main__":
  app.run(main)