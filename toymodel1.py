import tensorflow as tf
import edward2 as ed

import numpy as np
import matplotlib.pyplot as plt

import os
import fnmatch
from absl import app
from absl import flags

import utils

import galsim
import galflow
lp = galflow.lightprofiles

flags.DEFINE_integer("N", 5, "Number of stamps on x and y axes")
flags.DEFINE_boolean("plot", True, "Should we plot the simulations?")
flags.DEFINE_string("output_dir", "data", "Path to output simulations")
flags.DEFINE_string("model_name", "toymodel1", "Name of the probabilistic model")
flags.DEFINE_boolean("save", False, "Should we store the simulations?")
flags.DEFINE_float("sigma_n", 1e-3, "Level of noise in data")

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi

FLAGS = flags.FLAGS

stamp_size = 64
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

@tf.function
def model(batch_size, stamp_size, shear):
  """Toy model
  """
  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_n = FLAGS.sigma_n
  noise = ed.Normal(loc=tf.zeros((batch_size, nx, ny)), scale=sigma_n)

  # prior on Sersic index n
  log_l_n = ed.Normal(loc=.1*tf.ones(batch_size), scale=.39)
  n = tf.math.exp(log_l_n * _log10)

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68*tf.ones(batch_size), scale=.3)
  hlr = tf.math.exp(log_l_hlr * _log10)

  # Flux
  F = 16.693710205567005 * tf.ones(batch_size)

  # Generate light profile
  profile = lp.sersic(n, half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)

  # prior on intrinsic galaxy ellipticity
  e = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=.2, name="e")
  tfe1 = e[:, 0]
  tfe2 = e[:, 1]
  ims = tf.cast(tf.reshape(profile, (batch_size,stamp_size,stamp_size,1)), tf.float32)
  ims = galflow.shear(ims, tfe1, tfe2)

  # prior on shear
  # gamma = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=.09)
  gamma = tf.convert_to_tensor(shear, dtype=tf.float32)
  gamma = tf.repeat(tf.expand_dims(gamma, 0), batch_size, axis=0)
  # gamma = tf.zeros(2)

  # Shear the image
  tfg1 = gamma[:, 0]
  tfg2 = gamma[:, 1]
  #ims = tf.cast(tf.reshape(profile, (batch_size,stamp_size,stamp_size,1)), tf.float32)
  ims = galflow.shear(ims, tfg1, tfg2)
  
  # Convolve the image with the PSF
  profile = galflow.convolve(ims, kpsf,
                      zero_padding_factor=padding_factor,
                      interp_factor=interp_factor)[...,0]

  # Add noise
  image = profile + noise
  return image

def main(_):
  stamp_size = 64
  N = FLAGS.N
  batch_size = N*N
  sigma_n = FLAGS.sigma_n
  true_shear = [0.05, 0.0]

  sims = model(batch_size, stamp_size, true_shear)
  print(sims.shape)

  if FLAGS.save:
    file_root = "sims_"
    file_name = file_root + FLAGS.model_name + "_" + str(len(fnmatch.filter(os.listdir(FLAGS.output_dir), file_root + FLAGS.model_name + "*"))) + ".fits"
    file_path = os.path.join(FLAGS.output_dir, file_name)
    #np.save(file_path, sims)
    utils.save_sims(file_path, sims, true_shear, sigma_n)

  if FLAGS.plot:
    sims_reshaped = sims.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
    
    sep = dict(color='k', linestyle=':', linewidth=.5)
    plt.figure()
    for i in range(N):
      if i>0:
        plt.axvline(x=i*stamp_size, **sep)
        plt.axhline(y=i*stamp_size, **sep)
        plt.axvline(x=i*stamp_size, **sep)
        plt.axhline(y=i*stamp_size, **sep)
    
    plt.imshow(np.arcsinh(sims_reshaped/sigma_n)*sigma_n, cmap='gray_r')
    plt.title(r'$Arcsinh(\frac{X}{\sigma})\cdot \sigma$')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
  app.run(main)