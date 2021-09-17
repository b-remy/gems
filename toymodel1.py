import edward2 as ed
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
import fnmatch
from absl import app
from absl import flags

import galsim
import galflow
lp = galflow.lightprofiles

flags.DEFINE_integer("N", 5, "Number of stamps on x and y axes")
flags.DEFINE_boolean("plot", False, "Should we plot the simulations?")
flags.DEFINE_string("prior_path", "toymodel1_prior.txt", "Path to prior parameters")
flags.DEFINE_string("output_dir", "./data", "Path to output simulations")
flags.DEFINE_string("model_name", "toymodel1", "Name of the probabilistic model")
flags.DEFINE_boolean("save", True, "Should we store the simulations?")

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi

FLAGS = flags.FLAGS

def model(stamp_size):
  """Toy model
  """
  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_e = 0.003
  noise = ed.Normal(loc=tf.zeros((nx, ny)), scale=sigma_e)

  # prior on Sersic index n
  log_l_n = ed.Normal(loc=.1, scale=.39)
  n = tf.math.exp(log_l_n * _log10)

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68, scale=.3)
  hlr = tf.math.exp(log_l_hlr * _log10)

  # prior on shear
  gamma = ed.Normal(loc=tf.zeros((2)), scale=.09)

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

  # Add noise
  image = profile + noise
  return image

def main(_):
  stamp_size = 56
  N = FLAGS.N
  sigma_e = 0.003
  sims = np.zeros((N*stamp_size, N*stamp_size))
  
  for i in range(N):
    for j in range(N):
      sims[i*stamp_size:(i+1)*stamp_size, j*stamp_size:(j+1)*stamp_size] = model(stamp_size)

  if FLAGS.save:
    file_root = "sims_"
    file_name = file_root + FLAGS.model_name + "_" + str(len(fnmatch.filter(os.listdir(FLAGS.output_dir), file_root + FLAGS.model_name + "*"))) + ".npy"
    file_path = os.path.join(FLAGS.output_dir, file_name)
    print(os.path.dirname(FLAGS.output_dir))
    print(file_path)
    np.save(file_path, sims)

  if FLAGS.plot:
    sep = dict(color='k', linestyle=':', linewidth=.5)
    plt.figure()
    for i in range(N):
      if i>0:
        plt.axvline(x=i*stamp_size, **sep)
        plt.axhline(y=i*stamp_size, **sep)
        plt.axvline(x=i*stamp_size, **sep)
        plt.axhline(y=i*stamp_size, **sep)
    
    plt.imshow(np.arcsinh(sims/sigma_e)*sigma_e, cmap='Greys')
    plt.title(r'$Arcsinh(\frac{X}{\sigma})\cdot \sigma$')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
  app.run(main)