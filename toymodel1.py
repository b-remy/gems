import edward2 as ed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absl import app

import galflow
lp = galflow.lightprofiles

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi

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

  # prior on intensity
  #log_l_I = ed.Normal(loc=-1.97, scale=.53)
  #I = tf.math.exp(log_l_I * _log10)

  # Flux
  F = 16.693710205567005
  #b_n = lp.calculate_b(n)
  #F = 2 * _pi * tf.math.exp(tf.math.lgamma(2*n)) * tf.math.exp(b_n) * hlr  * hlr * I / tf.math.pow(b_n, 2*n)

  # generate light profile
  profile = lp.sersic(n, scale_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)
  image = profile + noise
  return image

def main(_):
  stamp_size = 55
  N = 5
  sigma_e = 0.003
  plot = np.zeros((N*stamp_size, N*stamp_size))
  
  sep = dict(color='k', linestyle=':', linewidth=.5)
  plt.figure()
  for i in range(N):
    if i>0:
      plt.axvline(x=i*stamp_size, **sep)
      plt.axhline(y=i*stamp_size, **sep)
    for j in range(N):
      if i>0:
        plt.axvline(x=i*stamp_size, **sep)
        plt.axhline(y=i*stamp_size, **sep)
      plot[i*stamp_size:(i+1)*stamp_size, j*stamp_size:(j+1)*stamp_size] = np.arcsinh(model(stamp_size)/sigma_e)*sigma_e
  
  plt.imshow(plot, cmap='Greys')
  plt.title(r'$Arcsinh(\frac{X}{\sigma})\cdot \sigma$')
  plt.colorbar()
  plt.show()

if __name__ == "__main__":
  app.run(main)