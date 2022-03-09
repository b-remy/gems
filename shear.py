import tensorflow as tf
import edward2 as ed
from utils import get_ps_map, ks93inv
from functools import partial

def latent_to_shear(z, map_width, resolution):
  
  # rescale convergence in Fourier with the right power spectrum
  tf_z = tf.signal.fft2d(tf.cast(z, tf.complex64))

  rescale = tf_z * tf.cast(tf.math.sqrt(get_ps_map(map_width, resolution)), tf.complex64)

  k = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(rescale)))

  # Kaiser-Squires
  g1, g2 = ks93inv(k, tf.zeros_like(k))
  
  return g1, g2

def shear_map(batch_size, map_width, resolution, name="latent_shear"):
  """
  squared shear map
  """

  # latent variable, ps of z is 1
  z = ed.Normal(loc=tf.zeros((batch_size, map_width, map_width)), scale=1., name=name)
  return tf.stack(latent_to_shear(z, map_width, resolution), axis=-1)

if __name__=='__main__':
  import matplotlib.pyplot as plt
  '''
  g1, g2 = shear_map(map_width=16, resolution=10.)
  plt.subplot(121)
  plt.imshow(g1)
  plt.colorbar()
  plt.subplot(122)
  plt.imshow(g2)
  plt.colorbar()
  plt.savefig('shear.png')
  '''

  from models import varying_shear_gaussian_model
  # shear_field(batch_size=1, num_gal=25, stamp_size=64, scale=0.03, fixed_flux=False)
  
  stamp_size = 64
  N = 8
  batch_size = 1

  # Execute probabilistic program and record execution trace
  with ed.tape() as true_params:
    # ims =  partial(gaussian_model, fixed_flux=True)(batch_size=N*N, stamp_size=stamp_size)
    ims =  partial(varying_shear_gaussian_model, fixed_flux=True)(stamp_size=stamp_size)
  
  res = ims.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  
  plt.figure()
  plt.imshow(res, cmap='gray_r')
  plt.savefig('res/gals.png')

  


