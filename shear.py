import tensorflow as tf
import edward2 as ed
from utils import get_ps_map, ks93inv

def shear_map(map_width, resolution):
  """
  squared shear map
  """

  # latent variable
  z = ed.Normal(loc=tf.zeros((map_width, map_width)), scale=1., name="latent_shear")
  tf_z = tf.signal.fft2d(tf.cast(z, tf.complex64))
  rescale = tf_z * get_ps_map(map_width, resolution)

  k = tf.signal.ifft2d(rescale)

  shear = ks93inv(k, tf.zeros_like(k))
  return shear

if __name__=='__main__':
  import matplotlib.pyplot as plt

  g1, g2 = shear_map(map_width=4, resolution=10.)
  plt.subplot(121)
  plt.imshow(g1)
  plt.subplot(122)
  plt.imshow(g2)
  plt.colorbar()
  plt.savefig('shear.png')

  


