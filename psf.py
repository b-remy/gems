import galsim
import tensorflow as tf
import numpy as np

_log10 = tf.math.log(10.)
_pi = np.pi

## Gaussian PSF model
def get_gaussian_psf(scale, stamp_size, interp_factor, padding_factor):
  psf = galsim.Gaussian(0.06)

  interp_factor=1
  padding_factor=1
  Nk = stamp_size*interp_factor*padding_factor
  from galsim.bounds import _BoundsI
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                        scale=2.*_pi/(stamp_size*padding_factor*scale),
                        recenter=False)

  return tf.cast(np.fft.fftshift(imkpsf.array.reshape(1, Nk, Nk//2+1), axes=1), tf.complex64)