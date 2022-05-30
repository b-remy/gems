import galsim
import tensorflow as tf
import numpy as np
from galsim.bounds import _BoundsI

_log10 = tf.math.log(10.)
_pi = np.pi

## Gaussian PSF model
def get_gaussian_psf(scale, stamp_size, interp_factor, padding_factor):
  psf = galsim.Gaussian(0.06)

  interp_factor=1
  padding_factor=1
  Nk = stamp_size*interp_factor*padding_factor
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                        scale=2.*_pi/(stamp_size*padding_factor*scale),
                        recenter=False)

  return tf.cast(np.fft.fftshift(imkpsf.array.reshape(1, Nk, Nk//2+1), axes=1), tf.complex64)

def get_cosmos_psf(stamp_size, _scale):
  catalog = galsim.COSMOSCatalog()
  gal = catalog.makeGalaxy(0, gal_type='real', noise_pad_size=stamp_size * _scale*2)
  psf = gal.original_psf
  N = stamp_size
  interp_factor=2
  padding_factor=2
  Nk = N*interp_factor*padding_factor
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)
  imkpsf = psf.drawKImage(bounds=bounds,
                    scale=2.*np.pi/(N*padding_factor*_scale),
                    recenter=False)
  return tf.convert_to_tensor(np.fft.fftshift(imkpsf.array.reshape(1,Nk,Nk//2+1), axes=1), tf.complex64)