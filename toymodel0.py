"""
Toy model 0

Matching ngmix.examples.metacal.metacal.py data
https://github.com/esheldon/ngmix/blob/master/examples/metacal/metacal.py

# Exponential light profiles
Gaussian light profiles
  - constant shear
  - constant size
  - no intrinsic e

Moffat PSF
"""

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
flags.DEFINE_string("model_name", "toymodel0", "Name of the probabilistic model")
flags.DEFINE_boolean("save", False, "Should we store the simulations?")
flags.DEFINE_float("sigma_n", 1e-6, "Level of noise in data")

_log10 = tf.math.log(10.)
_scale = 0.263
_pi = np.pi

FLAGS = flags.FLAGS

_stamp_size = 44

# PSF model from galsim COSMOS catalog
# cat = galsim.COSMOSCatalog()
# psf = cat.makeGalaxy(2,  gal_type='real', noise_pad_size=0).original_psf

psf_fwhm = 0.9
psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.01,
    )

psf = galsim.Convolve(psf, galsim.Pixel(_scale))

interp_factor=1
padding_factor=1
Nk = _stamp_size*interp_factor*padding_factor
from galsim.bounds import _BoundsI
bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

imkpsf = psf.drawKImage(bounds=bounds,
                        scale=2.*_pi/(_stamp_size*padding_factor*_scale),
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

  # constant hlr
  hlr = 0.5 * tf.ones(batch_size)

  # Flux
  F = tf.ones(batch_size)

  # Generate light profile
  profile = lp.exponential(half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)
  # profile = lp.gaussian(half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)
  ims = tf.cast(tf.reshape(profile, (batch_size,stamp_size,stamp_size,1)), tf.float32)

  # constant shear
  gamma = tf.convert_to_tensor(shear, dtype=tf.float32)
  gamma = tf.repeat(tf.expand_dims(gamma, 0), batch_size, axis=0)
  # Shear the image
  tfg1 = gamma[:, 0]
  tfg2 = gamma[:, 1]
  ims = galflow.shear(ims, tfg1, tfg2)
  
  # Convolve the image with the PSF
  # profile = galflow.convolve(ims, kpsf,
  #                     zero_padding_factor=padding_factor,
  #                     interp_factor=interp_factor)[...,0]
  # imk = tf.signal.rfft2d(ims[...])
  # # Performing k space convolution
  # imconv = galflow.kconvolve(imk, kpsf,
  #                 zero_padding_factor=1,
  #                 interp_factor=interp_factor)

  # profile = tf.image.resize_with_crop_or_pad(imconv, _stamp_size, _stamp_size)[...,0]
  
  # imk = tf.signal.rfft2d(ims[...,0])

  # print(imk.shape)
  # print()
  # print()
  # print()
  # # imk = tf.cast(tf.signal.fftshift(tf.reshape(imk, (batch_size, Nk, Nk//2+1)), axes=1), tf.complex64)
  # imk = tf.cast(tf.reshape(imk, (batch_size, Nk, Nk//2+1)), tf.complex64)

  # imk = imk * kpsf
  # # im = tf.signal.fftshift(tf.signal.irfft2d(imk, [_stamp_size*padding_factor, _stamp_size*padding_factor]))
  # im = tf.signal.irfft2d(imk, [_stamp_size*padding_factor, _stamp_size*padding_factor])
  
  # im = tf.image.resize_with_crop_or_pad(im[...,tf.newaxis], _stamp_size, _stamp_size)[...,0]

  ims = tf.cast(ims[...,0], tf.complex64)

  im_psf = psf.drawImage(nx=_stamp_size, ny=_stamp_size, scale=_scale, use_true_center=False, method='no_pixel').array
  im_psf = tf.expand_dims(tf.cast(im_psf, tf.complex64), axis=0)

  def convolve_tf(im_gal, im_psf):
    """
    im_psf = psf.drawImage(nx=_stamp_size, ny=_stamp_size, scale=_scale, use_true_center=False, method='no_pixel')
    im_gal = obj0.drawImage(nx=_stamp_size, ny=_stamp_size, scale=_scale, method='no_pixel')
    """
    im_gal_k = tf.signal.fft2d(im_gal)
    im_psf_k = tf.signal.fft2d(im_psf)
    # Fourier-based convolution
    im_conv_k = im_gal_k * (im_psf_k)

    # Inverse FFT
    im = tf.signal.fftshift(tf.math.real(tf.signal.ifft2d(im_conv_k)), axes=(1,2))
    return im

  profile = convolve_tf(ims, im_psf)

  # Add noise
  image = profile + noise
  return image

def main(_):
  stamp_size = _stamp_size
  N = FLAGS.N
  batch_size = N*N
  sigma_n = FLAGS.sigma_n
  true_shear = [0.01, 0.0]

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