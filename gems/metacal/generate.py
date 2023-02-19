import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability import edward2 as ed
import numpy as np

import numpy as np

from functools import partial

from gems.models import sersic2morph_model, dgm2morph_model, shear_fourier, convolve_fourier, dgm_model
from gems.ed_utils import make_value_setter, make_log_joint_fn

encoder = hub.Module('../deep_galaxy_models/modules/vae_16/encoder')
decoder = hub.Module('../deep_galaxy_models/modules/vae_16/decoder')

import galsim
from galsim.bounds import _BoundsI

cat = galsim.COSMOSCatalog(dir='/Users/br263581/miniconda3/envs/gems/lib/python3.6/site-packages/galsim/share/COSMOS_25.2_training_sample')

# Global setting for our autoencoded galaxies

PIXEL_SCALE = 0.03
STAMP_SIZE = 128
interp_factor=2
padding_factor=1
Nk = STAMP_SIZE*interp_factor*padding_factor
bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

def gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale):
  Nk = stamp_size*interp_factor*padding_factor
  bounds = _BoundsI(-Nk//2, Nk//2-1, -Nk//2, Nk//2-1)
  imkpsf = psf.drawKImage(bounds=bounds,
                    scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                    recenter=False)
  # imkpsf = tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk])
  imkpsf = imkpsf.array.reshape([1, Nk, Nk])
  return imkpsf

def gen_batch(batch_size=2, start_index=0, shear=[0., 0.], noise_level=0.01):
  """
  Util funnction generating batch of autoencoded 
  batch_size: must be a pair intege (every galaxy is doubled to cancel intrinsic shear)
  start_index: index from which to start looking for galaxies in galsim catalog
  shear: shear to apply to the galaxies
  """

  NUM_GAL = batch_size

  im_real_list = []
  im_psf_list = []
  psfs = []
  psfs_ngmix = []

  mag_auto_list = []
  z_phot_list = []
  flux_radius_list = []

  indices = []
  degrees = galsim.AngleUnit(np.pi / 180.)
  angle = galsim.Angle(90, unit=degrees)

  ind = start_index

  while len(im_real_list) < NUM_GAL:
    galp = cat.makeGalaxy(ind, gal_type='parametric')
    if cat.param_cat['use_bulgefit'][cat.orig_index[ind]] == 0:
      
      # CUTS
      if galp.original.n < 0.4 or galp.original.half_light_radius > .2 or cat.param_cat['mag_auto'][cat.orig_index[ind]] < 23.5 or ind < start_index:
        ind += 1
      else:
        galp = cat.makeGalaxy(ind, gal_type='parametric')
        im_real = galsim.ImageF(STAMP_SIZE, STAMP_SIZE, scale=PIXEL_SCALE)
        galr= cat.makeGalaxy(ind, gal_type='real', noise_pad_size=0.8*PIXEL_SCALE*STAMP_SIZE)
        psf = galr.original_psf

        if indices.count(ind)==1:
          galr = galr.rotate(angle)
          psf = psf.rotate(angle)

        real = galsim.Convolve(psf, galr)
        real.drawImage(im_real, method='no_pixel', use_true_center=False)

        # PSF for the autocoder
        imCp = psf.drawKImage(bounds=bounds,
                                scale=2.*np.pi/(Nk * PIXEL_SCALE / interp_factor),
                                recenter=False)
        im_psf = np.abs(np.fft.fftshift(imCp.array, axes=0)).astype('float32')

        # PSF for reconvolution
        imkpsf = gpsf2ikpsf(psf=psf, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)
        psfs.append(imkpsf)

        psfs_ngmix.append(psf.drawImage(scale=PIXEL_SCALE).array)

        im_real_list.append(im_real.array)
        im_psf_list.append(im_psf)
        indices.append(ind)

        mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
        z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
        flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

        print(ind, len(im_real_list))

        if indices.count(ind)==2:
          ind += 1

    else:
      ind += 1

  im_real_list = np.stack(im_real_list, axis=0)
  im_psf_list = np.stack(im_psf_list, axis=0)
  imkpsfs = tf.cast(tf.concat(psfs, axis=0), tf.complex64)
  
  psf_in = tf.placeholder(shape=[NUM_GAL, 256, 129, 1], dtype=tf.float32)
  im_in = tf.placeholder(shape=[NUM_GAL, 128, 128, 1], dtype=tf.float32)

  code = encoder({'input':im_in, 'psf':psf_in})
  reconstruction = decoder(code)

  ims = tf.reshape(reconstruction, (1, NUM_GAL, STAMP_SIZE, STAMP_SIZE))

  g1 = shear[0]
  g2 = shear[1]
  im_sheared = shear_fourier(ims, g1, g2)

  ims = convolve_fourier(im_sheared, imkpsfs)

  obs = ims + tf.random_normal([1, NUM_GAL, STAMP_SIZE, STAMP_SIZE]) * noise_level

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  y = sess.run(obs, feed_dict={psf_in:im_psf_list.reshape((NUM_GAL,256,129,1)), im_in:im_real_list.reshape((NUM_GAL,128,128,1))})

  return y[0], psfs_ngmix


if __name__=="__main__":
  N = 4
  batch_size = N*N
  y = gen_batch(batch_size=batch_size, start_index=0, shear=[0.01, 0.], noise_level=0.01)

  print(y.shape)

  import matplotlib.pyplot as plt
  im_obs = y.reshape(N,N,STAMP_SIZE,STAMP_SIZE).transpose([0,2,1,3]).reshape([N*(STAMP_SIZE),N*(STAMP_SIZE)])

  plt.figure(figsize=[20,20])
  plt.imshow(im_obs)#, vmax=0.1)
  plt.colorbar()
  plt.show()