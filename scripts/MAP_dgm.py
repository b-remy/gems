# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from absl import app
from absl import flags
import time
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability import edward2 as ed

from gems.models import sersic2morph_model, dgm2morph_model
from gems.ed_utils import make_value_setter, make_log_joint_fn

from tqdm import tqdm

import numpy as np
from functools import partial

import galsim
from galsim.bounds import _BoundsI

import matplotlib.pyplot as plt

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
stamp_size = 128
noise_level = 0.01

N = 12 # number of stamp in a row/col

def gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale):
  Nk = stamp_size*interp_factor*padding_factor
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                        scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                        recenter=False)
  imkpsf = tf.signal.fftshift(tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk//2+1]), axes=1)
  return imkpsf

def main(_):
  # Prepare results storage
  folder_name = 'dgm_MAP'
  job_name = str(int(time.time()))
  if not os.path.isdir('./res'):
    os.mkdir('res')
    os.mkdir('res/'+folder_name)
  elif not os.path.isdir('./res/'+folder_name):
    os.mkdir('res/'+folder_name)
    
  os.mkdir("res/"+folder_name+"/{}".format(job_name))
  os.mkdir("res/"+folder_name+"/{}/params".format(job_name))


  num_gal = N*N
  #cat = galsim.COSMOSCatalog(sample='23.5', dir='/gpfswork/rech/xdy/commun/galsim_catalogs/COSMOS_23.5_training_sample')
  cat = galsim.COSMOSCatalog(dir='/gpfswork/rech/xdy/commun/galsim_catalogs/COSMOS_25.2_training_sample')
  #cat = galsim.COSMOSCatalog(dir='/local/home/br263581/miniconda3/envs/gems/lib/python3.6/site-packages/galsim/share/COSMOS_25.2_training_sample')
  index = range(N*N)

  # Prepare parameters
  obs = []
  n = []
  flux = []
  hlr = []
  q_list = []
  phi_list = []
  psfs = []

  mag_auto_list = []
  z_phot_list = []
  flux_radius_list = []

  ind = 0
  ind_ = 0

  # PSF parameters
  im_scale = 0.03
  interp_factor=1
  padding_factor=1

  while len(obs) < num_gal:
    galp = cat.makeGalaxy(ind, gal_type='parametric')
    if cat.param_cat['use_bulgefit'][cat.orig_index[ind]] == 0:
      if galp.original.n < 0.4 or galp.original.half_light_radius > .5 or cat.param_cat['mag_auto'][cat.orig_index[ind]] > 22.5 or cat.param_cat['mag_auto'][cat.orig_index[ind]] < 22. or ind==2020:
        ind += 1
      else:
        if False:#ind_==6 or ind_==93 or ind_==56 or ind_==55:
          ind+=1
          ind_+=1
        else:
          galr = cat.makeGalaxy(ind, gal_type='real')
          galp = cat.makeGalaxy(ind, gal_type='parametric')
          psf = galr.original_psf
          imkpsf = gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale)

          # Store sersic fit parameters
          n.append(galp.original.n)
          flux.append(galp.original.flux)
          hlr.append(galp.original.half_light_radius)
          q = cat.param_cat['sersicfit'][cat.orig_index[ind]][3]
          q_list.append(q)
          phi = cat.param_cat['sersicfit'][cat.orig_index[ind]][7]
          phi_list.append(phi)
          psfs.append(imkpsf)
          
          mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
          z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
          flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

          # Apply shear
          galr = galr.shear(g1=0.05, g2=-0.05)
          conv = galsim.Convolve(galr, psf)

          # Add Gaussian noise
          img = conv.drawImage(nx=stamp_size, ny=stamp_size, scale=im_scale)
          seed = ind
          generator = galsim.random.BaseDeviate(seed=seed)
          g_noise = galsim.GaussianNoise(rng=generator, sigma=noise_level)
          img.addNoise(g_noise)
          obs_ = tf.convert_to_tensor(img.array)
          obs.append(obs_)

          ind_ += 1
          ind += 1
    else:
      ind += 1

  obs_64 = tf.expand_dims(tf.stack(obs, axis=0), 0) # [1, batch, nx, ny]
  obs = tf.expand_dims(tf.stack(obs, axis=0), 0)[..., 10:-10, 10:-10] # [1, batch, nx, ny]
  n = tf.expand_dims(tf.stack(n, axis=0), 0)
  flux = tf.expand_dims(tf.stack(flux, axis=0), 0)
  hlr = tf.expand_dims(tf.stack(hlr, axis=0), 0)
  q_ = tf.expand_dims(tf.stack(q_list, axis=0), 0)
  phi = tf.expand_dims(tf.stack(phi_list, axis=0), 0)
  e1 = tf.cast((1-q_)/(1+q_) * tf.math.cos(2*phi), tf.float32)
  e2 = tf.cast((1-q_)/(1+q_) * tf.math.sin(2*phi), tf.float32)
  imkpsfs = tf.concat(psfs, axis=0)

  # Ground truth parameters
  true_hlr = hlr
  true_e = tf.stack([e1, e2], -1)
  true_gamma = tf.expand_dims(tf.convert_to_tensor([0.05, -0.05]), 0)

  #TODO: does not work with batch size > 1 yet...
  batch_size = 1

  log_prob = make_log_joint_fn(partial(dgm2morph_model, 
                                      batch_size=batch_size, 
                                      sigma_e=noise_level, 
                                      stamp_size=stamp_size,
                                      num_gal=N*N, 
                                      kpsf=imkpsfs, 
                                      interp_factor = interp_factor,
                                      padding_factor = padding_factor,
                                      mag_auto_list=mag_auto_list, 
                                      z_phot_list=z_phot_list, 
                                      flux_radius_list=flux_radius_list,
                                      fit_centroid=False))

  scale_gamma = 1.

  def target_log_prob_fn(prior_z, gamma):
    return log_prob(
        prior_z=prior_z,
        gamma=gamma*scale_gamma,
        obs=obs)

  def loss_fn(lz, gamma):
    return - target_log_prob_fn(lz, gamma)  

  # Initialize variable
  lz = tf.Variable(tf.zeros([batch_size, num_gal,16]), trainable=True, dtype=tf.float32)
  gamma = tf.Variable(tf.zeros([batch_size, 2]), trainable=True, dtype=tf.float32)

  # Evaluate loss
  loss = loss_fn(lz, gamma)
  
  # Define the optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
  train = optimizer.minimize(loss, var_list=[lz, gamma])

  # initialize the variables
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)

  losses = []
  for i in tqdm(range(200)):
      _, l, lz_, g_ = sess.run([train, loss, lz, gamma])
      losses.append(l)

  print('g_MAP:', g_*scale_gamma)

  plt.figure()
  plt.plot(losses)
  plt.savefig("res/"+folder_name+"/"+job_name+"/losses.png")


  observations = sess.run(obs_64)
  im_rec1 = observations.reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])

  plt.figure(figsize=(11,11))
  plt.title('Real galaxies (COSMOS)')
  s = 1e-3
  plt.imshow(np.arcsinh(im_rec1/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/obs.png")


  with ed.interception(make_value_setter(prior_z=lz, gamma=gamma*scale_gamma)):
    res_fit = dgm2morph_model(batch_size=batch_size, 
                                 sigma_e=0.,#noise_level, 
                                 stamp_size=stamp_size,
                                 num_gal=N*N, 
                                 kpsf=imkpsfs, 
                                 interp_factor = interp_factor,
                                 padding_factor = padding_factor,
                                 mag_auto_list=mag_auto_list, 
                                 z_phot_list=z_phot_list, 
                                 flux_radius_list=flux_radius_list,
                                 fit_centroid=False, display=True)

  res_fit_ = sess.run(res_fit)
  im_res_fit = res_fit_.reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])

  plt.figure(figsize=(11,11))
  plt.title('Real galaxies (COSMOS)')
  s = 1e-3
  plt.imshow(np.arcsinh(im_res_fit/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/obs.png")


  plt.figure(figsize=(11,11))
  plt.title('Redisuals (COSMOS)')
  s = 1e-3
  plt.imshow(np.arcsinh((im_res_fit-im_rec1)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/residuals.png")
  
  np.save("res/"+folder_name+"/"+job_name+"/latent_z", lz_)
  np.save("res/"+folder_name+"/"+job_name+"/gamma", g_*scale_gamma)

if __name__ == "__main__":
    app.run(main)
