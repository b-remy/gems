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

N = 10 # number of stamp in a row/col

flags.DEFINE_integer("n", 10, "number of interations")
flags.DEFINE_string("samples_path", None, "number of interations")
flags.DEFINE_boolean("MAP", False, "number of interations")
FLAGS = flags.FLAGS

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
  folder_name = 'dgm_hmc'
  job_name = str(int(time.time()))
  if not os.path.isdir('./res'):
    os.mkdir('res')
    os.mkdir('res/'+folder_name)
  elif not os.path.isdir('./res/'+folder_name):
    os.mkdir('res/'+folder_name)
    
  os.mkdir("res/"+folder_name+"/{}".format(job_name))
  os.mkdir("res/"+folder_name+"/{}/params".format(job_name))

  begin = time.time()

  num_gal = N*N
  
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
  interp_factor=2
  padding_factor=1

  while len(obs) < num_gal:
    galp = cat.makeGalaxy(ind, gal_type='parametric')
    if cat.param_cat['use_bulgefit'][cat.orig_index[ind]] == 0:
      if galp.original.n < 0.4 or galp.original.half_light_radius > .3:
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
          galr.shear(g1=0.05, g2=-0.05)
          conv = galsim.Convolve(galr, psf)

          # Add Gaussian noise
          img = conv.drawImage(nx=stamp_size, ny=stamp_size, scale=im_scale, method='no_pixel')
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
                                      interp_factor = 2,
                                      padding_factor = 1,
                                      mag_auto_list=mag_auto_list, 
                                      z_phot_list=z_phot_list, 
                                      flux_radius_list=flux_radius_list,
                                      fit_centroid=False))

  scale_gamma = .1
  
  def target_log_prob_fn(prior_z, gamma):
    return log_prob(
        prior_z=prior_z,
        gamma=gamma*scale_gamma,
        obs=obs)

  def loss_fn(lz, gamma):
    return - target_log_prob_fn(lz, gamma)
  
  if FLAGS.samples_path:
    lz =  tf.convert_to_tensor(np.load(FLAGS.samples_path + 'latent_z.npy'), tf.float32)
    gamma = tf.convert_to_tensor(np.load(FLAGS.samples_path + 'gamma.npy'), tf.float32)
    # shift = tf.convert_to_tensor(np.load(FLAGS.samples_path + 'samples_shift.npy'), tf.float32)
  else:
    lz = tf.Variable(tf.zeros([batch_size, num_gal,16]), trainable=True, dtype=tf.float32)
    gamma = tf.Variable(tf.zeros((batch_size, 2)), trainable=True, dtype=tf.float32)
    # shift = tf.Variable(tf.zeros((batch_size, num_gal,2)), trainable=True, dtype=tf.float32)


  #########
  # Run HMC
  #########

  num_results = FLAGS.n
  num_burnin_steps = 1

  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    num_leapfrog_steps=3,
    # step_size=.00005)
    # step_size=.0001)
    step_size=.006)

  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
                      lz,
                      gamma/scale_gamma,
                      
        ],
        kernel=adaptive_hmc)
    return samples, trace
  
  samples, trace = get_samples()

  end = time.time()
  print('Time: {:.2f}'.format((end - begin)/60.))
  print('')

  lz_est = samples[0][:,0,:]
  gamma_est = samples[1][:,0,:]*scale_gamma
  gamma_true = true_gamma[0,:]

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    lz_est, gamma_est, trace, gamma_true = sess.run([lz_est, gamma_est, trace, gamma_true])

  print('accptance ratio:', trace.is_accepted.sum()/len(trace.is_accepted))
  
  # Save results
  np.save("res/"+folder_name+"/"+job_name+"/samples_gamma_hmc.npy", gamma_est)
  np.save("res/"+folder_name+"/"+job_name+"/samples_lz_hmc.npy", lz_est)

  print(lz_est.shape)


  ## Save chains plots

  # Shear chains
  plt.figure()
  plt.plot(gamma_est)
  plt.axhline(gamma_true[0], color='C0', label='g1')
  plt.axhline(gamma_true[1], color='C1', label='g2')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/shear.png")

  # Latent z chains
  plt.figure(figsize=[16, 7])
  for k in range(6):
    plt.title('latent_var ({})'.format(k))
    plt.subplot(2,3,k+1)
    for i in range(5):
      plt.plot(lz_est[:,i,k], label='{}'.format(i))
    plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/latent_z.png")

if __name__ == "__main__":
    app.run(main)

