from absl import app
from absl import flags
import os

import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from functools import partial
from gems.ed_utils import make_log_joint_fn

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import galsim
import galflow
lp = galflow.lightprofiles
import time

from gems.models import gaussian_model, sersic_model, sersic2morph_model, deep2morph_model
from galsim.bounds import _BoundsI

flags.DEFINE_integer("n", 10, "number of interations")
FLAGS = flags.FLAGS

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
stamp_size = 128
noise_level = 0.01

N = 10 # number of stamp in a row/col

def gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale):
  Nk = stamp_size*interp_factor*padding_factor
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                        scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                        recenter=False)
  imkpsf = tf.signal.fftshift(tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk//2+1]), axes=1)
  return imkpsf

def main(_):
  begin = time.time()

  num_gal = N*N

  # Load galaxies from galsim COSMOS catalog
  cat = galsim.COSMOSCatalog(sample='25.2')

  # Prepare parameters
  obs = []
  n = []
  flux = []
  hlr = []
  q_list = []
  phi_list = []
  psfs = []
  mag_auto = []
  zphot = []
  flux_radius = []

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

          # Apply shear
          #galr.shear(g1=0.05, g2=-0.05)
          conv = galsim.Convolve(galr, psf)
          
          # Add Gaussian noise
          img = conv.drawImage(nx=stamp_size, ny=stamp_size, scale=im_scale, method='no_pixel')
          seed = ind
          generator = galsim.random.BaseDeviate(seed=seed)
          g_noise = galsim.GaussianNoise(rng=generator, sigma=noise_level)
          img.addNoise(g_noise)
          obs_ = tf.convert_to_tensor(img.array)
          obs.append(obs_)

          # Add conditional params
          mag_auto.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
          zphot.append(cat.param_cat['zphot'][cat.orig_index[ind]])
          flux_radius.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

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

  mag_auto = tf.reshape(tf.convert_to_tensor(mag_auto, tf.float32), [num_gal, ])
  zphot = tf.reshape(tf.convert_to_tensor(zphot, tf.float32), [num_gal, ])
  flux_radius = tf.reshape(tf.convert_to_tensor(flux_radius, tf.float32), [num_gal, ])
  batch_params = {'mag_auto': mag_auto, 'zphot': zphot, 'flux_radius':flux_radius}


  # Ground truth parameters
  true_hlr = hlr
  true_e = tf.stack([e1, e2], -1)
  true_gamma = tf.expand_dims(tf.convert_to_tensor([0.05, -0.05]), 0)


  # Prepare results storage
  # folder_name = 'cosmos_real'
  folder_name = 'deep_morpho'
  job_name = str(int(time.time()))
  os.mkdir("res/"+folder_name+"/{}".format(job_name))
  os.mkdir("res/"+folder_name+"/{}/params".format(job_name))

  res = obs_64.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  plt.figure(figsize=(11,11))
  plt.title('Real galaxies (COSMOS)')
  s = 1e-3
  plt.imshow(np.arcsinh(res/s)*s)
  plt.savefig("res/"+folder_name+"/"+job_name+"/gals.png")

  '''
  # saving true params for later comparison
  np.save("res/"+folder_name+"/"+job_name+"/params/gals.npy", ims.numpy())
  np.save("res/"+folder_name+"/"+job_name+"/params/shear.npy", true_params['gamma'].numpy())
  '''

  np.save("res/"+folder_name+"/"+job_name+"/params/e.npy", true_e.numpy())
  np.save("res/"+folder_name+"/"+job_name+"/params/hlr.npy", true_gamma.numpy())

  # Get the joint log prob
  batch_size = 1

  # zero shear, only fitting the morphology
  log_prob = make_log_joint_fn(partial(deep2morph_model, batch_size=batch_size, sigma_e=noise_level, num_gal=N*N, kpsf=imkpsfs, batch_params=batch_params, gamma=tf.zeros([batch_size, 2]), fit_centroid=False))

  def target_log_prob_fn(z):
    return log_prob(
           #gamma=gamma*scale_gamma, # trick to adapt the step size
           #e=e*scale_e,
           #shift=shift*scale_shift,
           latent_z = z,
           obs=obs)

  # Initialize latent variable
  z_init = tf.Variable(tf.zeros([batch_size, num_gal, 16]))

  # Compute the MAP
  #@tf.function
  def loss(z):
    return - target_log_prob_fn(z)



  '''
  num_results = FLAGS.n
  num_burnin_steps = 1

  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    num_leapfrog_steps=3,
    # step_size=.00005)
    # step_size=.0001)
    step_size=.002)

  @tf.function
  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
          # hlr, # init with prior mean
                      # true_hlr,
                      # tf.zeros((1, 2)), # init with zero shear
                      # start_gamma,
                      true_gamma/scale_gamma,
                      # tf.zeros((1, num_gal, 2)), # init with zero ellipticity
                      # start_e,
                      true_e,
                      true_e*0., # offset shift init to 0.
                      
        ],
        kernel=adaptive_hmc)
    return samples, trace
  
  samples, trace = get_samples()
  print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  end = time.time()
  print('Time: {:.2f}'.format((end - begin)/60.))
  print('')

  # hlr_est = samples[0].numpy()[:,0,:]
  gamma_est = samples[0].numpy()[:,0,:]*scale_gamma
  e_est = samples[1].numpy()[:,0,:]*scale_e
  shift_est = samples[2].numpy()[:,0,:]*scale_shift
  gamma_true = true_gamma.numpy()[0,:]
  
  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_gamma_{}_{}.npy".format(N*N, num_results, gamma_true[0], gamma_true[1]), gamma_est)
  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_e.npy".format(N*N, num_results), e_est)
  # np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_r.npy".format(N*N, num_results), hlr_est)
  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_shift.npy".format(N*N, num_results), shift_est)

  plt.figure()
  plt.plot(gamma_est)
  plt.axhline(gamma_true[0], color='C0', label='g1')
  plt.axhline(gamma_true[1], color='C1', label='g2')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/shear.png")


  plt.figure()
  plt.subplot(121)
  plt.title('e1')
  for i in range(5):
    # print(true_e.numpy().shape)
    # print(true_e.numpy()[0,i,0])
    plt.plot(e_est[:,i,0], label='{}'.format(i))
    plt.axhline(true_e.numpy()[0,i,0], color='gray')
  plt.legend()

  print(np.where((e_est[:,:,0]-true_e.numpy()[:,:,0]) > 0.02))

  plt.subplot(122)
  plt.title('e2')
  for i in range(5):
    plt.plot(e_est[:,i,1], label='{}'.format(i))
    plt.axhline(true_e.numpy()[0,i,1], color='gray')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/e.png")

  plt.figure()
  plt.title('offset x')
  for i in range(5):
    plt.plot(shift_est[:,i,0], label='{}'.format(i))
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/shift.png")
  
  with ed.condition(e=e_est.mean(axis=0, keepdims=True),
                    gamma=gamma_est.mean(axis=0, keepdims=True),
                    shift=shift_est.mean(axis=0, keepdims=True)):
    rec1 = sersic2morph_model(batch_size=batch_size, 
                              sigma_e=noise_level,
                              num_gal=N*N, 
                              kpsf=imkpsfs, 
                              fixed_flux=True, 
                              n=n, flux=flux, hlr=hlr, 
                              fit_centroid=True, display=True)
  im_rec1 = rec1.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  
  plt.figure(figsize=(11,11))
  plt.title('Mean posterior (sersic2morph)')
  s = 1e-2
  plt.imshow(np.arcsinh((im_rec1)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/mean_posterior.png")


  plt.figure(figsize=(11,11))
  plt.title('Mean posterior (sersic2morph)')
  s = 1e-2
  plt.imshow(np.arcsinh((im_rec1-res)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/residuals.png")
  '''

if __name__ == "__main__":
    app.run(main)

