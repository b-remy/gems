from absl import app
from absl import flags
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability import edward2 as ed
from gems.ed_utils import make_value_setter, make_log_joint_fn

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import galsim
import galflow
lp = galflow.lightprofiles
import time

from gems.models import gaussian_model, sersic_model, sersic2morph_model
from galsim.bounds import _BoundsI

flags.DEFINE_integer("n", 10, "number of interations")
flags.DEFINE_boolean("flat_p_e", False, "number of interations")

FLAGS = flags.FLAGS

_log10 = tf.math.log(10.)
_scale = im_scale = 0.03 # COSMOS pixel size in arcsec
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

  #cat = galsim.COSMOSCatalog(sample='23.5', dir='/gpfswork/rech/xdy/commun/galsim_catalogs/COSMOS_23.5_training_sample')
  cat = galsim.COSMOSCatalog(sample='25.2', dir='/gpfswork/rech/xdy/commun/galsim_catalogs/COSMOS_25.2_training_sample')
  index = range(N*N)
  print('ok')


  im_scale = 0.03
  interp_factor=1
  padding_factor=1

  """
  Nk = stamp_size*interp_factor*padding_factor
  from galsim.bounds import _BoundsI
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                        scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                        recenter=False)
  imkpsf = tf.signal.fftshift(tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk//2+1]), axes=1)
  """

  # generate observations
  obs = []
  n = []
  flux = []
  hlr = []
  q_list = []
  phi_list = []
  psfs = []

  ind = 0
  ind_ = 0
  
  sess = tf.Session()
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
          #print(len(obs))
          galp = cat.makeGalaxy(ind, gal_type='parametric')
          galr = cat.makeGalaxy(ind, gal_type='real')
          psf = galr.original_psf
          imkpsf = gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale)


          n.append(galp.original.n)
          flux.append(galp.original.flux)
          hlr.append(galp.original.half_light_radius)
          q = cat.param_cat['sersicfit'][cat.orig_index[ind]][3]
          q_list.append(q)
          phi = cat.param_cat['sersicfit'][cat.orig_index[ind]][7]
          phi_list.append(phi)
          psfs.append(imkpsf)

          e1 = tf.reshape(tf.cast((1-q)/(1+q) * tf.math.cos(2*phi), tf.float32), (1,))
          e2 = tf.reshape(tf.cast((1-q)/(1+q) * tf.math.sin(2*phi), tf.float32), (1,))
          #'''
          gal = lp.sersic(n=[galp.original.n], half_light_radius=[galp.original.half_light_radius], flux=[galp.original.flux], nx=stamp_size, scale=0.03)
          obs_ = galflow.shear(tf.reshape(gal, (1,stamp_size,stamp_size,1)), e1, e2)
          obs_ = galflow.shear(tf.reshape(obs_, (1,stamp_size,stamp_size,1)), 0.05*tf.ones(1,), -0.05*tf.ones(1,))
          

          ims = tf.reshape(obs_, (1,stamp_size,stamp_size,1))
          kpsf = imkpsf
          obs_ = galflow.convolve(ims, kpsf,
                        zero_padding_factor=padding_factor,
                        interp_factor=interp_factor)[0,...,0]
          '''
          gal.shear(g1=0.05, g2=-0.05)
          conv = galsim.Convolve(gal, psf)
          #obs_ = tf.convert_to_tensor(conv.drawImage(nx=stamp_size, ny=stamp_size).array, tf.float32)
          obs_ = conv.drawImage(nx=stamp_size, ny=stamp_size).array
          # obs_ = obs_ + 0.003 * tf.random.normal(obs_.shape)
          '''
          #with tf.Session() as sess:
          obs__ = sess.run(obs_)
          img = galsim.Image(obs__, copy=False)
          img.addNoise(galsim.Convolve(galr, galr.original_psf).noise)
          seed = ind
          generator = galsim.random.BaseDeviate(seed=seed)
          g_noise = galsim.GaussianNoise(rng=generator, sigma=noise_level)
          img.addNoise(g_noise)
          obs_ = tf.convert_to_tensor(img.array)
          obs.append(obs_)
         

          print(len(obs), ind)
          ind_ += 1
          ind += 1
    else:
      ind += 1
  folder_name = 'cosmos_parametric'
  job_name = str(int(time.time()))
  os.mkdir("res/"+folder_name+"/{}".format(job_name))
  os.mkdir("res/"+folder_name+"/{}/params".format(job_name))

  #obs_64 = tf.convert_to_tensor(np.load('res/cosmos_parametric/1656973656/obs_batched.npy'))
  
  obs_64 = tf.expand_dims(tf.stack(obs, axis=0), 0)
  obs_64_ = sess.run(obs_64)
  #np.save("res/"+folder_name+"/"+job_name+"/obs_batched.npy", obs_64_)
  res = obs_64_.reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  plt.figure(figsize=(11,11))
  plt.title('Observations (sersic)')
  s = 1e-1
  plt.imshow(np.arcsinh((res)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/obs.png")
  np.save("res/"+folder_name+"/"+job_name+"/obs.npy", obs_64_)


  #obs = tf.expand_dims(tf.stack(obs, axis=0), 0)[..., 10:-10, 10:-10] # [1, batch, nx, ny]
  obs = obs_64[..., 10:-10, 10:-10]
  n = tf.expand_dims(tf.stack(n, axis=0), 0)
  flux = tf.expand_dims(tf.stack(flux, axis=0), 0)
  hlr = tf.expand_dims(tf.stack(hlr, axis=0), 0)
  q_ = tf.expand_dims(tf.stack(q_list, axis=0), 0)
  phi = tf.expand_dims(tf.stack(phi_list, axis=0), 0)
  e1 = tf.cast((1-q_)/(1+q_) * tf.math.cos(2*phi), tf.float32)
  e2 = tf.cast((1-q_)/(1+q_) * tf.math.sin(2*phi), tf.float32)
  imkpsfs = tf.concat(psfs, axis=0)
  # print(n.shape)
  # print(hlr.shape)
  # print(flux.shape)
  
  
  

  true_hlr = hlr
  true_n = n
  true_e = tf.stack([e1, e2], -1)
  # print(true_e.shape)
  true_gamma = tf.expand_dims(tf.convert_to_tensor([0.05, -0.05]), 0)
  
  t1, t2, t3 = sess.run([true_hlr, true_e, true_gamma])
  print('t hlr', t1)
  print('t e', t2)
  print('t gamma', t3)

  # print(obs.shape)
  # print(true_gamma.shape)
  # print(true_e.shape)



  
  '''
  res = obs_64.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  plt.figure(figsize=(11,11))
  plt.title('Real galaxies (COSMOS)')
  s = 1e-2
  plt.imshow(np.arcsinh(res/s)*s)
  plt.savefig("res/"+folder_name+"/"+job_name+"/gals.png")

  ## saving true params for later comparison
  # np.save("res/"+folder_name+"/"+job_name+"/params/gals.npy", ims.numpy())
  # np.save("res/"+folder_name+"/"+job_name+"/params/shear.npy", true_params['gamma'].numpy())
  '''

  # Get the joint log prob
  batch_size = 1
  log_prob = make_log_joint_fn(partial(sersic2morph_model, 
                                       flat_prior_e=FLAGS.flat_p_e,
                                       stamp_size=stamp_size,
                                       batch_size=batch_size, 
                                       sigma_e=noise_level, 
                                       num_gal=N*N, kpsf=imkpsfs, fixed_flux=True, n=n, flux=flux, 
                                       #hlr=hlr
                                      )
                               )

  scale_hlr = 0.5
  scale_e = .5
  scale_gamma = .1
  def target_log_prob_fn(hlr, gamma, e):#, F):
    return log_prob(
           #n=n,
           hlr=hlr*scale_hlr,
           gamma=gamma*scale_gamma, # trick to adapt the step size
           e=e*scale_e,
           obs=obs)

  num_results = FLAGS.n
  num_burnin_steps = 1

  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    num_leapfrog_steps=3,
    # step_size=.00005)
    # step_size=.0001)
    step_size=.003)





  # start_gamma = np.load('/local/home/br263581/Bureau/gems/res/"+folder_name+"/1652309115/samples100_30000_gamma_0.05000000074505806_-0.05000000074505806.npy')
  # start_gamma = tf.expand_dims(tf.convert_to_tensor(start_gamma)[-1], axis=0)
  # start_e = np.load('/local/home/br263581/Bureau/gems/res/"+folder_name+"/1652309115/samples100_30000_e.npy')
  # start_e = tf.expand_dims(tf.convert_to_tensor(start_e)[-1], axis=0)
  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
                      #true_hlr/scale_hlr,#
                      tf.math.log(true_hlr)/tf.math.log(10.)/scale_hlr,
                      true_gamma/scale_gamma*0.,
                      true_e,
        ],
        kernel=adaptive_hmc)
    return samples, trace

  samples, trace = get_samples()

  end = time.time()
  print('Time: {:.2f}'.format((end - begin)/60.))
  print('')
 
  hlr_est = samples[0][:,0,:]*scale_hlr
  gamma_est = samples[1][:,0,:]*scale_gamma
  e_est = samples[2][:,0,:]*scale_e
  gamma_true = true_gamma[0,:]

  with ed.interception(make_value_setter(hlr=tf.reduce_mean(hlr_est, axis=0, keepdims=True),
                                         e=tf.reduce_mean(e_est, axis=0, keepdims=True),
                                         gamma=tf.reduce_mean(gamma_est, axis=0, keepdims=True),
                                         )):
    rec1 = sersic2morph_model(batch_size=batch_size,
                              stamp_size=stamp_size,
                              sigma_e=0.,
                              num_gal=N*N,
                              kpsf=imkpsfs,
                              interp_factor = interp_factor,
                              padding_factor = padding_factor,
                              fixed_flux=True,
                              n=n, flux=flux,# hlr=hlr,
                              #fit_centroid=True, 
                              display=True)






  #with tf.Session() as sess:
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    e_est, gamma_est, trace, true_e, gamma_true, obs_64, true_hlr, hlr_est, rec1 = sess.run([e_est, gamma_est, trace, true_e, gamma_true, obs_64,true_hlr,  hlr_est, rec1])

  im_rec1 = rec1.reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])  
  obs_64 = obs_64.reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  print('accptance ratio:', trace.is_accepted.sum()/len(trace.is_accepted))


  """
  plt.figure(figsize=(11,11))
  plt.title('Observations (sersic)')
  s = 1e-1
  plt.imshow(np.arcsinh((obs_64)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/obs.png")
  np.save("res/"+folder_name+"/"+job_name+"/obs.npy", obs_64)
  """

  np.save("res/"+folder_name+"/"+job_name+"/samples_gamma.npy", gamma_est)
  np.save("res/"+folder_name+"/"+job_name+"/samples_e.npy", e_est)
  # np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_r.npy".format(N*N, num_results), hlr_est)


  ## Save fitted image
  plt.figure(figsize=(11,11))
  plt.title('Mean posterior (sersic2morph)')
  s = 1e-2
  #plt.imshow(np.arcsinh((im_rec1)/s)*s)
  plt.imshow(im_rec1, vmax=0.1)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/mean_posterior.png")

  print(gamma_est.shape)
  plt.figure()
  plt.plot(gamma_est)
  plt.axhline(gamma_true[0], color='C0', label='g1')
  plt.axhline(gamma_true[1], color='C1', label='g2')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/shear_flat_prior_{}.png".format(FLAGS.flat_p_e))


  plt.figure()
  plt.title('hlr')
  for i in range(5):
    plt.plot(hlr_est[:,i], label='{}'.format(i))
    #plt.axhline(true_hlr[0,i], color='gray')
    plt.axhline(np.log(true_hlr[0,i])/np.log(10.), color='gray')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/hlr.png")


  plt.figure()
  plt.subplot(121)
  plt.title('e1')
  for i in range(5):
    plt.plot(e_est[:,i,0], label='{}'.format(i))
    plt.axhline(true_e[0,i,0], color='gray')
  plt.legend()

  print(np.where((e_est[:,:,0]-true_e[:,:,0]) > 0.02))

  plt.subplot(122)
  plt.title('e2')
  for i in range(5):
    plt.plot(e_est[:,i,1], label='{}'.format(i))
    plt.axhline(true_e[0,i,1], color='gray')
  plt.legend()  
  plt.savefig("res/"+folder_name+"/"+job_name+"/e.png")

if __name__ == "__main__":
    app.run(main)
