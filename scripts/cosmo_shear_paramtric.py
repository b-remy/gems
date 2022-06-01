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

from gems.models import gaussian_model, sersic_model, sersic2morph_model
from galsim.bounds import _BoundsI

flags.DEFINE_integer("n", 10, "number of interations")
FLAGS = flags.FLAGS

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
stamp_size = 64

N = 10 # number of stamp in a row/col

def main(_):
  begin = time.time()

  num_gal = N*N

  cat = galsim.COSMOSCatalog(sample='23.5')
  index = range(N*N)
  
  gal = cat.makeGalaxy(0, gal_type='real')
  psf = gal.original_psf
  im_scale = 0.03
  interp_factor=1
  padding_factor=1
  Nk = stamp_size*interp_factor*padding_factor
  from galsim.bounds import _BoundsI
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                        scale=2.*np.pi/(N*padding_factor* im_scale),
                        recenter=False)
  imkpsf = tf.signal.fftshift(tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk//2+1]), axes=1)

  # generate observations
  obs = []
  n = []
  flux = []
  hlr = []
  q_list = []
  phi_list = []
  ind = 0
  ind_ = 0
  while len(obs) < num_gal:
    galp = cat.makeGalaxy(ind, gal_type='parametric')
    if cat.param_cat['use_bulgefit'][cat.orig_index[ind]] == 0:
      if galp.original.n < 0.4:
        ind += 1
      else:
        if ind_==6 or ind_==93 or ind_==56 or ind_==55:
          ind+=1
          ind_+=1
        else:
          # gal = cat.makeGalaxy(ind, gal_type='real')
          gal = cat.makeGalaxy(ind, gal_type='parametric')
          galr = cat.makeGalaxy(ind, gal_type='real')
          n.append(galp.original.n)
          flux.append(galp.original.flux)
          hlr.append(galp.original.half_light_radius)
          q = cat.param_cat['sersicfit'][cat.orig_index[ind]][3]
          q_list.append(q)
          phi = cat.param_cat['sersicfit'][cat.orig_index[ind]][7]
          phi_list.append(phi)
          e1 = tf.reshape(tf.cast((1-q)/(1+q) * tf.math.cos(2*phi), tf.float32), (1,))
          e2 = tf.reshape(tf.cast((1-q)/(1+q) * tf.math.sin(2*phi), tf.float32), (1,))
          gal = lp.sersic(n=[galp.original.n], half_light_radius=[galp.original.half_light_radius], flux=[galp.original.flux], nx=stamp_size, scale=0.03)
          obs_ = galflow.shear(tf.reshape(gal, (1,stamp_size,stamp_size,1)), e1, e2)
          obs_ = galflow.shear(tf.reshape(obs_, (1,stamp_size,stamp_size,1)), 0.05*tf.ones(1,), -0.05*tf.ones(1,))
          

          ims = tf.reshape(obs_, (1,stamp_size,stamp_size,1))
          kpsf = imkpsf
          obs_ = galflow.convolve(ims, kpsf,
                        zero_padding_factor=padding_factor,
                        interp_factor=interp_factor)[0,...,0]
          
          # gal.shear(g1=0.05, g2=-0.05)
          # conv = galsim.Convolve(gal, psf)
          # obs_ = tf.convert_to_tensor(conv.drawImage(nx=stamp_size, ny=stamp_size).array, tf.float32)
          
          # obs_ = obs_ + 0.003 * tf.random.normal(obs_.shape)
          img = galsim.Image(obs_.numpy(), copy=False)
          img.addNoise(galsim.Convolve(galr, galr.original_psf).noise)
          seed = ind
          generator = galsim.random.BaseDeviate(seed=seed)
          g_noise = galsim.GaussianNoise(rng=generator, sigma=0.003)
          img.addNoise(g_noise)
          obs_ = tf.convert_to_tensor(img.array)
          obs.append(obs_)


          #if ind_==93:
          ind_ += 1
          ind += 1
    else:
      ind += 1
      # convolve with a constant PSF, from COSMOS index 0

  obs = tf.expand_dims(tf.stack(obs, axis=0), 0)[..., 10:-10, 10:-10] # [1, batch, nx, ny]
  n = tf.expand_dims(tf.stack(n, axis=0), 0)
  flux = tf.expand_dims(tf.stack(flux, axis=0), 0)
  hlr = tf.expand_dims(tf.stack(hlr, axis=0), 0)
  q_ = tf.expand_dims(tf.stack(q_list, axis=0), 0)
  phi = tf.expand_dims(tf.stack(phi_list, axis=0), 0)
  e1 = tf.cast((1-q_)/(1+q_) * tf.math.cos(2*phi), tf.float32)
  e2 = tf.cast((1-q_)/(1+q_) * tf.math.sin(2*phi), tf.float32)

  # print(n.shape)
  # print(hlr.shape)
  # print(flux.shape)
  
  
  

  true_hlr = hlr
  true_e = tf.stack([e1, e1], -1)
  # print(true_e.shape)
  true_gamma = tf.expand_dims(tf.convert_to_tensor([0.05, -0.05]), 0)
  
  # print(obs.shape)
  # print(true_gamma.shape)
  # print(true_e.shape)



  folder_name = 'cosmos_parametric'
  job_name = str(int(time.time()))
  os.mkdir("res/"+folder_name+"/{}".format(job_name))
  os.mkdir("res/"+folder_name+"/{}/params".format(job_name))

  # plt.figure()
  # plt.imshow(res, cmap='gray_r')
  # plt.savefig("res/"+folder_name+"/"+job_name+"/gals.png")

  ## saving true params for later comparison
  # np.save("res/"+folder_name+"/"+job_name+"/params/gals.npy", ims.numpy())
  # np.save("res/"+folder_name+"/"+job_name+"/params/shear.npy", true_params['gamma'].numpy())
  np.save("res/"+folder_name+"/"+job_name+"/params/e.npy", true_e.numpy())
  np.save("res/"+folder_name+"/"+job_name+"/params/hlr.npy", true_gamma.numpy())


  # Get the joint log prob
  batch_size = 1
  # log_prob = ed.make_log_joint_fn(gaussian_model)
  log_prob = make_log_joint_fn(partial(sersic2morph_model, batch_size=batch_size, num_gal=N*N, kpsf=imkpsf, fixed_flux=True,
                                      n=n, flux=flux, hlr=hlr))

  scale_e = 1.
  scale_F = 1.
  scale_gamma = .1
  def target_log_prob_fn(gamma, e):#, F):
    return log_prob(
            # hlr=hlr,
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
    step_size=.001)


  # start_gamma = np.load('/local/home/br263581/Bureau/gems/res/"+folder_name+"/1652309115/samples100_30000_gamma_0.05000000074505806_-0.05000000074505806.npy')
  # start_gamma = tf.expand_dims(tf.convert_to_tensor(start_gamma)[-1], axis=0)
  # start_e = np.load('/local/home/br263581/Bureau/gems/res/"+folder_name+"/1652309115/samples100_30000_e.npy')
  # start_e = tf.expand_dims(tf.convert_to_tensor(start_e)[-1], axis=0)
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
  gamma_true = true_gamma.numpy()[0,:]

  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_gamma_{}_{}.npy".format(N*N, num_results, gamma_true[0], gamma_true[1]), gamma_est)
  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_e.npy".format(N*N, num_results), e_est)
  # np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_r.npy".format(N*N, num_results), hlr_est)


  print(gamma_est.shape)
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

if __name__ == "__main__":
    app.run(main)
