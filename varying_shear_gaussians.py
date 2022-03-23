from absl import app
from absl import flags

import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import os

import galsim
import galflow
lp = galflow.lightprofiles
import time
import matplotlib.pyplot as plt

from ed_utils import make_log_joint_fn

from models import gaussian_model, varying_shear_gaussian_model
from shear import latent_to_shear

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
# N = 7 # number of stamp in a row/col
# stamp_size = 64

def main(_):
  begin = time.time()
  # shear_field(batch_size=1, num_gal=25, stamp_size=64, scale=0.03, fixed_flux=False)
  
  stamp_size = 64
  N = 23
  batch_size = 1

  # Execute probabilistic program and record execution trace
  with ed.tape() as true_params:
    # ims =  partial(gaussian_model, fixed_flux=True)(batch_size=N*N, stamp_size=stamp_size)
    ims =  partial(varying_shear_gaussian_model, fixed_flux=True)(num_gal=N*N, stamp_size=stamp_size)
  
  res = ims.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])

  job_name = str(int(time.time()))
  os.mkdir("res/varying_shear/{}".format(job_name))
  
  plt.figure()
  plt.imshow(res, cmap='gray_r')
  plt.savefig('res/varying_shear/'+job_name+'/gals.png')

  batch_size = 1
  log_prob = make_log_joint_fn(partial(varying_shear_gaussian_model, batch_size=batch_size, num_gal=N*N, fixed_flux=True))

  scale_factor = 1.
  # hlr, gamma and e are free parameters
  def target_log_prob_fn(hlr, gamma, e):
    return log_prob(hlr=hlr,
           latent_shear=gamma/scale_factor, # trick to adapt the step size
           e=e,
           obs=ims)

  ## define the kernel sampler
  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      num_leapfrog_steps=3,
      step_size=.0005)

  num_results = 900
  num_burnin_steps = 1

  # init_hlr = tf.expand_dims(true_params['hlr']*0.-.68, 0)
  # init_gamma = tf.expand_dims(true_params['gamma']*0., 0)
  # init_e = tf.expand_dims(true_params['e']*0., 0)
  
  # init_hlr = true_params['hlr']*0.-.68
  # # init_gamma = true_params['gamma']#*0.
  # init_gamma = true_params['latent_shear']*0.#*0.
  # init_e = true_params['e']*0.#*0.
  
  init_hlr = true_params['hlr'] + 0.001*tf.random.normal(true_params['hlr'].numpy().shape)
  # init_gamma = true_params['gamma']#*0.
  init_gamma = true_params['latent_shear'] + 0.001*tf.random.normal(true_params['latent_shear'].numpy().shape)
  init_e = true_params['e'] + 0.01*tf.random.normal(true_params['e'].numpy().shape)

  init_state = [init_hlr, init_gamma, init_e]
  # print(init_state.shape)

  @tf.function
  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=init_state,
        kernel=adaptive_hmc)
    return samples, trace

  print("start sampling...")

  samples, trace = get_samples()
  end = time.time()

  print('Time:', (end - begin)/60.)
  print('')

  print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  hlr_est = samples[0].numpy()[:,0,:]
  shear_est = samples[1].numpy()[:,0,:]/scale_factor
  # print(shear_est.shape)
  # shear_est = shear_est[0]
  # convert convergence to shear
  shear_est = tf.stack(latent_to_shear(shear_est, 16, 5.), -1)
  e_est = samples[2].numpy()[:,0,:]
  # gamma_true = custom_shear
  # shear_true = true_params['latent_shear'].numpy()[0]
  # print(shear_true.shape)
  shear_true = true_params['latent_shear']
  shear_true = tf.stack(latent_to_shear(shear_true, 16, 5.), -1)[0,...]

  # print(shear_true)

  # print(shear_est.shape)
  # print(shear_true.shape)
  # print(shear_est)

  # TODO: COMPARE convergence

  plt.figure()
  plt.subplot(221)
  plt.plot(shear_est[:,0,0])
  plt.axhline(shear_true[0,0,0], color='C0', label='g1', alpha=0.2)
  plt.axhline(shear_true[0,0,1], color='C1', label='g2', alpha=0.2)
  plt.subplot(222)
  plt.plot(shear_est[:,0,1])
  plt.axhline(shear_true[0,1,0], color='C0', label='g1', alpha=0.2)
  plt.axhline(shear_true[0,1,1], color='C1', label='g2', alpha=0.2)
  plt.subplot(223)
  plt.plot(shear_est[:,1,0])
  plt.axhline(shear_true[1,0,0], color='C0', label='g1', alpha=0.2)
  plt.axhline(shear_true[1,0,1], color='C1', label='g2', alpha=0.2)
  plt.subplot(224)
  plt.plot(shear_est[:,1,1])
  plt.axhline(shear_true[1,1,0], color='C0', label='g1', alpha=0.2)
  plt.axhline(shear_true[1,1,1], color='C1', label='g2', alpha=0.2)
  plt.savefig('res/varying_shear/'+job_name+'/shear_inference.png')

  plt.figure()
  plt.subplot(121)
  plt.title('e1')
  for i in range(16):
    plt.plot(e_est[:,i,0])
    plt.axhline(true_params['e'][0].numpy()[i,0], color='gray')

  plt.subplot(122)
  plt.title('e2')
  for i in range(16):
    plt.plot(e_est[:,i,1])
    plt.axhline(true_params['e'][0].numpy()[i,1], color='gray')

  plt.savefig("res/varying_shear/"+job_name+"/e.png")

  plt.figure()
  plt.title('hlr')
  for i in range(16):
    plt.plot(hlr_est[:,i])
    plt.axhline(true_params['hlr'][0].numpy()[i], color='gray')

  plt.savefig('res/varying_shear/'+job_name+'/hlr.png')

if __name__ == "__main__":
    app.run(main)
