from absl import app
from absl import flags

import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from ed_utils import make_log_joint_fn

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import galsim
import galflow
lp = galflow.lightprofiles
import timeit

from models import gaussian_model

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
N = 5 # number of stamp in a row/col
stamp_size = 64

def main(_):

  batch_size = 1
  # Execute probabilistic program and record execution trace
  with ed.tape() as true_params:
    # ims =  partial(gaussian_model, fixed_flux=True)(batch_size=N*N, stamp_size=stamp_size)
    ims =  partial(gaussian_model, fixed_flux=True)(num_gal=N*N, stamp_size=stamp_size)
  
  # Apply a constant shear on the field
  # custom_shear = [0.015, 0.005]
  # with ed.condition(hlr=true_params['hlr'], 
  #                 gamma=true_params['gamma'],
  #                 # gamma=custom_shear,
  #                 e=true_params['e'],
  #                 ):
  #   ims = gaussian_model(batch_size=N*N, stamp_size=stamp_size)

  # Display things
  res = ims.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  
  plt.figure()
  plt.imshow(res, cmap='gray_r')
  plt.savefig('res/gals.png')

  # Get the joint log prob
  # log_prob = ed.make_log_joint_fn(partial(gaussian_model, fixed_flux=True))
  batch_size = 1
  log_prob = make_log_joint_fn(partial(gaussian_model, batch_size=batch_size, num_gal=N*N, fixed_flux=True))
  
  scale_factor = 1.
  # hlr, gamma and e are free parameters
  def target_log_prob_fn(hlr, gamma, e):
    return log_prob(hlr=hlr,
           gamma=gamma/scale_factor, # trick to adapt the step size
           e=e,
           obs=ims)

  ## define the kernel sampler
  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      num_leapfrog_steps=3,
      step_size=.00001)

  num_results = 10000
  num_burnin_steps = 1

  # init_hlr = tf.expand_dims(true_params['hlr']*0.-.68, 0)
  # init_gamma = tf.expand_dims(true_params['gamma']*0., 0)
  # init_e = tf.expand_dims(true_params['e']*0., 0)
  init_hlr = true_params['hlr']#*0.-.68
  init_gamma = true_params['gamma']#*0.
  init_e = true_params['e']#*0.

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
  # print(samples)
  
  print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  print(true_params['gamma'].shape)
  print(true_params['e'].shape)
  hlr_est = samples[0].numpy()[:,0,:]
  gamma_est = samples[1].numpy()[:,0,:]/scale_factor
  e_est = samples[2].numpy()[:,0,:]
  # gamma_true = custom_shear
  gamma_true = true_params['gamma'].numpy()[0,:]

  np.save("res/gaussian/samples{}_{}_shear_{}_{}.npy".format(N*N, num_results, gamma_true[0], gamma_true[1]), gamma_est)
  np.save("res/gaussian/samples{}_{}_e.npy".format(N*N, num_results), e_est)
  np.save("res/gaussian/samples{}_{}_r.npy".format(N*N, num_results), hlr_est)

  # Display things

  plt.figure()
  plt.plot(gamma_est)
  plt.axhline(gamma_true[0], color='C0', label='g1')
  plt.axhline(gamma_true[1], color='C1', label='g2')
  plt.legend()
  plt.savefig('res/gaussian/shear.png')

  plt.figure()
  az.plot_pair(
    {'gamma1':gamma_est[:,0], 
     'gamma2':gamma_est[:,1]},
    var_names=["gamma1", "gamma2"],
    kind="kde",
    divergences=True,
    textsize=18,
    )

  plt.axvline(gamma_true[0])
  plt.axhline(gamma_true[1])
  plt.savefig('res/gaussian/shear_countours.png', bbox_inches='tight')

  plt.figure()
  plt.subplot(121)
  plt.title('e1')
  for i in range(16):
    plt.plot(e_est[:,i,0])
    plt.axhline(true_params['e'].numpy()[0,i,0], color='gray')

  plt.subplot(122)
  plt.title('e2')
  for i in range(16):
    plt.plot(e_est[:,i,1])
    plt.axhline(true_params['e'].numpy()[0,i,1], color='gray')

  plt.savefig('res/gaussian/e.png')

  plt.figure()
  plt.title('hlr')
  for i in range(16):
    plt.plot(hlr_est[:,i])
    plt.axhline(true_params['hlr'].numpy()[0,i], color='gray')

  plt.savefig('res/gaussian/hlr.png')

if __name__ == "__main__":
    app.run(main)
