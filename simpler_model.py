from absl import app
from absl import flags
import os

import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import galsim
import galflow
lp = galflow.lightprofiles
import timeit

from models import gaussian_model, sersic_model

flags.DEFINE_integer("n", 10, "number of interations")
FLAGS = flags.FLAGS

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
N = 5 # number of stamp in a row/col
stamp_size = 64

def main(_):

  # Execute probabilistic program and record execution trace
  with ed.tape() as true_params:
    # ims = gaussian_model(batch_size=N*N, stamp_size=stamp_size)
    ims = sersic_model(batch_size=N*N, stamp_size=stamp_size)
  
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
  log_prob = ed.make_log_joint_fn(gaussian_model)
  
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
      step_size=.00005)

  num_results = FLAGS.n
  num_burnin_steps = 1

  @tf.function
  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[true_params['hlr']*0.-.68, # init with prior mean
                      true_params['gamma']*0., # init with zero shear
                      true_params['e']*0., # init with zero ellipticity
        ],
        kernel=adaptive_hmc)
    return samples, trace

  print("start sampling...")

  samples, trace = get_samples()
  print(samples)
  
  print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  hlr_est = samples[0].numpy()
  gamma_est = samples[1].numpy()/scale_factor 
  e_est = samples[2].numpy()
  # gamma_true = custom_shear
  gamma_true = true_params['gamma'].numpy()

  job_name = str(int(time.time()))
  os.mkdir("res/{}".format(job_name))
  np.save("res/simpler_model/"+job_name+"/samples{}_{}_shear_{}_{}.npy".format(N*N, num_results, gamma_true[0], gamma_true[1]), gamma_est)
  np.save("res/simpler_model/"+job_name+"/samples{}_{}_e.npy".format(N*N, num_results), e_est)
  np.save("res/simpler_model/"+job_name+"/samples{}_{}_r.npy".format(N*N, num_results), hlr_est)

  # Display things

  plt.figure()
  plt.plot(gamma_est)
  plt.axhline(gamma_true[0], color='C0', label='g1')
  plt.axhline(gamma_true[1], color='C1', label='g2')
  plt.legend()
  plt.savefig("res/simpler_model/"+job_name+"/shear.png")

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
  plt.savefig("res/simpler_model/"+job_name+"/shear_countours.png", bbox_inches='tight')

  plt.figure()
  plt.subplot(121)
  plt.title('e1')
  for i in range(16):
    plt.plot(e_est[:,i,0])
    plt.axhline(true_params['e'].numpy()[i,0], color='gray')

  plt.subplot(122)
  plt.title('e2')
  for i in range(16):
    plt.plot(e_est[:,i,1])
    plt.axhline(true_params['e'].numpy()[i,1], color='gray')

  plt.savefig("res/simpler_model/"+job_name+"/e.png")

  plt.figure()
  plt.title('hlr')
  for i in range(16):
    plt.plot(hlr_est[:,i])
    plt.axhline(true_params['hlr'].numpy()[i], color='gray')

  plt.savefig("res/simpler_model/"+job_name+"/hlr.png")

if __name__ == "__main__":
    app.run(main)