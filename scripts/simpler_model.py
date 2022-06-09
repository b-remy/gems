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

from gems.models import gaussian_model, sersic_model

flags.DEFINE_integer("n", 10, "number of interations")
FLAGS = flags.FLAGS

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
stamp_size = 64

N = 10 # number of stamp in a row/col


def main(_):
  begin = time.time()
  # Execute probabilistic program and record execution trace
  batch_size = 1
  with ed.tape() as true_params:
    # ims = gaussian_model(batch_size=N*N, stamp_size=stamp_size)
    # ims = sersic_model(batch_size=N*N, stamp_size=stamp_size)
    ims =  partial(gaussian_model, fixed_flux=True)(num_gal=N*N, stamp_size=stamp_size)
    # ims =  partial(sersic_model)(batch_size=batch_size, num_gal=N*N, stamp_size=stamp_size)
  
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
  
  job_name = str(int(time.time()))
  os.mkdir("res/simpler_model/{}".format(job_name))
  os.mkdir("res/simpler_model/{}/params".format(job_name))

  plt.figure()
  plt.imshow(res, cmap='gray_r')
  plt.savefig("res/simpler_model/"+job_name+"/gals.png")

  ## saving true params for later comparison
  np.save("res/simpler_model/"+job_name+"/params/gals.npy", ims.numpy())
  np.save("res/simpler_model/"+job_name+"/params/shear.npy", true_params['gamma'].numpy())
  np.save("res/simpler_model/"+job_name+"/params/e.npy", true_params['e'].numpy())
  np.save("res/simpler_model/"+job_name+"/params/hlr.npy", true_params['hlr'].numpy())

  # Get the joint log prob
  batch_size = 1
  # log_prob = ed.make_log_joint_fn(gaussian_model)
  log_prob = make_log_joint_fn(partial(gaussian_model, batch_size=batch_size, num_gal=N*N, fixed_flux=True))

  
  scale_e = 1.#.01
  scale_F = 1.#10.
  scale_gamma = 1.#.01
  # hlr, gamma and e are free parameters
  # def target_log_prob_fn(hlr, gamma, e, F):
  def target_log_prob_fn(hlr, gamma, e):#, F):
    return log_prob(hlr=hlr,
           gamma=gamma*scale_gamma, # trick to adapt the step size
           e=e*scale_e,
          #  F=F*scale_F,
           obs=ims)


  num_results = FLAGS.n
  num_burnin_steps = 1

  # # define the kernel sampler
  # adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
  #     target_log_prob_fn=target_log_prob_fn,
  #     num_leapfrog_steps=3,
  #     step_size=.0001),
  #   num_adaptation_steps=int(num_results * 0.8))

  # @tf.function
  # def get_samples():
  #   samples, [step_size, log_accept_ratio] = tfp.mcmc.sample_chain(
  #       num_results=num_results,
  #       num_burnin_steps=num_burnin_steps,
  #       current_state=[true_params['hlr']*0.-.68, # init with prior mean
  #                     true_params['gamma']*0., # init with zero shear
  #                     true_params['e']*0., # init with zero ellipticity
  #                     # tf.math.log(16.693710205567005)/tf.math.log(10.), # init with zero Flux
  #       ],
  #       kernel=adaptive_hmc,
  #       trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
  #                            pkr.inner_results.log_accept_ratio])
  #   return samples, step_size, log_accept_ratio

  # define the kernel sampler
  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      num_leapfrog_steps=3,
      # step_size=.00005)
      # step_size=.0001)
      step_size=.00002)

  # @tf.function
  # def get_samples():
  #   samples, trace = tfp.mcmc.sample_chain(
  #       num_results=num_results,
  #       num_burnin_steps=num_burnin_steps,
  #       current_state=[true_params['hlr']*0.-.68, # init with prior mean
  #                     true_params['gamma']*0., # init with zero shear
  #                     true_params['e']*0., # init with zero ellipticity
  #                     true_params['hlr']*0.+10.,
  #                     # tf.math.log(16.693710205567005)/tf.math.log(10.), # init with zero Flux
  #       ],
  #       kernel=adaptive_hmc)
  #   return samples, trace

  @tf.function
  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[true_params['hlr'], # init with prior mean
                      true_params['gamma'], # init with zero shear
                      true_params['e'], # init with zero ellipticity
                      # true_params['hlr']*0.+10./scale_F,
                      # tf.math.log(16.693710205567005)/tf.math.log(10.), # init with zero Flux
        ],
        kernel=adaptive_hmc)
    return samples, trace

  # @tf.function
  # def get_samples():
  #   samples, [step_size, log_accept_ratio] = tfp.mcmc.sample_chain(
  #       num_results=num_results,
  #       num_burnin_steps=num_burnin_steps,
  #       current_state=[true_params['hlr'], # init with prior mean
  #               true_params['gamma']*0., # init with zero shear
  #               true_params['e']*0., # init with zero ellipticity
  #               true_params['hlr']+10.,
  #       ],
  #       kernel=adaptive_hmc,
  #       trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
  #                            pkr.inner_results.log_accept_ratio])
  #   return samples, step_size, log_accept_ratio

  # print("start sampling...")

  samples, trace = get_samples()
  print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  # samples, step_size, log_accept_ratio = get_samples()
  # p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(
  #   tf.minimum(log_accept_ratio, 0.)))
  # print(p_accept.numpy())
  # print('step_size:', step_size.numpy())

  # print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  end = time.time()  
  print('Time: {:.2f}'.format((end - begin)/60.))
  print('')

  hlr_est = samples[0].numpy()[:,0,:]
  gamma_est = samples[1].numpy()[:,0,:]*scale_gamma
  e_est = samples[2].numpy()[:,0,:]*scale_e
  # F_est = samples[3].numpy()[:,0,:]*scale_F
  # gamma_true = custom_shear
  gamma_true = true_params['gamma'][0].numpy()
  # F_est = samples[3].numpy()

  np.save("res/simpler_model/"+job_name+"/samples{}_{}_gamma_{}_{}.npy".format(N*N, num_results, gamma_true[0], gamma_true[1]), gamma_est)
  np.save("res/simpler_model/"+job_name+"/samples{}_{}_e.npy".format(N*N, num_results), e_est)
  np.save("res/simpler_model/"+job_name+"/samples{}_{}_r.npy".format(N*N, num_results), hlr_est)

  # Display results
  with ed.condition(hlr=samples[0][-1],
                  gamma=samples[1][-1],
                  e=samples[2][-1],
                  # F=F_est[-1]
                  ):
    rec0 = gaussian_model(batch_size=batch_size, num_gal=N*N, fixed_flux=True)
  im_rec0 = rec0.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  plt.figure()
  plt.imshow(im_rec0, cmap='gray_r')
  plt.savefig("res/simpler_model/"+job_name+"/gals_res.png")

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
  plt.xlim([-.1,.1])
  plt.ylim([-.1,.1])

  plt.axvline(gamma_true[0])
  plt.axhline(gamma_true[1])
  # plt.xlim([gamma_true[0]-0.02, gamma_true[0]+0.02])
  # plt.ylim([gamma_true[1]-0.02, gamma_true[1]+0.02])
  plt.savefig("res/simpler_model/"+job_name+"/shear_countours.png", bbox_inches='tight')

  # plt.figure()
  # plt.title('F')
  # for i in range(16):
  #   plt.plot(F_est[:,i])
  # plt.savefig("res/simpler_model/"+job_name+"/F.png")

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

  plt.savefig("res/simpler_model/"+job_name+"/e.png")

  plt.figure()
  plt.title('hlr')
  for i in range(16):
    plt.plot(hlr_est[:,i])
    plt.axhline(true_params['hlr'][0].numpy()[i], color='gray')

  plt.savefig("res/simpler_model/"+job_name+"/hlr.png")

if __name__ == "__main__":
    app.run(main)
