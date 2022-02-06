from absl import app
from absl import flags

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

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
N = 4 # number of stamp in a row/col
stamp_size = 64

## PSF model from galsim COSMOS catalog
psf = galsim.Gaussian(0.06)

interp_factor=1
padding_factor=1
Nk = stamp_size*interp_factor*padding_factor
from galsim.bounds import _BoundsI
bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

imkpsf = psf.drawKImage(bounds=bounds,
                      scale=2.*_pi/(stamp_size*padding_factor*_scale),
                      recenter=False)

kpsf = tf.cast(np.fft.fftshift(imkpsf.array.reshape(1, Nk, Nk//2+1), axes=1), tf.complex64)

## Forward model
def model(batch_size=N*N, stamp_size=stamp_size):
  """Model:
  - Gaussian light profiles
  - Varying intrinsic ellipticity
  - Constant shear
  """
  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_e = 0.0003

  # prior on Sersic size half light radius
  log_l_hlr = ed.Normal(loc=-.68*tf.ones(batch_size), scale=.3, name="hlr")
  hlr = tf.math.exp(log_l_hlr * _log10)

  # prior on intrinsic galaxy ellipticity
  e = ed.Normal(loc=tf.zeros((batch_size, 2)), scale=.2, name="e")

  # Constant shear in the field
  gamma = ed.Normal(loc=tf.zeros((2,)), scale=0.05, name="gamma")

  # Flux
  F = 16.693710205567005 * tf.ones(batch_size)

  # Generate light profile
  profile = lp.gaussian(half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)

  # Apply intrinsic ellipticity on profiles the image
  ims = tf.expand_dims(profile, -1)  
  ims = galflow.shear(ims, e[:,0], e[:,1])
    
  # Apply same shear on all images
  ims = galflow.shear(ims, 
                        gamma[0]*tf.ones(batch_size),
                        gamma[1]*tf.ones(batch_size))

  # Convolve the image with the PSF

  profile = galflow.convolve(ims, kpsf,
                      zero_padding_factor=padding_factor,
                      interp_factor=interp_factor)[...,0]

  # Returns likelihood
  return  ed.Normal(loc=profile, scale=sigma_e, name="obs")

def main(_):

  # Execute probabilistic program and record execution trace
  with ed.tape() as true_params:
    ims = model(N*N, stamp_size)
  
  # Apply a constant shear on the field
  custom_shear = [0.01, 0.]
  with ed.condition(hlr=true_params['hlr'],
                  gamma=custom_shear,
                  e=true_params['e'],
                  ):
    ims = model(N*N, stamp_size)

  # Display things
  res = ims.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])
  
  plt.figure()
  plt.imshow(res, cmap='gray_r')
  plt.savefig('res/gals.png')

  # Get the joint log prob
  log_prob = ed.make_log_joint_fn(model)
  
  # hlr, gamma and e are free parameters
  def target_log_prob_fn(gamma, e):
    return log_prob(hlr=true_params['hlr'],
           gamma=gamma/10., # trick to adapt the step size
           e=e,
           obs=ims)

  ## define the kernel sampler
  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      num_leapfrog_steps=3,
      step_size=.0002)

  num_results = 50000
  num_burnin_steps = 1

  @tf.function
  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[true_params['gamma']*0., # init with zero shear
                      true_params['e']*0., # init with zero ellipticity
        ],
        kernel=adaptive_hmc)
    return samples, trace

  samples, trace = get_samples()
  print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  np.save("res/samples{}_shear-e{}_shear.npy".format(N*N, num_results), samples[0].numpy())
  np.save("res/samples{}_shear-e{}_e.npy".format(N*N, num_results), samples[1].numpy())

  # Display things

  plt.figure()
  plt.plot(samples[0][:]/10.)
  plt.axhline(custom_shear[0], color='C0', label='g1')
  plt.axhline(custom_shear[1], color='C1', label='g2')
  plt.legend()
  plt.savefig('res/shear.png')

  plt.figure()
  az.plot_pair(
    {'gamma1':samples[0].numpy()[:,0], 
     'gamma2':samples[0].numpy()[:,1]},
    var_names=["gamma1", "gamma2"],
    kind="kde",
    divergences=True,
    textsize=22,
    )

  plt.axvline(true_params['gamma'][0].numpy())
  plt.axhline(true_params['gamma'][1].numpy())
  plt.savefig('res/shear_countours.png', bbox_inches='tight')

  plt.figure()
  plt.subplot(121)
  plt.title('e1')
  for i in range(16):
    plt.plot(samples[1][:,i,0])

  plt.subplot(122)
  plt.title('e2')
  for i in range(16):
    plt.plot(samples[1][:,i,1])

  plt.savefig('res/e.png')

if __name__ == "__main__":
    app.run(main)