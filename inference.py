import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np

import os
import fnmatch
from absl import app
from absl import flags

import galsim
import galflow
lp = galflow.lightprofiles

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi

def main(_):

  y = tf.convert_to_tensor(np.load("data/stamp.npy"), dtype=tf.float32)
  print(y.shape)

  # INPUTS
  stamp_size = y.shape[0]
  # N = 5

  # stamp size
  nx = ny = stamp_size

  # pixel noise std
  sigma_e = 0.003
  # noise = ed.Normal(loc=tf.zeros((nx, ny)), scale=sigma_e, name="noise")

  # PSF model from galsim COSMOS catalog
  cat = galsim.COSMOSCatalog()
  psf = cat.makeGalaxy(2,  gal_type='real', noise_pad_size=0).original_psf

  interp_factor=2
  padding_factor=2
  Nk = stamp_size*interp_factor*padding_factor
  from galsim.bounds import _BoundsI
  bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

  imkpsf = psf.drawKImage(bounds=bounds,
                          scale=2.*_pi/(stamp_size*padding_factor*_scale),
                          recenter=False)

  kpsf = tf.cast(np.fft.fftshift(imkpsf.array.reshape(1, Nk, Nk//2+1), axes=1), tf.complex64)

  # Flux
  F = 16.693710205567005
    
  def model(target):
    # prior on Sersic index n
    log_l_n = ed.Normal(loc=.1, scale=.39, name="n")
    n = tf.math.exp(log_l_n * _log10)

    # prior on Sersic size half light radius
    log_l_hlr = ed.Normal(loc=-.68, scale=.3, name="hlr")
    hlr = tf.math.exp(log_l_hlr * _log10)

    # prior on shear
    gamma = ed.Normal(loc=tf.zeros((2)), scale=.09, name="shear")

    # Generate light profile
    profile = lp.sersic(n, scale_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)

    # Shear the image
    tfg1 = tf.reshape(tf.convert_to_tensor(gamma[0], tf.float32), (1))
    tfg2 = tf.reshape(tf.convert_to_tensor(gamma[1], tf.float32), (1))
    ims = tf.cast(tf.reshape(profile, (1,stamp_size,stamp_size,1)), tf.float32)
    ims = galflow.shear(ims, tfg1, tfg2)

    # Convolve the image with the PSF
    profile = galflow.convolve(ims, kpsf,
                        zero_padding_factor=padding_factor,
                        interp_factor=interp_factor)[0,...,0]

    # Evaluate likelihood
    # image = profile + noise
    Z = tf.math.pow(tf.math.sqrt(2*_pi) * sigma_e, stamp_size**2)
    l = tf.math.exp(tf.reduce_sum((target - profile)*(target - profile)) / (2*sigma_e*sigma_e)) / Z

    return l

  # Target log-probability function
  log_joint = ed.make_log_joint_fn(model)

  def target_log_prob_fn(n, hlr, g1, g2):
    # = state
    gamma = [g1, g2]
    return log_joint(n=n, hlr=hlr, shear=gamma, target=y)

  n = 1.
  hlr = 1.
  #gamma = tf.zeros(2)#.5 * tf.ones(2)
  g1 = 0.
  g2 = 0.
  print(target_log_prob_fn(n=n, hlr=hlr, g1=g1, g2=g2))

  n_init = tf.math.exp(tfd.Normal(loc=1., scale=.39).sample() * _log10)
  hlr_init = tf.math.exp(tfd.Normal(loc=-.68, scale=.3).sample() * _log10)
  gamma_init = tf.math.exp(tfd.MultivariateNormalDiag(loc=[0., 0.], scale_identity_multiplier=.09).sample() * _log10)
  target_log_prob = None
  grads_target_log_prob = None

  num_results = int(1e3)
  num_burnin_steps = int(1e2)
  adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=3,
        step_size=.001),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

  # Run the chain (with burn-in).
  #@tf.function
  def run_chain():
    # Run the chain (with burn-in).
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        #current_state=[n_init, hlr_init, gamma_init],
        current_state=[n, hlr, g1, g2],
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    sample_mean = tf.reduce_mean(samples)
    sample_stddev = tf.math.reduce_std(samples)
    is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
    return samples#sample_mean, sample_stddev, is_accepted
  
  samples_n, samples_hlr, samples_g1, samples_g2 = run_chain()
  print("n", tf.reduce_mean(samples_n).numpy())
  print("hlr", tf.reduce_mean(samples_hlr).numpy())
  print("g1", tf.reduce_mean(samples_g1).numpy())
  print("g2", tf.reduce_mean(samples_g2).numpy())

  # print("Let's run for {} burn-in and {} chain steps".format(num_burnin_steps, num_results))
  # sample_mean, sample_stddev, is_accepted = run_chain()

  # print(samples.shape)
  # print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
  #   sample_mean.numpy(), sample_stddev.numpy(), is_accepted.numpy()))

if __name__ == "__main__":
  app.run(main)