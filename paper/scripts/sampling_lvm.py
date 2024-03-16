#from absl import app
#from absl import flags

import os
os.chdir('../..')

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import numpy as np
tfd = tfp.distributions

import galsim
from galsim.bounds import _BoundsI
import galflow

import time
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

from gems.models import sersic2morph_model, dgm2morph_model, shear_fourier, convolve_fourier, dgm_model
from gems.ed_utils import make_value_setter, make_log_joint_fn
from gems.gen_obs import gen_obs

NUM_GAL = 144 # number of galaxies
NOISE_LEVEL = 0.01 # gaussian noise standard deviation
STAMP_SIZE = 128 # width of a postage stamp in pixels
PIXEL_SCALE = 0.03 # pixel size in arcsec
N = int(np.sqrt(NUM_GAL))

g1 = -0.05
g2 = +0.05

out = gen_obs(NUM_GAL, NOISE_LEVEL, STAMP_SIZE, PIXEL_SCALE, max_hlr=0.4, g1=g1, g2=g2)
    
y, true_gamma, z_samples, indices, imkpsfs, mag_auto_list, z_phot_list, flux_radius_list = out

mag_auto_list = np.repeat(np.array(mag_auto_list, np.float32), 2)
z_phot_list = np.repeat(np.array(z_phot_list, np.float32), 2)
flux_radius_list = np.repeat(np.array(flux_radius_list, np.float32), 2)

k = 10
obs_cropped = y[:,:,k:-k, k:-k]

#TODO: does not work with batch size > 1 yet...
batch_size = 2
obs_ = tf.repeat(obs_cropped, repeats=batch_size, axis=0)

log_prob = make_log_joint_fn(partial(dgm_model, 
                                  batch_size=batch_size, 
                                  sigma_e=NOISE_LEVEL, 
                                  stamp_size=STAMP_SIZE,
                                  num_gal=N*N, 
                                  kpsf=imkpsfs,
                                  mag_auto_list= mag_auto_list, 
                                  z_phot_list = z_phot_list, 
                                  flux_radius_list = flux_radius_list,
                                  fit_centroid = False))

s_gamma = .05
def target_log_prob_fn(prior_z, gamma):
    return log_prob(
      prior_z=prior_z,
      gamma=gamma*s_gamma,
      obs=obs_)

def loss_fn(lz, gamma):
    return - target_log_prob_fn(lz, gamma)
  
lz = tf.Variable(tf.zeros([batch_size, NUM_GAL,16]), trainable=True, dtype=tf.float32)
gamma = tf.Variable(tf.zeros((batch_size, 2)), trainable=True, dtype=tf.float32)

loss = loss_fn(lz, gamma)

# Define the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss, var_list=[lz, gamma])

# initialize the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    losses = []
    for i in tqdm(range(200)):
        _, l, lz_, g_ = sess.run([train, loss, lz, gamma])
        losses.append(l)

print('g_MAP:', g_*s_gamma)

# Initialize the HMC transition kernel.
#num_results = int(100)
#num_burnin_steps = int(100)
num_results = int(3000)
num_burnin_steps = int(150)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=3,
        step_size=.1),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
                      lz_,
                      g_,
        ],
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                                 pkr.inner_results.is_accepted]
    )
    return samples, trace
  
samples, trace = get_samples()

true_gamma = tf.cast(tf.convert_to_tensor(true_gamma), tf.float32)

lz_samples = samples[0]
gamma_samples = samples[1]*s_gamma
gamma_true = true_gamma[0,:]

start = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #lz_samples_, gamma_samples_, gamma_true_ = sess.run([lz_samples, gamma_samples, gamma_true])
    lz_samples_, gamma_samples_, gamma_true_, trace_sz, trace_ia = sess.run([lz_samples, gamma_samples, gamma_true, trace[0], trace[1]])
    
end = time.time()

print((end - start)/60)

np.save('./paper/scripts/results/mcmc_init/gamma.npy', gamma_samples_)
np.save('./paper/scripts/results/mcmc_init/lz.npy', lz_samples_)
np.save('./paper/scripts/results/mcmc_init/y.npy', y)
np.save('./paper/scripts/results/mcmc_init/imkpsfs.npy', imkpsfs)
np.save('./paper/scripts/results/mcmc_init/mag_auto_list.npy', mag_auto_list)
np.save('./paper/scripts/results/mcmc_init/z_phot_list.npy', z_phot_list)
np.save('./paper/scripts/results/mcmc_init/flux_radius_list.npy', flux_radius_list)
np.save('./paper/scripts/results/mcmc_init/trace_ia.npy', trace_ia)
np.save('./paper/scripts/results/mcmc_init/last_step_size.npy', trace_sz[-1:])