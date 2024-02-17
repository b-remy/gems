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

NUM_GAL = 144 # number of galaxies
NOISE_LEVEL = 0.01 # gaussian noise standard deviation
STAMP_SIZE = 128 # width of a postage stamp in pixels
PIXEL_SCALE = 0.03 # pixel size in arcsec
N = int(np.sqrt(NUM_GAL))

# Load auto encoder weights
encoder = hub.Module('../deep_galaxy_models/modules/vae_16/encoder')
decoder = hub.Module('../deep_galaxy_models/modules/vae_16/decoder')

def gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale):
  Nk = stamp_size*interp_factor*padding_factor
  bounds = _BoundsI(-Nk//2, Nk//2-1, -Nk//2, Nk//2-1)
  imkpsf = psf.drawKImage(bounds=bounds,
                    scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                    recenter=False)
  # imkpsf = tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk])
  imkpsf = imkpsf.array.reshape([1, Nk, Nk])
  return imkpsf

cat = galsim.COSMOSCatalog(sample='25.2',
                            dir='/linkhome/rech/genpuc01/utb76xl/shear-projects/shear-projects/gems/paper/galsim_catalog/COSMOS_25.2_training_sample')

interp_factor=2
padding_factor=1
Nk = STAMP_SIZE*interp_factor*padding_factor
bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)    

begin = time.time()


# Generate observations
im_real_list = []
im_psf_list = []
psfs = []

mag_auto_list = []
z_phot_list = []
flux_radius_list = []

indices = []
degrees = galsim.AngleUnit(np.pi / 180.)
angle = galsim.Angle(90, unit=degrees)

ind = 0
while len(im_real_list) < NUM_GAL:
    galp = cat.makeGalaxy(ind, gal_type='parametric')
    if cat.param_cat['use_bulgefit'][cat.orig_index[ind]] == 0:
        if galp.original.n < .4 or galp.original.half_light_radius > 3. or cat.param_cat['mag_auto'][cat.orig_index[ind]] < 22.5:
            ind += 1
        else:
            galp = cat.makeGalaxy(ind, gal_type='parametric')
            im_real = galsim.ImageF(STAMP_SIZE, STAMP_SIZE, scale=PIXEL_SCALE)
            galr= cat.makeGalaxy(ind, gal_type='real', noise_pad_size=0.8*PIXEL_SCALE*STAMP_SIZE)
            psf = galr.original_psf

            if indices.count(ind)==1:
                galr = galr.rotate(angle)
                psf = psf.rotate(angle)

            real = galsim.Convolve(psf, galr)
            real.drawImage(im_real, method='no_pixel', use_true_center=False)

            # PSF for the autocoder
            imCp = psf.drawKImage(bounds=bounds,
                                    scale=2.*np.pi/(Nk * PIXEL_SCALE / interp_factor),
                                    recenter=False)
            im_psf = np.abs(np.fft.fftshift(imCp.array, axes=0)).astype('float32')

            # PSF for reconvolution
            imkpsf = gpsf2ikpsf(psf=psf, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)
            psfs.append(imkpsf)

            im_real_list.append(im_real.array)
            im_psf_list.append(im_psf)
            indices.append(ind)

            mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
            z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
            flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

            print(ind, len(im_real_list))

            if indices.count(ind)==2:
                ind += 1
    else:
        ind += 1

im_real_list = np.stack(im_real_list, axis=0)
im_psf_list = np.stack(im_psf_list, axis=0)

imkpsfs = tf.cast(tf.concat(psfs, axis=0), tf.complex64)

psf_in = tf.placeholder(shape=[NUM_GAL, 256, 129, 1], dtype=tf.float32)
im_in = tf.placeholder(shape=[NUM_GAL, 128, 128, 1], dtype=tf.float32)

code = encoder({'input':im_in, 'psf':psf_in})
reconstruction = decoder(code)

ims = tf.reshape(reconstruction, (1, NUM_GAL, STAMP_SIZE, STAMP_SIZE))

g1 = -0.03
g2 = +0.03
im_sheared = shear_fourier(ims, g1, g2)

ims = convolve_fourier(im_sheared, imkpsfs)

obs = ims + tf.random_normal([1, NUM_GAL, STAMP_SIZE, STAMP_SIZE]) * NOISE_LEVEL

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    y = sess.run(obs, feed_dict={psf_in:im_psf_list.reshape((NUM_GAL,256,129,1)), im_in:im_real_list.reshape((NUM_GAL,128,128,1))})
    #np.save('/content/drive/MyDrive/GEMS/obs16AE.npy', y)

    im_obs = y.reshape(N,N,STAMP_SIZE,STAMP_SIZE).transpose([0,2,1,3]).reshape([N*(STAMP_SIZE),N*(STAMP_SIZE)])
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
                                  interp_factor = interp_factor,
                                  padding_factor = padding_factor,
                                  mag_auto_list=mag_auto_list, 
                                  z_phot_list=z_phot_list, 
                                  flux_radius_list=flux_radius_list,
                                  fit_centroid=False))

s_gamma = .05
def target_log_prob_fn(prior_z, gamma):
    return log_prob(
      prior_z=prior_z,
      gamma=gamma*s_gamma,
      obs=obs_)


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
num_results = int(7_000)
num_burnin_steps = int(200)
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
        kernel=adaptive_hmc)
    return samples, trace
  
samples, trace = get_samples()

true_gamma = tf.expand_dims(tf.convert_to_tensor([g1, g2]), 0)

lz_samples = samples[0]
gamma_samples = samples[1]*s_gamma
gamma_true = true_gamma[0,:]

start = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lz_samples_, gamma_samples_, gamma_true_ = sess.run([lz_samples, gamma_samples, gamma_true])
    
end = time.time()

print((end - start)/60)

np.save('./paper/scripts/results/gamma', gamma_samples_)
np.save('./paper/scripts/results/lz', lz_samples_)
np.save('./paper/scripts/results/im_obs', im_obs)