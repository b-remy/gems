import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np
import matplotlib.pyplot as plt

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

    # PSF model from galsim COSMOS catalog
    stamp_size = 64
    psf = galsim.Gaussian(0.06)

    interp_factor=2
    padding_factor=2
    Nk = stamp_size*interp_factor*padding_factor
    from galsim.bounds import _BoundsI
    bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

    imkpsf = psf.drawKImage(bounds=bounds,
                            scale=2.*_pi/(stamp_size*padding_factor*_scale),
                            recenter=False)

    kpsf = tf.cast(np.fft.fftshift(imkpsf.array.reshape(1, Nk, Nk//2+1), axes=1), tf.complex64)

    def model(batch_size=16, stamp_size=64):
        """Toy model
        """
        # stamp size
        nx = ny = stamp_size

        # pixel noise std
        sigma_e = 0.003

        # prior on Sersic index n
        log_l_n = ed.Normal(loc=.1*tf.ones(batch_size), scale=.39, name="n")
        n = tf.math.exp(log_l_n * _log10)

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
        profile = lp.sersic(n, half_light_radius=hlr, flux=F, nx=nx, ny=ny, scale=_scale)

        # Apply intrinsic ellipticity on profiles the image
        tfg1 = e[:, 0]
        tfg2 = e[:, 1]
        ims = tf.cast(tf.reshape(profile, (batch_size,stamp_size,stamp_size,1)), tf.float32)
        ims = galflow.shear(ims, tfg1, tfg2)

        # Apply same shear on all images
        ims = galflow.shear(ims, 
                            gamma[0]*tf.ones(batch_size),
                            gamma[1]*tf.ones(batch_size))

        # Convolve the image with the PSF
        profile = galflow.convolve(ims, kpsf,
                            zero_padding_factor=padding_factor,
                            interp_factor=interp_factor)[...,0]

        # Returns likelihood
        return ed.Normal(loc=profile, scale=sigma_e, name="obs")

    # Execute probabilistic program and record execution trace
    with ed.tape() as true_params:
        ims = model(16, 64)

    # Display things
    res = ims.numpy().reshape(4,4,64,64).transpose([0,2,1,3]).reshape([4*64,4*64])
    # plt.figure()
    # plt.imshow(res, cmap='gray_r')

    print("True shear:", true_params['gamma'])

    # Get the joint log prob
    log_prob = ed.make_log_joint_fn(model)

    log_prob(n=true_params['n'], 
            hlr=true_params['hlr'],
            gamma=true_params['gamma'],
            e=true_params['e'],
            obs=ims)

    # Let gamma, e, n and hlr free
    def target_log_prob_fn(n, hlr, gamma, e,):
        return log_prob(n=n, 
                hlr=hlr,
                gamma=gamma,
                e=e,
                obs=ims)

    adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=3,
        step_size=.0005)
    
    @tf.function
    def sample():
      samples, trace = tfp.mcmc.sample_chain(
          num_results=50000,
          num_burnin_steps=1,
          current_state=[true_params['n']*0.+1.,     # Init with 1.
                      true_params['hlr']*0.+1.,      # Inint with 1.
                      true_params['gamma']*0., # Init with just zero ellipticity
                      true_params['e']*0.],
          kernel=adaptive_hmc)
      return samples
    samples = sample()

    # Draw images using a sample from the chain
    with ed.condition(n=samples[0][0],
                    hlr=samples[1][0],
                    gamma=samples[2][0],
                    e=samples[3][0],
                    ):
        rec0 = model()


    with ed.condition(n=samples[0][-1],
                    hlr=samples[1][-1],
                    gamma=samples[2][-1],
                    e=samples[3][-1],):
        rec1 = model()

    im_rec0 = rec0.numpy().reshape(4,4,64,64).transpose([0,2,1,3]).reshape([4*64,4*64])
    im_rec1 = rec1.numpy().reshape(4,4,64,64).transpose([0,2,1,3]).reshape([4*64,4*64])

    plt.figure(figsize=[15,5])
    plt.subplot(131)
    plt.imshow(res, cmap='gray_r')
    plt.title('data')
    plt.subplot(132)
    plt.imshow(im_rec0, cmap='gray_r')
    plt.title('first sample from chain')
    plt.subplot(133)
    plt.imshow(im_rec1, cmap='gray_r')
    plt.title('last sample from chain')
    print("Diff 1st versus last sample", np.linalg.norm(im_rec0 - im_rec1))

    print("Last value shear:", samples[2][-1].numpy(), true_params['gamma'].numpy())

    plt.figure()
    plt.plot(samples[2][:])
    plt.axhline(true_params['gamma'][0].numpy(), color='C0', label='g1')
    plt.axhline(true_params['gamma'][1].numpy(), color='C1', label='g2')
    plt.legend()

    plt.figure()
    for i in range(16):
        plt.plot(samples[3][:,i,0])

    plt.figure()
    for i in range(16):
        plt.plot(samples[3][:,i,1])

    plt.show()

if __name__ == "__main__":
    app.run(main)