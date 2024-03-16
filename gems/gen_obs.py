import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import numpy as np
tfd = tfp.distributions


import galsim
from galsim.bounds import _BoundsI

from gems.models import sersic2morph_model, dgm2morph_model, shear_fourier, convolve_fourier, dgm_model
from gems.ed_utils import make_value_setter, make_log_joint_fn

NUM_GAL = 144 # number of galaxies
NOISE_LEVEL = 0.01 # gaussian noise standard deviation
STAMP_SIZE = 128 # width of a postage stamp in pixels
PIXEL_SCALE = 0.03 # pixel size in arcsec

def gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale):
  Nk = stamp_size*interp_factor*padding_factor
  bounds = _BoundsI(-Nk//2, Nk//2-1, -Nk//2, Nk//2-1)
  imkpsf = psf.drawKImage(bounds=bounds,
                    scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                    recenter=False)
  # imkpsf = tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk])
  imkpsf = imkpsf.array.reshape([1, Nk, Nk])
  return imkpsf

def intertwine(img1, img2):
    """
    Args: [NUM_IMAGES, WIDTH, HEIGHT]
    Return: [NUM_IMAGES*2, WIDTH, HEIGHT]
    """
    NUM_IMAGES = img1.shape[0]
    
    img1 = tf.expand_dims(img1, 0)
    img2 = tf.expand_dims(img2, 0)

    pos_orig = tf.cast(tf.expand_dims(tf.one_hot(indices=tf.range(NUM_IMAGES)*2, depth=NUM_IMAGES*2),0), img1.dtype)
    pos_rot = tf.cast(tf.expand_dims(tf.one_hot(indices=tf.range(NUM_IMAGES)*2+1, depth=NUM_IMAGES*2),0), img1.dtype)
    
    img1_reshape = tf.transpose(img1, [0, 2, 3, 1])
    img2_reshape = tf.transpose(img2, [0, 2, 3, 1])
    
    ims_tot = tf.einsum('...i,...ik', img1_reshape, pos_orig)
    ims_tot = ims_tot + tf.einsum('...i,...ik', img2_reshape, pos_rot)
    
    ims_tot = tf.transpose(ims_tot, [0, 3, 1, 2])
    return ims_tot[0]
  
def intertwine_batch(img1, img2):
    """
    Args: [BATCH_SIZE, NUM_IMAGES, WIDTH, HEIGHT]
    Return: [BATCH_SIZE, NUM_IMAGES*2, WIDTH, HEIGHT]
    """
    NUM_IMAGES = img1.shape[1]
    
    #img1 = tf.expand_dims(img1, 0)
    #img2 = tf.expand_dims(img2, 0)

    pos_orig = tf.cast(tf.expand_dims(tf.one_hot(indices=tf.range(NUM_IMAGES)*2, depth=NUM_IMAGES*2),0), img1.dtype)
    pos_rot = tf.cast(tf.expand_dims(tf.one_hot(indices=tf.range(NUM_IMAGES)*2+1, depth=NUM_IMAGES*2),0), img1.dtype)
    
    img1_reshape = tf.transpose(img1, [0, 2, 3, 1])
    img2_reshape = tf.transpose(img2, [0, 2, 3, 1])
    
    ims_tot = tf.einsum('...i,...ik', img1_reshape, pos_orig)
    ims_tot = ims_tot + tf.einsum('...i,...ik', img2_reshape, pos_rot)
    
    ims_tot = tf.transpose(ims_tot, [0, 3, 1, 2])
    return ims_tot[0]

def double_rotate_images(images):
    """
    Args: [batch, NUM_IMAGES, WIDTH, HEIGHT]
    Return: [batch, NUM_IMAGES*2, WIDTH, HEIGHT]
    """
    
    NUM_IMAGES = images.shape[1]
    
    ims_rot = tf.reshape(images, [-1, images.shape[2], images.shape[3], 1])
    ims_rot = tf.image.rot90(ims_rot)
    ims_rot = tf.reshape(ims_rot, [images.shape[0], NUM_IMAGES, images.shape[2], images.shape[3]])
    
    pos_orig = tf.cast(tf.expand_dims(tf.one_hot(indices=tf.range(NUM_IMAGES)*2, depth=NUM_IMAGES*2),0), images.dtype)
    pos_rot = tf.cast(tf.expand_dims(tf.one_hot(indices=tf.range(NUM_IMAGES)*2+1, depth=NUM_IMAGES*2),0), images.dtype)
    
    ims_reshape = tf.transpose(images, [0, 2, 3, 1])
    ims_rot_reshape = tf.transpose(ims_rot, [0, 2, 3, 1])
    
    ims_tot = tf.einsum('...i,...ik', ims_reshape, pos_orig)
    ims_tot = ims_tot + tf.einsum('...i,...ik', ims_rot_reshape, pos_rot)
    
    ims_tot = tf.transpose(ims_tot, [0, 3, 1, 2])
    
    return ims_tot

def gen_obs(NUM_GAL, NOISE_LEVEL, STAMP_SIZE, PIXEL_SCALE, max_hlr=0.5, g1=None, g2=None, return_only_theta_obs=False):

    # Priors
    std_gamma = 0.15
    prior_gamma = tfd.Normal(loc=tf.zeros(2), scale=std_gamma*tf.ones(2))
    prior_z = tfd.Normal(loc=tf.zeros(16), scale=tf.ones(16))

    N = int(np.sqrt(NUM_GAL))
    # Load auto encoder weights
    encoder = hub.Module('../deep_galaxy_models/modules/vae_16/encoder')
    decoder = hub.Module('../deep_galaxy_models/modules/vae_16/decoder')
    code = hub.Module('../deep_galaxy_models/modules/latent_maf_16/code_sampler')
    
    cat = galsim.COSMOSCatalog(sample='25.2',
                               max_hlr=max_hlr,
                               dir='/linkhome/rech/genpuc01/utb76xl/shear-projects/shear-projects/gems/paper/galsim_catalog/COSMOS_25.2_training_sample')
    
    # select random cosmos galaxies indices
    gal_index = cat.selectRandomIndex(NUM_GAL//2)
    
    # Parameters to draw the PSF
    interp_factor=2
    padding_factor=1
    Nk = STAMP_SIZE*interp_factor*padding_factor
    bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)    

    # Generate observations
    im_real_list = []
    im_psf_list = []
    psfs = []
    psfs2 = []

    mag_auto_list = []
    z_phot_list = []
    flux_radius_list = []

    indices = []
    degrees = galsim.AngleUnit(np.pi / 180.)
    angle = galsim.Angle(90, unit=degrees)

    # sample theta from priors
    if g1==None and g2==None:
      gamma_samples = prior_gamma.sample(1)
    else:
      gamma_samples = tf.expand_dims(tf.cast(tf.convert_to_tensor([g1, g2]), tf.float32), 0)
      
    z_samples = prior_z.sample(NUM_GAL//2)

    for i in range(NUM_GAL//2):
        ind = gal_index[i]
        
        galr= cat.makeGalaxy(ind, gal_type='real', noise_pad_size=0.8*PIXEL_SCALE*STAMP_SIZE)

        psf = galr.original_psf
        psf2 = psf.rotate(angle)

        # PSF for reconvolution
        imkpsf = gpsf2ikpsf(psf=psf, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)
        imkpsf2 = gpsf2ikpsf(psf=psf2, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)

        #store PSF and rotated PSF
        psfs.append(imkpsf)
        psfs2.append(imkpsf2)
        indices.append(ind)

        mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
        z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
        flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

    mag_auto_g = tf.convert_to_tensor(mag_auto_list)
    z_phot_g = tf.convert_to_tensor(z_phot_list)
    flux_radius_g = tf.convert_to_tensor(flux_radius_list)

    imkpsfs = tf.cast(tf.concat(psfs, axis=0), tf.complex64)
    imkpsfs2 = tf.cast(tf.concat(psfs2, axis=0), tf.complex64)
    
    print()
    print(imkpsfs.shape)
    print(imkpsf2.shape)
    print()
    imkpsfs = intertwine(imkpsfs, imkpsfs2)
    
    # get random code
    z = code({'mag_auto':mag_auto_g, 
                'flux_radius':flux_radius_g, 
                'zphot':z_phot_g , 
                'random_normal':z_samples})

    reconstruction = decoder(z)

    ims = tf.reshape(reconstruction, (1, NUM_GAL//2, STAMP_SIZE, STAMP_SIZE))
    
    #imkpsfs = tf.reshape(imkpsfs, (1, NUM_GAL//2, STAMP_SIZE, STAMP_SIZE))    
    
    imkpsfs = tf.reshape(imkpsfs, (1, NUM_GAL, STAMP_SIZE, STAMP_SIZE))    

    # double and rotate each galaxy
    ims_tot = double_rotate_images(ims)
    
    #imkpsfs = double_rotate_images(imkpsfs)
    
    g1 = gamma_samples[:,0]
    g2 = gamma_samples[:,1]
    
    im_sheared = shear_fourier(ims_tot, g1, g2)

    ims = convolve_fourier(im_sheared, imkpsfs)
    
    obs = ims_tot + tf.random_normal([1, NUM_GAL, STAMP_SIZE, STAMP_SIZE]) * NOISE_LEVEL

    #imkpsfs = intertwine(imkpsfs, imkpsfs2)
    
    if not(return_only_theta_obs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ouput = sess.run([
                obs,
                gamma_samples,
                z_samples,
                imkpsfs,
            ])

        y, gamma_samples, z_samples, imkpsfs = ouput
        print()
        print("gen obs with g1={} g2={}".format(gamma_samples[:,0], gamma_samples[:,1]))
        print()
        # return observations, parameters and catalog selection
        return y, gamma_samples, z_samples, indices, imkpsfs, mag_auto_list, z_phot_list, flux_radius_list
      
    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ouput = sess.run([
                obs,
                gamma_samples,
            ])

        y, gamma_samples = ouput
        print()
        print("gen obs with g1={} g2={}".format(gamma_samples[:,0], gamma_samples[:,1]))
        print()
        # return observations, parameters and catalog selection
        return y, gamma_samples, indices
      

def gen_batch_obs(batch_size, NUM_GAL, NOISE_LEVEL, STAMP_SIZE, PIXEL_SCALE, max_hlr=0.6, g1=None, g2=None, return_only_theta_obs=False):
    
    batch_size = batch_size
    
    # select random cosmos galaxies indices
    gal_index = cat.selectRandomIndex(batch_size*NUM_GAL//2)
    
    # Parameters to draw the PSF
    interp_factor=2
    padding_factor=1
    Nk = STAMP_SIZE*interp_factor*padding_factor
    bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)    

    # Generate observations
    im_real_list = []
    im_psf_list = []
    psfs = []
    psfs2 = []

    mag_auto_list = []
    z_phot_list = []
    flux_radius_list = []

    indices = []
    degrees = galsim.AngleUnit(np.pi / 180.)
    angle = galsim.Angle(90, unit=degrees)

    # sample theta from priors
    gamma_samples = prior_gamma.sample(batch_size)  
    z_samples = prior_z.sample(batch_size*NUM_GAL//2)

    for i in range(batch_size*NUM_GAL//2):
        ind = gal_index[i]
        
        galr= cat.makeGalaxy(ind, gal_type='real', noise_pad_size=0.8*PIXEL_SCALE*STAMP_SIZE)

        psf = galr.original_psf
        psf2 = psf.rotate(angle)

        # PSF for reconvolution
        imkpsf = gpsf2ikpsf(psf=psf, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)
        #imkpsf2 = gpsf2ikpsf(psf=psf2, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)

        #store PSF and rotated PSF
        psfs.append(imkpsf)
        psfs2.append(imkpsf2)

        indices.append(ind)

        mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
        z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
        flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

    mag_auto_g = tf.convert_to_tensor(mag_auto_list)
    z_phot_g = tf.convert_to_tensor(z_phot_list)
    flux_radius_g = tf.convert_to_tensor(flux_radius_list)

    imkpsfs = tf.reshape(tf.cast(tf.concat(psfs, axis=0), tf.complex64),
                         (batch_size, NUM_GAL//2, STAMP_SIZE, STAMP_SIZE)
                        )
    
    imkpsfs2 = tf.reshape(tf.cast(tf.concat(psfs2, axis=0), tf.complex64),
                         (batch_size, NUM_GAL//2, STAMP_SIZE, STAMP_SIZE)
                         )
    

    imkpsfs = intertwine(imkpsfs, imkpsfs2)

    
    # get random code
    z = code({'mag_auto':mag_auto_g, 
                'flux_radius':flux_radius_g, 
                'zphot':z_phot_g , 
                'random_normal':z_samples})

    reconstruction = decoder(z)

    ims = tf.reshape(reconstruction, (batch_size, NUM_GAL//2, STAMP_SIZE, STAMP_SIZE))

    # double and rotate each galaxy
    ims_tot = double_rotate_images(ims)

    g1 = gamma_samples[:,0]
    g2 = gamma_samples[:,1]
    
    im_sheared = shear_fourier(ims_tot, g1, g2)

    ims = convolve_fourier(im_sheared, imkpsfs)
        
    obs = ims_tot + tf.random_normal([batch_size, NUM_GAL, STAMP_SIZE, STAMP_SIZE]) * NOISE_LEVEL

    ouput = sess.run([
        obs,
        gamma_samples,
        z_samples,
    ])

    y, gamma_samples, z_samples = ouput

    # return observations, parameters and catalog selection
    return y, gamma_samples, z_samples, indices