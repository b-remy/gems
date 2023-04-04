import os
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from tqdm.notebook import tqdm
import numpy as np
tfd = tfp.distributions

os.chdir('..')

import galsim
from galsim.bounds import _BoundsI

cat = galsim.COSMOSCatalog(dir='/Users/br263581/miniconda3/envs/gems/lib/python3.6/site-packages/galsim/share/COSMOS_25.2_training_sample')

import numpy as np

stamp_size = 128
noise_level = 0.01

PIXEL_SCALE = 0.03
STAMP_SIZE = 128
interp_factor=2
padding_factor=1
Nk = STAMP_SIZE*interp_factor*padding_factor
bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)

from gems.models import shear_fourier, convolve_fourier

def gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale):
  Nk = stamp_size*interp_factor*padding_factor
  bounds = _BoundsI(-Nk//2, Nk//2-1, -Nk//2, Nk//2-1)
  imkpsf = psf.drawKImage(bounds=bounds,
                    scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                    recenter=False)
  # imkpsf = tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk])
  imkpsf = imkpsf.array.reshape([1, Nk, Nk])
  return imkpsf

# N = 75

# NUM_GAL = N*N

gauss_ = []
psf_gauss_ = []
im_list = []
im_real_list = []
im_gauss_list = []
im_psf_list = []
psfs = []
psfs_gauss = []
psfs_im = []

mag_auto_list = []
z_phot_list = []
flux_radius_list = []

indices = []
degrees = galsim.AngleUnit(np.pi / 180.)
angle = galsim.Angle(90, unit=degrees)

lp_ = []
psf_ = []

hlr_ = []
flux_ = []
jac_ = []

obs = np.load("results/obs/obs_0_2221.npy")
obs = np.concatenate([obs, np.load("results/obs/obs_2222_4430.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_4431_6699.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_6700_8952.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_8953_11222.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_11223_13398.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_13399_15701.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_15702_17944.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_17945_20318.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_20319_22628.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_22629_24908.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_24909_27237.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_24909_27237.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_27238_29436.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_29437_31652.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_31653_33860.npy")],1)
obs = np.concatenate([obs, np.load("results/obs/obs_33861_36084.npy")],1)

NUM_GAL = obs.shape[1]
print(obs.shape)

# assert 1==2

ind = 0
while len(indices) < NUM_GAL:
  galp = cat.makeGalaxy(ind, gal_type='parametric')
  if cat.param_cat['use_bulgefit'][cat.orig_index[ind]] == 0:
    if galp.original.n < .4 or galp.original.half_light_radius > 3. or cat.param_cat['mag_auto'][cat.orig_index[ind]] < 22.5:
      ind += 1
    else:
      galp = cat.makeGalaxy(ind, gal_type='parametric')
      im_real = galsim.ImageF(STAMP_SIZE, STAMP_SIZE, scale=PIXEL_SCALE)
      galr= cat.makeGalaxy(ind, gal_type='real', noise_pad_size=0.8*PIXEL_SCALE*STAMP_SIZE)
      psf = galr.original_psf
        
      psf_gauss = galsim.Gaussian(flux=1.0, fwhm=0.1)

      if indices.count(ind)==1:
        galr = galr.rotate(angle)
        galp = galp.rotate(angle)
        psf = psf.rotate(angle)
        psf_gauss = psf_gauss.rotate(angle)

      # hlr_.append(galp.original.half_light_radius)
      # flux_.append(galp.original.flux)
      # jac_.append(galp.jac)
    
      # gauss = galsim.Gaussian(half_light_radius=galp.original.half_light_radius, flux=galp.original.flux)
      # gauss = galsim.Transformation(gauss, galp.jac)

      # im_gauss_list.append(gauss.drawImage(nx=128, ny=128, scale=PIXEL_SCALE).array)

      # gauss_.append(gauss)
      # psf_gauss_.append(psf_gauss)
    
      # lp_.append(galr)
      psf_.append(psf)
      
      # real = galsim.Convolve(galr, psf)
      # real.drawImage(im_real, method='no_pixel', use_true_center=False)

      # PSF for the autocoder
      # imCp = psf.drawKImage(bounds=bounds,
      #                         scale=2.*np.pi/(Nk * PIXEL_SCALE / interp_factor),
      #                         recenter=False)
      # im_psf = np.abs(np.fft.fftshift(imCp.array, axes=0)).astype('float32')

      # PSF for reconvolution
      # imkpsf = gpsf2ikpsf(psf=psf, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)
      # psfs.append(imkpsf)
      
      # imkps_gauss = gpsf2ikpsf(psf=psf_gauss, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)
      # psfs_gauss.append(imkps_gauss)
    
      # psfs_im.append(psf.drawImage(scale=PIXEL_SCALE)) # 'no_pixel'

    
      # im_list.append(im_real)
      # im_real_list.append(im_real.array)
      # im_psf_list.append(im_psf)
      indices.append(ind)

      mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
      z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
      flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

      print(ind, len(im_real_list), cat.param_cat['mag_auto'][cat.orig_index[ind]], cat.param_cat['flux_radius'][cat.orig_index[ind]])

      if indices.count(ind)==2:
        ind += 1

  else:
    ind += 1

# Load the AutoEncoder
# encoder = hub.Module('../deep_galaxy_models/modules/vae_16/encoder')
# decoder = hub.Module('../deep_galaxy_models/modules/vae_16/decoder')

# im_real_list = np.stack(im_real_list, axis=0)
# im_gauss_list = np.stack(im_gauss_list, axis=0)
# im_psf_list = np.stack(im_psf_list, axis=0)
# imkpsfs = tf.cast(tf.concat(psfs, axis=0), tf.complex64)
# imkpsfs_gauss = tf.cast(tf.concat(psfs_gauss, axis=0), tf.complex64)

# psf_in = tf.placeholder(shape=[NUM_GAL, 256, 129, 1], dtype=tf.float32)
# im_in = tf.placeholder(shape=[NUM_GAL, 128, 128, 1], dtype=tf.float32)

# code = encoder({'input':im_in, 'psf':psf_in})
# reconstruction = decoder(code)

g1 = -0.03
g2 = +0.03

# COSMOS galaxy & COSMOS PSF
# ims = tf.reshape(reconstruction, (1, NUM_GAL, STAMP_SIZE, STAMP_SIZE))
# im_sheared = shear_fourier(ims, g1, g2)
# ims = convolve_fourier(im_sheared, imkpsfs)

"""
# COSMOS galaxy & Gaussian PSF
ims_ = tf.reshape(reconstruction, (1, NUM_GAL, STAMP_SIZE, STAMP_SIZE))
im_sheared = shear_fourier(ims_, g1, g2)
ims_GPSF = convolve_fourier(im_sheared, imkpsfs_gauss)

# Gaussian galaxy & Gaussian PSF
ims_gauss_ = tf.reshape(im_gauss_list, (1, NUM_GAL, STAMP_SIZE, STAMP_SIZE))
ims_gauss_sheared = shear_fourier(ims_gauss_, g1, g2)
ims_gauss_ = convolve_fourier(ims_gauss_sheared, imkpsfs_gauss)

# Gaussian galaxy & COSMOS PSF
ims_gauss = tf.reshape(im_gauss_list, (1, NUM_GAL, STAMP_SIZE, STAMP_SIZE))
ims_gauss_sheared = shear_fourier(ims_gauss, g1, g2)
ims_gauss_CPSF = convolve_fourier(ims_gauss_sheared, imkpsfs)
"""



# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# y = sess.run(ims, feed_dict={psf_in:im_psf_list.reshape((NUM_GAL,256,129,1)), im_in:im_real_list.reshape((NUM_GAL,128,128,1))})

"""
# y, y_GPSF = sess.run([ims, ims_GPSF], feed_dict={psf_in:im_psf_list.reshape((NUM_GAL,256,129,1)), im_in:im_real_list.reshape((NUM_GAL,128,128,1))})
# y_gauss, y_gauss_CPSF = sess.run([ims_gauss_, ims_gauss_CPSF])
"""

# Add Gaussian noise on images
# y = y + np.random.normal(size=y.shape) * noise_level
"""
# y_gauss = y_gauss + np.random.normal(size=y.shape) * noise_level
# y_gauss_CPSF = y_gauss_CPSF + np.random.normal(size=y.shape) * noise_level
# y_GPSF = y_GPSF + np.random.normal(size=y.shape) * noise_level
"""

def calibrate_hsm(e1, e2):
    e1 = np.stack(e1,0)
    e2 = np.stack(e2,0)

    R1 = 1-e1.std()**2
    R2 = 1-e2.std()**2

    e1 = e1/2/R1
    e2 = e2/2/R2
    return e1, e2

def plot_error_hsm(e1, e2, color='red'):
    plt.hlines(e2.mean(), e1.mean() - e1.std()/np.sqrt(e1.shape[0]), 
       e1.mean() + e1.std()/np.sqrt(e1.shape[0]), 
       color=color, 
       label='HSM (1 sigma)')
    plt.vlines(e1.mean(), e2.mean() - e2.std()/np.sqrt(e2.shape[0]), 
           e2.mean() + e2.std()/np.sqrt(e2.shape[0]), 
           color=color)

    # 2 sigma
    plt.hlines(e2.mean(), e1.mean() - 2*e1.std()/np.sqrt(e1.shape[0]), 
       e1.mean() + 2*e1.std()/np.sqrt(e1.shape[0]), 
           color=color, linestyle=':',  alpha=1., 
           label='HSM (2 sigma)')
    plt.vlines(e1.mean(), e2.mean() - 2*e2.std()/np.sqrt(e2.shape[0]), 
           e2.mean() + 2*e2.std()/np.sqrt(e2.shape[0]), 
           color=color, linestyle=':',  alpha=1., )


## Gaussian galaxy & Gaussian PSF
"""
print("Processing COSMOS galaxy & COSMOS PSF")
n_fail = 0
corr_e1 = []
corr_e2 = []
for i in tqdm(range(N*N)):
    psf_gauss = psf_gauss_[i]
    
    final_image = galsim.Image(y_gauss[0,i], scale=PIXEL_SCALE)
    final_epsf_image = psf_gauss.drawImage(nx=128, ny=128, scale=PIXEL_SCALE)

    result = galsim.hsm.EstimateShear(final_image, final_epsf_image, sky_var=noise_level, shear_est='REGAUSS', strict=False)
    
    if result.error_message != "":
        print("fail, %d"%i)
        n_fail += 1
    else:
        corr_e1.append(result.corrected_e1)
        corr_e2.append(result.corrected_e2)

e1, e2 = calibrate_hsm(corr_e1, corr_e2)

plot_error_hsm(e1, e2, color='tab:blue')
plt.scatter(x=g1, y=g2, marker='x', s=100, color='k', label='ground truth')
plt.legend()
plt.xlim([-0.2, 0.2])
plt.ylim([-0.2, 0.2])
plt.title('Gaussian galaxy & Gaussian PSF ({})'.format(e1.shape[0]))

## Gaussian galaxy & COSMOS PSF
print("Processing COSMOS galaxy & COSMOS PSF")
n_fail = 0
corr_e1 = []
corr_e2 = []

for i in tqdm(range(N*N)):
    psf = psf_[i]
    final_image = galsim.Image(y_gauss_CPSF[0,i], scale=PIXEL_SCALE)
    final_epsf_image = psf.drawImage(scale=PIXEL_SCALE)

    result = galsim.hsm.EstimateShear(final_image, final_epsf_image, sky_var=noise_level, shear_est='REGAUSS', strict=False)
  
    if result.error_message != "":
        print("fail, %d"%i)
        n_fail += 1
    else:
        corr_e1.append(result.corrected_e1)
        corr_e2.append(result.corrected_e2)

e1, e2 = calibrate_hsm(corr_e1, corr_e2)

plt.figure()
plot_error_hsm(e1, e2, color='tab:blue')
plt.scatter(x=g1, y=g2, marker='x', s=100, color='k', label='ground truth')
plt.legend()
plt.xlim([-0.1, 0.1])
plt.ylim([-0.1, 0.1])
plt.title('Gaussian galaxy & COSMOS PSF ({})'.format(e1.shape[0]))

## COSMOS galaxy & Gaussian PSF
n_fail = 0
corr_e1 = []
corr_e2 = []

for i in tqdm(range(N*N)):
    psf_gauss = psf_gauss_[i]

    final_image = galsim.Image(y_GPSF[0,i], scale=PIXEL_SCALE)
    final_epsf_image = psf_gauss.drawImage(scale=PIXEL_SCALE)

    result = galsim.hsm.EstimateShear(final_image, final_epsf_image, sky_var=noise_level, shear_est='REGAUSS', strict=False)
    
    if result.error_message != "":
        print("fail, %d"%i)
        n_fail += 1
    else:
        corr_e1.append(result.corrected_e1)
        corr_e2.append(result.corrected_e2)

e1, e2 = calibrate_hsm(corr_e1, corr_e2)

plt.figure()
plot_error_hsm(e1, e2, color='tab:blue')
plt.scatter(x=g1, y=g2, marker='x', s=100, color='k', label='ground truth')
plt.legend()
plt.xlim([-0.1, 0.1])
plt.ylim([-0.1, 0.1])
plt.title('COSMOS galaxy & Gaussian PSF ({})'.format(e1.shape[0]))
"""

## COSMOS galaxy and COSMOS PSF
print("Starting processing {} galaxies with HSM...".format(NUM_GAL))

n_fail = 0
corr_e1 = []
corr_e2 = []

y = obs
for i in range(NUM_GAL):
    psf = psf_[i]
    
    final_image = galsim.Image(y[0,i], scale=PIXEL_SCALE)
    final_epsf_image = psf.drawImage(scale=PIXEL_SCALE)

    result = galsim.hsm.EstimateShear(final_image, final_epsf_image, sky_var=noise_level, shear_est='REGAUSS', strict=False)
    
    if result.error_message != "":
        # print("fail, %d"%i)
        n_fail += 1
    else:
        corr_e1.append(result.corrected_e1)
        corr_e2.append(result.corrected_e2)
    
    if i%1000 == 0:
      e1, e2 = calibrate_hsm(corr_e1, corr_e2)      

      m1 = e1.mean()/g1 - 1
      merr1 = e1.std()/np.sqrt(e1.shape[0]) / g1
      m2 = e2.mean()/g2 - 1
      merr2 = e2.std()/np.sqrt(e2.shape[0]) / g2

      print("{} galaxies".format(i))
      # print("m1 = {} +/- {}".format(m1, abs(merr1)))
      # print("m2 = {} +/- {}".format(m2, abs(merr2)))

      if 2*abs(merr1) < abs(m1) and 2*abs(merr2) < abs(m2):
        print("2 sigma bias identified from about {} galaxies".format(i))
        print("m1 = {} +/- {}".format(m1, 2*abs(merr1)))
        print("m2 = {} +/- {}".format(m2, 2*abs(merr2)))

      if 3*abs(merr1) < abs(m1) and 3*abs(merr2) < abs(m2):
        print("3 sigma bias identified from about {} galaxies".format(i))
        print("m1 = {} +/- {}".format(m1, 3*abs(merr1)))
        print("m2 = {} +/- {}".format(m2, 3*abs(merr2)))


e1, e2 = calibrate_hsm(corr_e1, corr_e2)

plt.figure()
plot_error_hsm(e1, e2, color='tab:blue')
plt.scatter(x=g1, y=g2, marker='x', s=100, color='k', label='ground truth')
plt.legend()
plt.xlim([-0.1, 0.1])
plt.ylim([-0.1, 0.1])
plt.title('COSMOS galaxy and COSMOS PSF  ({})'.format(e1.shape[0]))
# plt.savefig('HSM_{}.png'.format(NUM_GAL))
plt.show()