from absl import app
from absl import flags
import time
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import edward2 as ed

from gems.models import sersic_model, shear_fourier, convolve_fourier
from gems.ed_utils import make_log_joint_fn

from tqdm import tqdm

import numpy as np
from functools import partial

import galsim
from galsim.bounds import _BoundsI

import matplotlib.pyplot as plt

_log10 = tf.math.log(10.)
_scale = 0.03 # COSMOS pixel size in arcsec
_pi = np.pi
stamp_size = 128
noise_level = 0.01

N=4

flags.DEFINE_integer("n", 10, "number of interations")
flags.DEFINE_string("samples_path", None, "number of interations")
flags.DEFINE_boolean("MAP", False, "number of interations")
flags.DEFINE_string("catalog", "25.2", "number of interations")
FLAGS = flags.FLAGS

def gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale):
  Nk = stamp_size*interp_factor*padding_factor
  #bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)
  bounds = _BoundsI(-Nk//2, Nk//2-1, -Nk//2, Nk//2-1)
  imkpsf = psf.drawKImage(bounds=bounds,
                    scale=2.*np.pi/(stamp_size*padding_factor* im_scale),
                    recenter=False)
  imkpsf = tf.reshape(tf.convert_to_tensor(imkpsf.array, tf.complex64), [1, Nk, Nk])
  return imkpsf

def main(_):
  # Prepare results storage
  folder_name = 'gf_dgm_hmc'
  job_name = str(int(time.time()))
  if not os.path.isdir('./res'):
    os.mkdir('res')
    os.mkdir('res/'+folder_name)
  elif not os.path.isdir('./res/'+folder_name):
    os.mkdir('res/'+folder_name)

  os.mkdir("res/"+folder_name+"/{}".format(job_name))
  os.mkdir("res/"+folder_name+"/{}/params".format(job_name))

  begin = time.time()

  num_gal = N*N

  if FLAGS.catalog=='25.2':
    cat = galsim.COSMOSCatalog(sample='25.2',
                              dir='/Users/br263581/miniconda3/envs/gems/lib/python3.6/site-packages/galsim/share/COSMOS_25.2_training_sample/')
    # cat = galsim.COSMOSCatalog(sample='25.2',
    #                            dir='/gpfswork/rech/xdy/commun/galsim_catalogs/COSMOS_25.2_training_sample')
  # elif FLAGS.catalog=='23.5':
  #   cat = galsim.COSMOSCatalog(sample='23.5',
  #                              dir='/gpfswork/rech/xdy/commun/galsim_catalogs/COSMOS_23.5_training_sample')

  index = range(N*N)

  # Prepare parameters
  obs = []
  n = []
  flux = []
  hlr = []
  q_list = []
  phi_list = []
  psfs = []

  mag_auto_list = []
  z_phot_list = []
  flux_radius_list = []

  ind = 0

  indices = []

  # PSF parameters
  im_scale = 0.03
  interp_factor=1
  padding_factor=1

  degrees = galsim.AngleUnit(_pi / 180.)
  angle = galsim.Angle(90, unit=degrees)

  while len(obs) < num_gal:
    galp = cat.makeGalaxy(ind, gal_type='parametric')
    if cat.param_cat['use_bulgefit'][cat.orig_index[ind]] == 0:
      if galp.original.n < 0.4 or galp.original.half_light_radius > .2 or cat.param_cat['mag_auto'][cat.orig_index[ind]] < 23.5:
        ind += 1
      else:
        if False:
          ind+=1
        else:
          galr = cat.makeGalaxy(ind, gal_type='real', noise_pad_size=0)
          galp = cat.makeGalaxy(ind, gal_type='parametric')
          psf = galr.original_psf

          if indices.count(ind)==1:
            galr = galr.rotate(angle)
            # psf = psf.rotate(angle)

          imkpsf = gpsf2ikpsf(psf, interp_factor, padding_factor, stamp_size, im_scale)

          # Store sersic fit parameters
          n.append(galp.original.n)
          flux.append(galp.original.flux)
          hlr.append(galp.original.half_light_radius)
          q = cat.param_cat['sersicfit'][cat.orig_index[ind]][3]
          q_list.append(q)
          phi = cat.param_cat['sersicfit'][cat.orig_index[ind]][7]
          phi_list.append(phi)
          psfs.append(imkpsf)

          mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
          z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
          flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

          ims = galr.drawImage(nx=stamp_size, ny=stamp_size, scale=im_scale, method='no_pixel').array
          ims = tf.convert_to_tensor(ims, tf.float32)
          ims = tf.reshape(ims, (1, 1, stamp_size, stamp_size))


          g1 = -0.05
          g2 = +0.05

          im_sheared = shear_fourier(ims, g1, g2)

          ims = convolve_fourier(im_sheared, imkpsf)[0,0]

          obs_ = ims + tf.random.normal([stamp_size, stamp_size]) * noise_level

          obs_ = obs_.numpy()
          obs.append(obs_)

          indices.append(ind)
          print(ind, len(obs))

          if indices.count(ind)==2:
            ind += 1

    else:
      ind += 1


  obs_128 = tf.expand_dims(tf.stack(obs, axis=0), 0) # [1, batch, nx, ny]

  k = 10
  obs = tf.expand_dims(tf.stack(obs, axis=0), 0)[..., k:-k, k:-k] # [1, batch, nx, ny]
  n = tf.expand_dims(tf.stack(n, axis=0), 0)
  flux = tf.expand_dims(tf.stack(flux, axis=0), 0)
  hlr = tf.expand_dims(tf.stack(hlr, axis=0), 0)
  q_ = tf.expand_dims(tf.stack(q_list, axis=0), 0)
  phi = tf.expand_dims(tf.stack(phi_list, axis=0), 0)
  e1 = tf.cast((1-q_)/(1+q_) * tf.math.cos(2*phi), tf.float32)
  e2 = tf.cast((1-q_)/(1+q_) * tf.math.sin(2*phi), tf.float32)
  imkpsfs = tf.concat(psfs, axis=0)


  res = obs_128.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])

  plt.figure(figsize=(11,11))
  plt.title('Observations')
  s = 1e-2
  plt.imshow(np.arcsinh((res)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/gals.png")

  # Ground truth parameters
  true_hlr = hlr
  true_e = tf.stack([e1, e2], -1)
  true_gamma = tf.expand_dims(tf.convert_to_tensor([g1, g2]), 0)

  #TODO: does not work with batch size > 1 yet...
  batch_size = 1

  log_prob = make_log_joint_fn(partial(sersic_model,
                                      batch_size=batch_size, num_gal=N*N, stamp_size=stamp_size, scale=0.03, 
                                      sigma_e=noise_level, fixed_flux=True, kpsf=imkpsfs, fit_centroid=True,
                                      #  hlr=None, 
                                      n=n, 
                                      flux=flux, 
                                      # e=None, 
                                      # gamma=None, 
                                      display=False
                                      ))

  
  scale_hlr = .5
  scale_e = .5
  scale_gamma = .1
  scale_shift = 1.
  def target_log_prob_fn(hlr, gamma, e, shift):
    return log_prob(
          hlr=hlr*scale_hlr,
          gamma=gamma*scale_gamma, # trick to adapt the step size
          e=e*scale_e,
          shift=shift*scale_shift,
          obs=obs)


  num_results = FLAGS.n
  num_burnin_steps = 1

  adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    num_leapfrog_steps=3,
    # step_size=.00005)
    # step_size=.0001)
    step_size=.005)

  @tf.function
  def get_samples():
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
          # hlr, # init with prior mean
                      tf.math.log(true_hlr)/tf.math.log(10.)/scale_hlr,
                      # tf.zeros((1, 2)), # init with zero shear
                      # start_gamma,
                      true_gamma/scale_gamma*0.,
                      # tf.zeros((1, num_gal, 2)), # init with zero ellipticity
                      # start_e,
                      true_e,
                      true_e*0., # offset shift init to 0.

        ],
        kernel=adaptive_hmc)
    return samples, trace


  
  samples, trace = get_samples()
  print('accptance ratio:', trace.is_accepted.numpy().sum()/len(trace.is_accepted.numpy()))

  end = time.time()
  print('Time: {:.2f}'.format((end - begin)/60.))
  print('')

  hlr_est = samples[0].numpy()[:,0,:]
  gamma_est = samples[1].numpy()[:,0,:]*scale_gamma
  e_est = samples[2].numpy()[:,0,:]*scale_e
  shift_est = samples[3].numpy()[:,0,:]*scale_shift
  gamma_true = true_gamma.numpy()[0,:]


  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_gamma_{}_{}.npy".format(N*N, num_results, gamma_true[0], gamma_true[1]), gamma_est)
  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_e.npy".format(N*N, num_results), e_est)
  # np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_r.npy".format(N*N, num_results), hlr_est)
  np.save("res/"+folder_name+"/"+job_name+"/samples{}_{}_shift.npy".format(N*N, num_results), shift_est)



  plt.figure()
  plt.title('hlr')
  for i in range(5):
    plt.plot(hlr_est[:,i], label='{}'.format(i))
    #plt.axhline(true_hlr[0,i], color='gray')
    plt.axhline(np.log(true_hlr[0,i])/np.log(10.), color='gray')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/hlr.png")


  plt.figure()
  plt.plot(gamma_est)
  plt.axhline(gamma_true[0], color='C0', label='g1')
  plt.axhline(gamma_true[1], color='C1', label='g2')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/shear.png")

  plt.figure()
  plt.subplot(121)
  plt.title('e1')
  for i in range(5):
    # print(true_e.numpy().shape)
    # print(true_e.numpy()[0,i,0])
    plt.plot(e_est[:,i,0], label='{}'.format(i))
    plt.axhline(true_e.numpy()[0,i,0], color='gray')
  plt.legend()

  plt.subplot(122)
  plt.title('e2')
  for i in range(5):
    plt.plot(e_est[:,i,1], label='{}'.format(i))
    plt.axhline(true_e.numpy()[0,i,1], color='gray')
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/e.png")

  plt.figure()
  plt.title('offset x')
  for i in range(5):
    plt.plot(shift_est[:,i,0], label='{}'.format(i))
  plt.legend()
  plt.savefig("res/"+folder_name+"/"+job_name+"/shift.png")

  with ed.condition(hlr=tf.reduce_mean(hlr_est, axis=0, keepdims=True),
                    e=e_est.mean(axis=0, keepdims=True),
                    gamma=gamma_est.mean(axis=0, keepdims=True),
                    shift=shift_est.mean(axis=0, keepdims=True)):
    rec1 = sersic_model(batch_size=batch_size, 
                        sigma_e=noise_level,
                        stamp_size=stamp_size,
                        num_gal=N*N,
                        kpsf=imkpsfs, 
                        fixed_flux=True, 
                        n=n, flux=flux, hlr=hlr, 
                        fit_centroid=True, display=True)
  im_rec1 = rec1.numpy().reshape(N,N,stamp_size,stamp_size).transpose([0,2,1,3]).reshape([N*stamp_size,N*stamp_size])

  plt.figure(figsize=(11,11))
  plt.title('Mean posterior (sersic2morph)')
  s = 1e-2
  plt.imshow(np.arcsinh((im_rec1)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/mean_posterior.png")

  plt.figure(figsize=(11,11))
  plt.title('Mean posterior (sersic2morph)')
  s = 1e-2
  plt.imshow(np.arcsinh((im_rec1-res)/s)*s)
  plt.colorbar()
  plt.savefig("res/"+folder_name+"/"+job_name+"/residuals.png")

if __name__ == "__main__":
    app.run(main)