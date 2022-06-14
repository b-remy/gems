import tensorflow as tf
import tensorflow_hub as hub

# Load autoencoder model
encoder = hub.load('../deep_galaxy_models/modules/vae_16/encoder')
decoder = hub.load('../deep_galaxy_models/modules/vae_16/decoder')

# Load sampler model
code_sampler = hub.load('../deep_galaxy_models/modules/latent_maf_16/code_sampler')

def encode(input, psf):
    return encoder.signatures['default'](input=input, psf=psf)['default']

def decode(z):
    return decoder.signatures['default'](z)['default']

def code_sample(mag_auto, flux_radius, zphot, random_normal):
  """
  mag_auto, flux_radius, zphot: [batch_size,]
  random_normal: [batch_size, 16]
  """
  sample = code_sampler.signatures['default'](mag_auto=mag_auto, 
                           flux_radius=flux_radius, 
                           zphot=zphot , 
                           random_normal=random_normal)['default']
  return sample
