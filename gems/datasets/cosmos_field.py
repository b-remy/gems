""" TensorFlow Dataset of COSMOS images. """
import tensorflow_datasets as tfds
import numpy as np
import galsim
from galsim.bounds import _BoundsI

import os

import tensorflow as tf

import time

import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow_datasets.core.utils import gcs_utils

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

import tensorflow_hub as hub
decoder = hub.Module('/linkhome/rech/genpuc01/utb76xl/shear-projects/shear-projects/deep_galaxy_models/modules/vae_16/decoder')
code = hub.Module('/linkhome/rech/genpuc01/utb76xl/shear-projects/shear-projects/deep_galaxy_models/modules/latent_maf_16/code_sampler')

from gems.models import shear_fourier, convolve_fourier
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    
_CITATION = """
"""

_DESCRIPTION = """
"""

class CosmosFieldConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Cosmos."""

    def __init__(self, *, sample="25.2", stamp_size=128, pixel_scale=0.03, num_gal=144, noise_level=0.01, **kwargs):
        """BuilderConfig for Cosmos.
        Args:
        sample: which Cosmos sample to use, "25.2".
        stamp_size: image stamp size in pixels.
        pixel_scale: pixel scale of stamps in arcsec.
        **kwargs: keyword arguments forwarded to super.
        """
        v1 = tfds.core.Version("0.1.0")
        super(CosmosFieldConfig, self).__init__(
            description=(
                "Cosmos stamps from %s sample in %d x %d resolution, %.2f arcsec/pixel."
                % (sample, stamp_size, stamp_size, pixel_scale)
            ),
            version=v1,
            **kwargs
        )
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.num_gal = num_gal
        self.noise_level = noise_level
        self.sample = sample
        
        
class CosmosField(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Cosmos dataset."""

    VERSION = tfds.core.Version("0.1.0")
    RELEASE_NOTES = {
        "0.1.0": "Initial release.",
    }

    BUILDER_CONFIGS = [CosmosFieldConfig(name="25.2", sample="25.2")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(kappatng): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Tensor(
                        shape=[
                            self.builder_config.num_gal,
                            self.builder_config.stamp_size,
                            self.builder_config.stamp_size,
                        ],
                        dtype=tf.float32,
                    ),
                    "z_prior": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal, 16],
                        dtype=tf.float32,
                    ),
                    "gamma": tfds.features.Tensor(
                        shape=[2],
                        dtype=tf.float32,
                    ),
                    "mag_auto": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal],
                        dtype=tf.float32,
                    ),
                    "z_phot": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal],
                        dtype=tf.float32,
                    ),
                    "flux_radius": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal],
                        dtype=tf.float32,
                    ),
                    "indices": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal],
                        dtype=tf.int32,
                    ),
                  "kpsf_real": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal,
                            self.builder_config.stamp_size,
                            self.builder_config.stamp_size,
                        ],
                        dtype=tf.float32),
                  "kpsf_imag": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal,
                            self.builder_config.stamp_size,
                            self.builder_config.stamp_size,
                        ],
                        dtype=tf.float32),
                  
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "image"),
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "size": 50_000,
                },
            ),
        ]
    
    def _generate_examples(self, size):
        """Yields examples."""
        # Loads the galsim COSMOS catalog
        
        # Priors
        std_gamma = 0.15
        prior_gamma = tfd.Normal(loc=tf.zeros(2), scale=std_gamma*tf.ones(2))
        prior_z = tfd.Normal(loc=tf.zeros(16), scale=tf.ones(16))
        
        # Load auto encoder weights
        encoder = hub.Module('../deep_galaxy_models/modules/vae_16/encoder')
        decoder = hub.Module('../deep_galaxy_models/modules/vae_16/decoder')
        code = hub.Module('../deep_galaxy_models/modules/latent_maf_16/code_sampler')
  
        NUM_GAL = self.builder_config.num_gal
        PIXEL_SCALE = self.builder_config.pixel_scale
        STAMP_SIZE = self.builder_config.stamp_size
        NOISE_LEVEL = self.builder_config.noise_level
        
        batch_size = 10
      
        ds = tfds.load('CosmosPSFParams/25.2', split='train')
        
        ds = ds.repeat()
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size*NUM_GAL)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds_train = iter(tfds.as_numpy(ds))
        
        mag_auto_in = tf.placeholder(shape=[batch_size*NUM_GAL], dtype=tf.float32)
        flux_radius_in = tf.placeholder(shape=[batch_size*NUM_GAL], dtype=tf.float32)  
        z_phot_in = tf.placeholder(shape=[batch_size*NUM_GAL], dtype=tf.float32)  
        imkpsfs_in = tf.placeholder(shape=[batch_size*NUM_GAL, STAMP_SIZE, STAMP_SIZE], dtype=tf.complex64) 
    
        # sample theta from priors
        gamma_samples = prior_gamma.sample(batch_size)  
        z_samples = prior_z.sample(batch_size*NUM_GAL)
          
        # get random code
        z = code({'mag_auto':mag_auto_in, 
                    'flux_radius':flux_radius_in, 
                    'zphot':z_phot_in, 
                    'random_normal':z_samples})
        
        reconstruction = decoder(z)
          
        ims_tot = tf.reshape(reconstruction, (batch_size, NUM_GAL, STAMP_SIZE, STAMP_SIZE))

        g1 = gamma_samples[:,0]
        g2 = gamma_samples[:,1]

        im_sheared = shear_fourier(ims_tot, g1, g2)
        
        im_sheared = tf.reshape(im_sheared, [1, batch_size*NUM_GAL, STAMP_SIZE, STAMP_SIZE])

        ims_tot = convolve_fourier(im_sheared, imkpsfs_in)
        
        ims_tot = tf.reshape(ims_tot, [batch_size, NUM_GAL, STAMP_SIZE, STAMP_SIZE])

        obs = ims_tot + tf.random_normal([batch_size, NUM_GAL, STAMP_SIZE, STAMP_SIZE]) * NOISE_LEVEL
        
        # initialize the variables
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        for i in tqdm(range(size//batch_size)):
          
          batch = next(ds_train)
          
          obs_, gamma_samples_, z_samples_ = sess.run([obs, gamma_samples, z_samples],
                                                     feed_dict={mag_auto_in:batch["mag_auto"],
                                                                flux_radius_in:batch["flux_radius"],
                                                                z_phot_in:batch["z_phot"],
                                                                imkpsfs_in:batch["kpsf_real"] + 1j * batch["kpsf_imag"],
                                                     })
          
          z_samples_ = z_samples_.reshape([batch_size, NUM_GAL, 16])
          
          for k in range(batch_size):
            yield "{}.{}".format(i,k), {
                      "image": obs_.astype('float32')[k],
                      "z_prior": z_samples_.astype('float32')[k],
                      "gamma": gamma_samples_.astype('float32')[k],
                      "mag_auto": batch["mag_auto"].reshape(batch_size, NUM_GAL)[k],
                      "z_phot": batch["z_phot"].reshape(batch_size, NUM_GAL)[k],
                      "flux_radius": batch["flux_radius"].reshape(batch_size, NUM_GAL)[k],
                      "indices": batch["indices"].reshape(batch_size, NUM_GAL)[k],
                      "kpsf_real": batch["kpsf_real"].reshape(batch_size, NUM_GAL, STAMP_SIZE, STAMP_SIZE)[k],
                      "kpsf_imag": batch["kpsf_imag"].reshape(batch_size, NUM_GAL, STAMP_SIZE, STAMP_SIZE)[k],
                  }