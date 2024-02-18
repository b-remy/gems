""" TensorFlow Dataset of COSMOS images. """
import tensorflow_datasets as tfds
import numpy as np
import galsim
from galsim.bounds import _BoundsI

import os
os.chdir('../..')

import tensorflow as tf

import time

import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow_datasets.core.utils import gcs_utils

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

import tensorflow_hub as hub
decoder = hub.Module('../deep_galaxy_models/modules/vae_16/decoder')
code = hub.Module('../deep_galaxy_models/modules/latent_maf_16/code_sampler')

from gems.models import shear_fourier, convolve_fourier
from gems.gen_obs import gpsf2ikpsf, intertwine, double_rotate_images

_CITATION = """
"""

_DESCRIPTION = """
"""

class CosmosConfig(tfds.core.BuilderConfig):
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
        super(CosmosConfig, self).__init__(
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
        
        
class Cosmos(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Cosmos dataset."""

    VERSION = tfds.core.Version("0.1.0")
    RELEASE_NOTES = {
        "0.1.0": "Initial release.",
    }

    BUILDER_CONFIGS = [CosmosConfig(name="25.2", sample="25.2")]

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
                        shape=[self.builder_config.num_gal//2, 16],
                        dtype=tf.float32,
                    ),
                    "gamma": tfds.features.Tensor(
                        shape=[2],
                        dtype=tf.float32,
                    ),
                    "indices": tfds.features.Tensor(
                        shape=[self.builder_config.num_gal//2],
                        dtype=tf.float32,
                    ),
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
                    "size": 30_000,
                },
            ),
        ]
    
    def _generate_examples(self, size):
        """Yields examples."""
        # Loads the galsim COSMOS catalog
        
        max_hlr=0.6
        
        # Priors
        std_gamma = 0.15
        prior_gamma = tfd.Normal(loc=tf.zeros(2), scale=std_gamma*tf.ones(2))
        prior_z = tfd.Normal(loc=tf.zeros(16), scale=tf.ones(16))
        
        # Load auto encoder weights
        encoder = hub.Module('../deep_galaxy_models/modules/vae_16/encoder')
        decoder = hub.Module('../deep_galaxy_models/modules/vae_16/decoder')
        code = hub.Module('../deep_galaxy_models/modules/latent_maf_16/code_sampler')

        cat = galsim.COSMOSCatalog(sample='25.2',
                                   max_hlr=max_hlr,
                                   dir='/linkhome/rech/genpuc01/utb76xl/shear-projects/shear-projects/gems/paper/galsim_catalog/COSMOS_25.2_training_sample')

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
          
        batch_size = 50 # fits into 32GB GPU 50
        NUM_GAL = self.builder_config.num_gal
        PIXEL_SCALE = self.builder_config.pixel_scale
        STAMP_SIZE = self.builder_config.stamp_size
        NOISE_LEVEL = self.builder_config.noise_level
        N = int(np.sqrt(NUM_GAL))
        
        for j in range(size//batch_size):

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
                #psfs2.append(imkpsf2)
                indices.append(ind)

                mag_auto_list.append(cat.param_cat['mag_auto'][cat.orig_index[ind]])
                z_phot_list.append(cat.param_cat['zphot'][cat.orig_index[ind]])
                flux_radius_list.append(cat.param_cat['flux_radius'][cat.orig_index[ind]])

            mag_auto_g = tf.convert_to_tensor(mag_auto_list)
            z_phot_g = tf.convert_to_tensor(z_phot_list)
            flux_radius_g = tf.convert_to_tensor(flux_radius_list)

            imkpsfs = tf.cast(tf.concat(psfs, axis=0), tf.complex64)
            #imkpsfs2 = tf.cast(tf.concat(psfs2, axis=0), tf.complex64)

            # get random code
            z = code({'mag_auto':mag_auto_g, 
                        'flux_radius':flux_radius_g, 
                        'zphot':z_phot_g , 
                        'random_normal':z_samples})

            reconstruction = decoder(z)

            ims = tf.reshape(reconstruction, (batch_size, NUM_GAL//2, STAMP_SIZE, STAMP_SIZE))
            imkpsfs = tf.reshape(imkpsfs, (batch_size, NUM_GAL//2, STAMP_SIZE, STAMP_SIZE))

            g1 = gamma_samples[:,0]
            g2 = gamma_samples[:,1]

            im_sheared = shear_fourier(ims, g1, g2)

            ims = convolve_fourier(im_sheared, imkpsfs)

            # double and rotate each galaxy
            ims_tot = double_rotate_images(ims)

            obs = ims_tot + tf.random_normal([batch_size, NUM_GAL, STAMP_SIZE, STAMP_SIZE]) * NOISE_LEVEL

            ouput = sess.run([
                obs,
                gamma_samples,
                z_samples,
            ])

            y, gamma_samples, z_samples = ouput
            
            z_samples = z_samples.reshape(batch_size, NUM_GAL//2, 16)
            indices = np.array(indices, np.float32).reshape(batch_size, NUM_GAL//2)
            
            for k in range(batch_size):
                print("{}.{}".format(i,k))
                yield "{}.{}".format(i,k), {
                    "image": y[k],
                    "z_prior": z_samples[k],
                    "gamma": gamma_samples[k],
                    "indices": indices[k]
                }