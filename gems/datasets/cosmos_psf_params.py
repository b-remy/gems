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

from gems.gen_obs import gpsf2ikpsf
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    
_CITATION = """
"""

_DESCRIPTION = """
"""

class CosmosPSFParamsConfig(tfds.core.BuilderConfig):
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
        super(CosmosPSFParamsConfig, self).__init__(
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
        
        
class CosmosPSFParams(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Cosmos dataset."""

    VERSION = tfds.core.Version("0.1.0")
    RELEASE_NOTES = {
        "0.1.0": "Initial release.",
    }

    BUILDER_CONFIGS = [CosmosPSFParamsConfig(name="25.2", sample="25.2")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(kappatng): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "mag_auto": tfds.features.Tensor(shape=[], dtype=tf.float32),
                    "z_phot": tfds.features.Tensor(shape=[], dtype=tf.float32),
                    "flux_radius": tfds.features.Tensor(shape=[], dtype=tf.float32),
                    "indices": tfds.features.Tensor(shape=[], dtype=tf.int32),
                    "kpsf_real": tfds.features.Tensor(
                        shape=[
                            self.builder_config.stamp_size,
                            self.builder_config.stamp_size,
                        ],
                        dtype=tf.float32),
                  "kpsf_imag": tfds.features.Tensor(
                        shape=[
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
                #gen_kwargs={
                #    "size": 25_000,
                #},
            ),
        ]
    
    def _generate_examples(self):
        """Yields examples."""
        # Loads the galsim COSMOS catalog
        
        max_hlr=0.5
        max_flux=150
        
        cat = galsim.COSMOSCatalog(sample='25.2',
                                   max_hlr=max_hlr,
                                   max_flux=max_flux,
                                   dir='/linkhome/rech/genpuc01/utb76xl/shear-projects/shear-projects/gems/paper/galsim_catalog/COSMOS_25.2_training_sample')

          
        PIXEL_SCALE = self.builder_config.pixel_scale
        STAMP_SIZE = self.builder_config.stamp_size
      
        interp_factor=2
        padding_factor=1
        Nk = STAMP_SIZE*interp_factor*padding_factor
        bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)   
        
        for j in tqdm(range(73050)): # 73050 is the size of the cosmos catalog with our cuts
          
          galr = cat.makeGalaxy(j, gal_type='real', noise_pad_size=0.8*PIXEL_SCALE*STAMP_SIZE)
          psf = galr.original_psf
          
          imkpsf = gpsf2ikpsf(psf=psf, interp_factor=1, padding_factor=1, stamp_size=STAMP_SIZE, im_scale=PIXEL_SCALE)[0]
          
          mag_auto = cat.param_cat['mag_auto'][cat.orig_index[j]]
          z_phot = cat.param_cat['zphot'][cat.orig_index[j]]
          flux_radius = cat.param_cat['flux_radius'][cat.orig_index[j]]
       
          yield "{}".format(j), {
                    "mag_auto": np.array(mag_auto).astype('float32'),
                    "z_phot": np.array(z_phot).astype('float32'),
                    "flux_radius": np.array(flux_radius).astype('float32'),
                    "indices": np.array(j).astype('int32'),
                    "kpsf_real": imkpsf.real.astype('float32'),
                    "kpsf_imag": imkpsf.imag.astype('float32'),
                }