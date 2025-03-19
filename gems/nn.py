import tensorflow_hub as hub
import gems
from pathlib import Path
path = Path(gems.__file__).parents[2]

encoder = hub.Module(str(path / 'deep_galaxy_models/modules/vae_16/encoder'))
decoder = hub.Module(str(path / 'deep_galaxy_models/modules/vae_16/decoder'))
code = hub.Module(str(path / 'deep_galaxy_models/modules/latent_maf_16/code_sampler'))