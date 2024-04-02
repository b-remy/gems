from gems.datasets import cosmos_field

import tensorflow_datasets as tfds

ds = tfds.load('CosmosField/25.2',split='train')