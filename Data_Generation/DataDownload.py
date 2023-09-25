import tensorflow as tf
import numpy as np
import os




DataFolder = '/home/mhafsati/KWS_EfficientNet/Data'
if not os.path.exists(DataFolder):
	os.makedirs(DataFolder)




assets = [("https://storage.googleapis.com/public-datasets-mswc/audio/en.tar.gz", os.path.join(DataFolder,"en"))
, ("https://storage.googleapis.com/public-datasets-mswc/audio/de.tar.gz", os.path.join(DataFolder,"de"))
, ("https://storage.googleapis.com/public-datasets-mswc/audio/fr.tar.gz", os.path.join(DataFolder,"fr"))
, ("https://storage.googleapis.com/public-datasets-mswc/audio/es.tar.gz", os.path.join(DataFolder,"es"))
, ("https://storage.googleapis.com/public-datasets-mswc/audio/it.tar.gz", os.path.join(DataFolder,"it"))
, ("https://storage.googleapis.com/public-datasets-mswc/audio/pt.tar.gz", os.path.join(DataFolder,"pt"))
, ("https://storage.googleapis.com/public-datasets-mswc/audio/nl.tar.gz", os.path.join(DataFolder,"nl"))
]

for asset,cache in assets:
    tf.keras.utils.get_file(origin=asset, untar=True, cache_subdir=cache)






