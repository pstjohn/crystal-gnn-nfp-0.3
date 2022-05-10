import math
import os
import shutil
from collections import Counter
from pathlib import Path

import nfp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from nfp.layers import RBFExpansion
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tqdm.auto import tqdm

tqdm.pandas()

inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")

data = pd.read_pickle(Path(inputs_dir, "20211227_all_data.p"))
max_distance = data['inputs'].apply(lambda x: x['distance'].max())
data = data[max_distance < 100]  # drop some problematic icsd structures

preprocessor = PymatgenPreprocessor()
preprocessor.from_json(Path(inputs_dir, "20211227_preprocessor.json"))

train, valid = train_test_split(data, test_size=2000, random_state=1)
valid, test = train_test_split(valid, test_size=0.5)

data['set'] = 'train'
data.loc[data.index.isin(valid.index), 'set'] = 'valid'
data.loc[data.index.isin(test.index), 'set'] = 'test'

model = tf.keras.models.load_model(
    '20211227_icsd_and_battery/best_model.hdf5',
    custom_objects={**nfp.custom_objects, **{'RBFExpansion': nfp.RBFExpansion}})

dataset = (
    tf.data.Dataset.from_generator(
        lambda: (row.inputs for _, row in data.iterrows()),
        output_signature=preprocessor.output_signature)
    .padded_batch(batch_size=128,
                  padding_values=preprocessor.padding_values)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

predictions = model.predict(dataset, verbose=1)

data['energy_predicted'] = predictions
data.to_csv('predicted_energies.csv')