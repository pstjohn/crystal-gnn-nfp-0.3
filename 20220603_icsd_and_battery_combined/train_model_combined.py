import math
import os
import shutil
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import nfp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from nfp.layers import RBFExpansion
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tqdm.auto import tqdm

from preprocess import preprocessor

tqdm.pandas()

inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")
data = pd.read_pickle(Path(inputs_dir, "20220603_outliers_removed.p"))
data.loc[data.hit_upper_bound, "volperatom"] = np.nan
# data = data[~data["hit_upper_bound"]]


# data = data.head(1000)


max_atomic_num = 84

composition_set = data.composition.isin(
    pd.Series(data.composition.unique()).sample(100, random_state=1)
)
test_composition = data[composition_set]
train_composition = data[~composition_set]

train, valid = train_test_split(
    train_composition,
    test_size=3000,
    random_state=1,
    stratify=train_composition["type"],
)
valid, test = train_test_split(
    train_composition,
    test_size=0.5,
    random_state=2,
    stratify=train_composition["type"],
)


# train = train.head(5000)
# valid = valid.head(100)
# data = data.head(1000)


def calculate_output_bias(site_counts, target):
    """ We can get a reasonable guess for the output bias by just assuming the crystal's
     volume is a linear sum over it's element types """
    # This just converts to a count of each element by crystal

    # Linear regression assumes a sum, while we average over sites in the neural network
    # Here, we make the regression target the total volume, not the site-averaged volume
    num_sites = site_counts.sum(1)
    total_quantity = target * num_sites

    # Do the least-squares regression, and stack on zeros for the mask and unknown
    # tokens
    output_bias = np.linalg.lstsq(site_counts, total_quantity, rcond=None)[0]
    return output_bias


def inputs_generator(split):
    for _, row in split.iterrows():
        inputs = row.inputs
        inputs["input_vol"] = row.scaled_input_volperatom
        inputs["true_vol"] = row.volperatom
        yield row.inputs, (
            np.atleast_1d(row.volperatom),
            np.atleast_1d(row.energyperatom),
        )


def build_dataset(split, batch_size):
    return (
        tf.data.Dataset.from_generator(
            partial(inputs_generator, split),
            output_signature=(
                {
                    **preprocessor.output_signature,
                    "input_vol": tf.TensorSpec(shape=(), dtype=tf.float32),
                    "true_vol": tf.TensorSpec(shape=(), dtype=tf.float32),
                },
                (
                    tf.TensorSpec((1,), dtype=tf.float32),
                    tf.TensorSpec((1,), dtype=tf.float32),
                ),
            ),
        )
        .cache()
        .shuffle(buffer_size=min(len(split), 1000))
        .padded_batch(
            batch_size=batch_size,
            padding_values=(
                {
                    **preprocessor.padding_values,
                    "input_vol": tf.constant(np.nan, dtype=tf.float32),
                    "true_vol": tf.constant(np.nan, dtype=tf.float32),
                },
                (
                    tf.constant(np.nan, dtype=tf.float32),
                    tf.constant(np.nan, dtype=tf.float32),
                ),
            ),
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


train_nona = train.dropna(subset=["volperatom", "energyperatom"])

# Calculate an initial guess for the output bias
site_counts = (
    train_nona.inputs.progress_apply(lambda x: pd.Series(Counter(x["site"])))
    .reindex(columns=np.arange(max_atomic_num))
    .fillna(0)
)

energy_bias = calculate_output_bias(site_counts, train_nona["energyperatom"])
volume_bias = calculate_output_bias(site_counts, train_nona["volperatom"])

assert not np.isnan(volume_bias).any()
assert not np.isnan(energy_bias).any()


batch_size = 64
train_dataset = build_dataset(train, batch_size=batch_size)
valid_dataset = build_dataset(valid, batch_size=batch_size)


class GnnModel(tf.keras.Model):
    def __init__(
        self, output_bias, max_atomic_num, embed_dimension, num_messages
    ) -> None:
        super().__init__()
        self.atom_embedding = layers.Embedding(
            max_atomic_num, embed_dimension, name="site_embedding", mask_zero=True
        )
        self.atom_mean = layers.Embedding(
            max_atomic_num,
            1,
            name="site_mean",
            mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(output_bias),
        )
        self.rbf_distance = RBFExpansion(
            dimension=128, init_max_distance=7, init_gap=30, trainable=True
        )
        self.bond_embedding = layers.Dense(embed_dimension)
        self.atom_offset = layers.Dense(
            1,
            name="site_energy_offset",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1e-6, seed=None
            ),
        )
        self.output_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.edge_updates = [nfp.EdgeUpdate() for _ in range(num_messages)]
        self.node_updates = [nfp.NodeUpdate() for _ in range(num_messages)]

    def call(self, inputs):

        site_class, distances, connectivity = inputs
        atom_state = self.atom_embedding(site_class)
        atom_mean = self.atom_mean(site_class)
        rbf_distance = self.rbf_distance(distances)
        bond_state = self.bond_embedding(rbf_distance)

        atom_mask = self.atom_embedding.compute_mask(site_class)
        bond_mask = self.rbf_distance.compute_mask(distances)

        for edge_update, node_update in zip(self.edge_updates, self.node_updates):
            bond_state += edge_update(
                [atom_state, bond_state, connectivity],
                mask=[atom_mask, bond_mask, None],
            )
            atom_state += node_update(
                [atom_state, bond_state, connectivity],
                mask=[atom_mask, bond_mask, None],
            )

        atom_state = self.atom_offset(atom_state)
        atom_mean += atom_state
        return self.output_pool(atom_mean, mask=atom_mask)


class CombinedModel(tf.keras.Model):
    def __init__(
        self, volume_bias, energy_bias, max_atomic_num, embed_dimension, num_messages
    ):
        super().__init__()

        self.volume_model = GnnModel(
            volume_bias, max_atomic_num, embed_dimension, num_messages
        )
        self.energy_model = GnnModel(
            energy_bias, max_atomic_num, embed_dimension, num_messages
        )

    def call(self, inputs, training=None):

        site_class, distances, connectivity, input_vol, true_vol = inputs

        pred_volperatom = self.volume_model([site_class, distances, connectivity])

        if training:
            vol = tf.where(
                tf.math.is_nan(true_vol), tf.stop_gradient(pred_volperatom), true_vol
            )
            scaled_dist = distances * tf.math.pow(vol / input_vol, 1.0 / 3.0)

        else:
            scaled_dist = distances * tf.math.pow(
                pred_volperatom / input_vol, 1.0 / 3.0
            )

        pred_energyperatom = self.energy_model([site_class, scaled_dist, connectivity])
        return pred_volperatom, pred_energyperatom


# Keras model
site_class = layers.Input(shape=[None], dtype=tf.int64, name="site")
distances = layers.Input(shape=[None], dtype=tf.float32, name="distance")
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
input_vol = layers.Input(shape=[1], dtype=tf.float32, name="input_vol")
true_vol = layers.Input(shape=[1], dtype=tf.float32, name="true_vol")

input_tensors = [site_class, distances, connectivity, input_vol, true_vol]

embed_dimension = 256
num_messages = 6

combined_model = CombinedModel(
    volume_bias, energy_bias, max_atomic_num, embed_dimension, num_messages
)

out = combined_model(input_tensors)

model = tf.keras.Model(input_tensors, out)

# Train the model
STEPS_PER_EPOCH = math.ceil(len(train) / batch_size)  # number of training examples
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-4, decay_steps=STEPS_PER_EPOCH * 50, decay_rate=1, staircase=False
)

wd_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-5, decay_steps=STEPS_PER_EPOCH * 50, decay_rate=1, staircase=False
)

optimizer = tfa.optimizers.AdamW(
    learning_rate=lr_schedule, weight_decay=wd_schedule, global_clipnorm=1.0
)

model.compile(
    loss=[nfp.losses.masked_mean_absolute_error, "mae"],
    loss_weights=[0.1, 1.0],
    optimizer=optimizer,
)

model_name = "20220603_icsd_and_battery_combined"

if not os.path.exists(model_name):
    os.makedirs(model_name)

# Make a backup of the job submission script
shutil.copy(__file__, model_name)

filepath = model_name + "/best_model"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, save_best_only=True, verbose=0
)
csv_logger = tf.keras.callbacks.CSVLogger(model_name + "/log.csv")

if __name__ == "__main__":
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=100,
        callbacks=[checkpoint, csv_logger],
        verbose=1,
    )

    data["set"] = "train"
    data.loc[data.index.isin(valid.index), "set"] = "valid"
    data.loc[data.index.isin(test.index), "set"] = "test"
    data.loc[data.index.isin(test_composition.index), "set"] = "test_composition"

    def eval_generator(split):
        for _, row in split.iterrows():
            inputs = dict(row.inputs)
            inputs["input_vol"] = row.scaled_input_volperatom
            inputs["true_vol"] = np.nan
            yield inputs

    dataset = (
        tf.data.Dataset.from_generator(
            partial(eval_generator, data),
            output_signature={
                **preprocessor.output_signature,
                "input_vol": tf.TensorSpec(shape=(), dtype=tf.float32),
                "true_vol": tf.TensorSpec(shape=(), dtype=tf.float32),
            },
        )
        .padded_batch(
            batch_size=128,
            padding_values={
                **preprocessor.padding_values,
                "input_vol": tf.constant(np.nan, dtype=tf.float32),
                "true_vol": tf.constant(np.nan, dtype=tf.float32),
            },
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    predictions = model.predict(dataset, verbose=1)

    data["volume_predicted"] = predictions[0]
    data["energy_predicted"] = predictions[1]
    data.drop("inputs", axis=1).to_csv(
        model_name + "/predicted_volumes.csv.gz", compression="gzip"
    )
