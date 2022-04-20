import math
import os
import shutil
from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from nfp import EdgeUpdate, NodeUpdate
from nfp.layers import RBFExpansion
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tqdm.auto import tqdm

tqdm.pandas()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data-file', type=Path, help='The pickled file with processed input structures. preprocessor.json should be in the same folder')
parser.add_argument('--out-dir', type=Path, help='Where to place the output files')
args = parser.parse_args()

#inputs_dir = Path("outputs/20220314_volunrelax_dls4")
inputs_dir = args.data_file.parents[0]
# can use run_id if testing different hyperparameters
#model_dir = Path(inputs_dir, run_id)
model_dir = args.out_dir


#data = pd.read_pickle(Path(inputs_dir, "volunrelax_data.p"))
data = pd.read_pickle(args.data_file)
# limit to structures with energy in the range -10 to 10
#num_strcs = len(data)
#data = data[(data['energyperatom'] > -10) & (data['energyperatom'] < 5)]
#print(f"{num_strcs} structures reduced to {len(data)} after filtering to the energy range -10 to 5")
preprocessor = PymatgenPreprocessor()
#preprocessor.from_json(Path(inputs_dir, "preprocessor.json"))
# TODO preprocessor should match the data files
# preprocessor should be in the same dir as the data files
preprocessor.from_json(Path(inputs_dir, "preprocessor.json"))


def random_split_df(df, test_size=.05, random_state=None):
    strc_ids = df.id.unique()
    train, valid, test = random_split(
        strc_ids, test_size=test_size, random_state=random_state)
    train_df = df[df.id.isin(train)]
    valid_df = df[df.id.isin(valid)]
    test_df = df[df.id.isin(test)]

    return train_df, valid_df, test_df


def random_split(structure_ids, test_size=.05, random_state=None):
    if test_size < 1:
        test_size_perc = test_size
        test_size = int(np.floor(len(structure_ids) * test_size))
    else:
        test_size_perc = test_size / float(len(structure_ids))
    print(f"\tsplitting {len(structure_ids)} structures using test_size: {test_size_perc} ({test_size})")
    valid_test_size = test_size*2
    train, valid  = train_test_split(structure_ids, test_size=valid_test_size, random_state=random_state)
    random_state2 = random_state + 1 if random_state else None
    valid, test = train_test_split(valid, test_size=.5, random_state=random_state2)
    return train, valid, test


#train, valid = train_test_split(data, test_size=.1, random_state=1)
#valid, test = train_test_split(valid, test_size=0.5, random_state=2)
# rather than just split randomly, try leaving out the same decorated structures in the relaxed and unrelaxed sets
relaxed_ids = data[data['data_type'] == 'battery_relaxed']['id']
train_ids, valid_ids, test_ids = random_split(relaxed_ids,
                                              test_size=.05,
                                              random_state=1)
batt_train = data[data.id.isin(train_ids)]
batt_valid = data[data.id.isin(valid_ids)]
batt_test = data[data.id.isin(test_ids)]
print(f"{len(train_ids)}, {len(valid_ids)} train and valid ids after split")
print(f"{len(batt_train)}, {len(batt_valid)} train and valid structures")

# also split the ICSD structures
icsd_train, icsd_valid, icsd_test = random_split_df(data[data['data_type'] == 'icsd'],
                                                    test_size=0.05,
                                                    random_state=1)
print(f"{len(icsd_train)}, {len(icsd_valid)} train and valid icsd ids after split")

train = pd.concat([batt_train, icsd_train])
valid = pd.concat([batt_valid, icsd_valid])
test = pd.concat([batt_test, icsd_test])
print(f"{len(train)}, {len(valid)} train and valid structures overall")

# write the ids to a file
train[['id', 'data_type']].to_csv(Path(model_dir, 'train_ids.txt'), index=False)
valid[['id', 'data_type']].to_csv(Path(model_dir, 'valid_ids.txt'), index=False)
test[['id', 'data_type']].to_csv(Path(model_dir, 'test_ids.txt'), index=False)


def calculate_output_bias(train):
    """ We can get a reasonable guess for the output bias by just assuming the crystal's
     energy is a linear sum over it's element types """
    # This just converts to a count of each element by crystal
    site_counts = train.inputs.progress_apply(
        lambda x: pd.Series(Counter(x["site"]))
    ).fillna(0)
    # Linear regression assumes a sum, while we average over sites in the neural network.
    # Here, we make the regression target the total energy, not the site-averaged energy
    num_sites = site_counts.sum(1)
    total_energies = train["energyperatom"] * num_sites

    # Do the least-squares regression, and stack on zeros for the mask and unknown tokens
    output_bias = np.linalg.lstsq(site_counts, total_energies, rcond=None)[0]
    output_bias = np.hstack([np.zeros(2), output_bias])
    return output_bias


def build_dataset(split, batch_size):
    return (
        tf.data.Dataset.from_generator(
            lambda: ((row.inputs, row.energyperatom) for _, row in split.iterrows()),
            output_signature=(
                preprocessor.output_signature,
                tf.TensorSpec((), dtype=tf.float32),
            ),
        )
        .cache()
        .shuffle(buffer_size=len(split))
        .padded_batch(
            batch_size=batch_size,
            padding_values=(
                preprocessor.padding_values,
                tf.constant(np.nan, dtype=tf.float32),
            ),
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


# Calculate an initial guess for the output bias
output_bias = calculate_output_bias(train)

batch_size = 64
train_dataset = build_dataset(train, batch_size=batch_size)
valid_dataset = build_dataset(valid, batch_size=batch_size)


# Keras model
site_class = layers.Input(shape=[None], dtype=tf.int64, name="site")
distances = layers.Input(shape=[None], dtype=tf.float32, name="distance")
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
input_tensors = [site_class, distances, connectivity]

embed_dimension = 256
num_messages = 6

atom_state = layers.Embedding(
    preprocessor.site_classes, embed_dimension, name="site_embedding", mask_zero=True
)(site_class)

atom_mean = layers.Embedding(
    preprocessor.site_classes,
    1,
    name="site_mean",
    mask_zero=True,
    embeddings_initializer=tf.keras.initializers.Constant(output_bias),
)(site_class)

rbf_distance = RBFExpansion(
    dimension=128, init_max_distance=7, init_gap=30, trainable=True
)(distances)

bond_state = layers.Dense(embed_dimension)(rbf_distance)

for _ in range(num_messages):
    new_bond_state = EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([bond_state, new_bond_state])
    new_atom_state = NodeUpdate()([atom_state, bond_state, connectivity])
    atom_state = layers.Add()([atom_state, new_atom_state])

# Reduce the atom state vector to a single energy prediction
atom_state = layers.Dense(
    1,
    name="site_energy_offset",
    kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=1e-6, seed=None
    ),
)(atom_state)

# Add this 'offset' prediction to the learned mean energy for the given element type
atom_state = layers.Add(name="add_energy_offset")([atom_state, atom_mean])

# Calculate a final mean energy per atom
out = tf.keras.layers.GlobalAveragePooling1D()(atom_state)

model = tf.keras.Model(input_tensors, [out])


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

model.compile(loss="mae", optimizer=optimizer)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Make a backup of the job submission script
dest_file = Path(model_dir, os.path.basename(__file__))
if not dest_file.is_file() or Path(__file__) != dest_file:
    shutil.copy(__file__, dest_file)

filepath = Path(model_dir, "best_model.hdf5")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, save_best_only=True, verbose=0
)
csv_logger = tf.keras.callbacks.CSVLogger(Path(model_dir, "log.csv"))

if __name__ == "__main__":
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=100,
        callbacks=[checkpoint, csv_logger],
        verbose=1,
    )
