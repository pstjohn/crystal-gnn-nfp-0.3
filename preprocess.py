import gzip
import json
import re
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen.core import Structure
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from tqdm.auto import tqdm

tqdm.pandas()
preprocessor = PymatgenPreprocessor()


def preprocess_structure(structure,
                         normalize_to_min_dist=False,
                         volume_prediction=None):
    scaled_struct = structure.copy()
    if volume_prediction is not None:
        scaled_struct.scale_lattice(volume_prediction)
    inputs = preprocessor(scaled_struct, train=True)

    # scale structures to a minimum of 1A interatomic distance
    min_distance = inputs["distance"].min()
    # skip structures that have a minimum distance = 0
    if np.isclose(min_distance, 0):
        #raise RuntimeError(f"Error with {structure}")
        return

    if normalize_to_min_dist:
        inputs["distance"] /= inputs["distance"].min()
    return inputs


def get_structures(filename, normalize_to_min_dist=False, volume_predictions=None):
    """ Load and preprocess structures from a pymatgen json.gz file
    :param volume_predictions: 
    """
    print(f"Reading {filename}")
    with gzip.open(Path(structure_dir, filename), "r") as f:
        for key, structure_dict in tqdm(json.loads(f.read().decode()).items()):
            structure = Structure.from_dict(structure_dict)
            try:
                # try starting from a predicted volume
                # for the unrelaxed structures
                vol_pred = None
                if volume_predictions is not None:
                    if key in volume_predictions.index:
                        vol_pred = volume_predictions.loc[key]
                    #else:
                    #    print(f"No volume prediction available for {key}")
                inputs = preprocess_structure(
                        structure, 
                        normalize_to_min_dist=normalize_to_min_dist,
                        volume_prediction=vol_pred,
                        )
                yield {"id": key, "inputs": inputs}
            except RuntimeError:
                print(f"Failed to load {key}")
                continue


def pred_vol(structure):
    # first predict the volume using the average volume per element (from ICSD)
    site_counts = pd.Series(Counter(
        str(site.specie) for site in structure.sites)).fillna(0)
    curr_site_bias = site_bias[site_bias.index.isin(site_counts.index)]
    linear_pred = site_counts @ curr_site_bias
    structure.scale_lattice(linear_pred)

    # then apply Pymatgen's DLS predictor
    pred_volume = dls_vol_predictor.predict(structure)
    return pred_volume


def scale_by_pred_vol(structure, site_bias, dls_vol_predictor):
    # first predict the volume using the average volume per element (from ICSD)
    site_counts = pd.Series(Counter(
        str(site.specie) for site in structure.sites)).fillna(0)
    curr_site_bias = site_bias[site_bias.index.isin(site_counts.index)]
    linear_pred = site_counts @ curr_site_bias
    structure.scale_lattice(linear_pred)

    # then apply Pymatgen's DLS predictor
    pred_volume = dls_vol_predictor.predict(structure)
    structure.scale_lattice(pred_volume)
    return structure


structure_dir = Path("/projects/rlmolecule/jlaw/inputs/structures")
inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")

battery_relaxed_file = Path(structure_dir, "battery/battery_relaxed_structures.json.gz")
battery_unrelaxed_file = Path(structure_dir, "battery/all_unrelaxed_structures.p")
icsd_strcs_file = Path(structure_dir, "icsd/icsd_structures.json.gz")

batt_energy_file = Path(inputs_dir, "20211223_deduped_energies_matminer_0.01.csv")
vol_energy_file = Path(structure_dir, "battery/volrelax/20220422_battery_volrelaxed_energies.csv")
icsd_energy_file = Path(structure_dir, "icsd/icsd_energies.csv")

# either normalize/scale all structures to a minimum interatomic distance=1
# or leave the relaxed and ICSD structures at their real volume, 
# and predict the volume of the unrelaxed structures (using linear + DLS approach)
# to use for training with the volume-only relaxations
normalize_to_min_dist = False
out_dir = Path("outputs/20220422_batt_icsd_and_volunrelax")
os.makedirs(out_dir, exist_ok=True)


print(f"reading unrelaxed structures from {battery_unrelaxed_file}")
unrelaxed_structures = pd.read_pickle(battery_unrelaxed_file)
print(f"\t{len(unrelaxed_structures)} read")

print(f"Reading {vol_energy_file}")
vol_energy = pd.read_csv(vol_energy_file)
print(f"\t{len(vol_energy)} structures with energy")
vol_data = vol_energy.merge(unrelaxed_structures, on="id", how="inner")
print(f"\t{len(vol_data)} after merge")
del unrelaxed_structures

if normalize_to_min_dist:
    print("processing unrelaxed structures")
    kwargs = {"normalize_to_min_dist": True}
    vol_data['inputs'] = vol_data.structure.progress_apply(
            preprocess_structure, **kwargs)
else:
    # Make the volume predictions here
    site_bias_file = "/projects/rlmolecule/pstjohn/crystal_inputs/site_volumes_from_icsd.csv"
    print(f"Reading {site_bias_file}")
    site_bias = pd.read_csv(site_bias_file,
                            index_col=0, squeeze=True)
    print(f"\t{len(site_bias)} elements")
    dls_vol_predictor = DLSVolumePredictor()

    print("making volume predictions")
    vol_data['vol_pred'] = vol_data.structure.progress_apply(pred_vol)
    print("processing unrelaxed structures")
    vol_data['inputs'] = vol_data.progress_apply(
        lambda row: preprocess_structure(row.structure,
                                         volume_prediction=row.vol_pred),
        axis=1,
        )
    # drop the structure column to save space
    vol_data.drop('structure', axis=1, inplace=True)

# only normalize if the unrelaxed structures are also normalized
battery_structures_relaxed = pd.DataFrame(get_structures(battery_relaxed_file,
                                                         normalize_to_min_dist=normalize_to_min_dist))
print(f"\t{len(battery_structures_relaxed)} read")
print(f"Reading {batt_energy_file}")
batt_energy = pd.read_csv(Path(batt_energy_file))
print(f"{len(batt_energy)} structures with energy")
batt_data = batt_energy.merge(battery_structures_relaxed, on="id", how="inner")
print(f"\t{len(batt_data)} after merge")

icsd_structures = pd.DataFrame(get_structures(icsd_strcs_file,
                                              normalize_to_min_dist=normalize_to_min_dist))
print(f"\t{len(icsd_structures)} read")
print(f"Reading {icsd_energy_file}")
icsd_energy = pd.read_csv(icsd_energy_file)
print(f"\t{len(icsd_energy)} structures with energy")
icsd_data = icsd_energy.merge(icsd_structures, on="id", how="inner")
print(f"\t{len(icsd_data)} after merge")


icsd_energy["comp_type"] = icsd_energy.composition.apply(
            lambda comp: int(
                        "".join(str(x) for x in sorted((int(x) for x in re.findall("(\d+)", comp))))
                            )
            )

batt_data['data_type'] = 'battery_relaxed'
vol_data['data_type'] = 'battery_volume_unrelaxed'
icsd_data['data_type'] = 'icsd'
all_data = pd.concat([batt_data, vol_data, icsd_data])
print(f"{len(all_data)} structures combined")

out_file = Path(out_dir, "all_data.p")
print(f"Writing {out_file}")
print(all_data.head(2))
all_data.to_pickle(out_file)

# the preprocessor is updated as the files are read,
# so need to load all structures before the preprocessor is written
preprocessor.to_json(Path(out_dir, "preprocessor.json"))
