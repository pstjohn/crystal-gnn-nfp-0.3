import gzip
import json
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen.core import Structure
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
    if np.isclose(min_distance, 0):
        raise RuntimeError(f"Error with {structure}")

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


structure_dir = Path("/projects/rlmolecule/jlaw/inputs/structures")
inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")

battery_relaxed_file = Path(structure_dir, "battery_relaxed_structures.json.gz")
battery_volunrelaxed_file = Path(structure_dir, "volrelax/battery_unrelaxed_vol_structures.json.gz")
icsd_strcs_file = Path(structure_dir, "icsd_structures.json.gz")

batt_energy_file = Path(inputs_dir, "20211223_deduped_energies_matminer_0.01.csv")
vol_energy_file = Path(structure_dir, "volrelax/battery_volrelaxed_energies.csv")
icsd_energy_file = Path(structure_dir, "icsd_energies.csv")

out_dir = Path("outputs/20220314_batt_icsd_and_volunrelax")
os.makedirs(out_dir, exist_ok=True)


# Read in the volume predictions and apply them to the structures
# The volume predictions don't actually help to correctly predict the eV/atom
#vol_pred_file = Path(structure_dir, "volrelax/linear_dls1.5_predicted_volumes.csv")
##vol_pred_file = Path(structure_dir, "volrelax/dls1.5_predicted_volumes.csv")
#print(f"Reading volume predictions from {vol_pred_file}")
#volume_predictions = pd.read_csv(vol_pred_file, index_col=0, squeeze=True)
#print(f"\t{len(volume_predictions)} read")
#volume_predictions = volume_predictions.dropna()
volunrelaxed_structures = pd.DataFrame(get_structures(
    battery_volunrelaxed_file, 
    normalize_to_min_dist=False,
#    volume_predictions=volume_predictions,
    ))
print(f"\t{len(volunrelaxed_structures)} read")

print(f"Reading {vol_energy_file}")
vol_energy = pd.read_csv(vol_energy_file)
print(f"{len(vol_energy)} structures with energy")
vol_data = vol_energy.merge(volunrelaxed_structures, on="id", how="inner")
print(f"{len(vol_data)} after merge")

battery_structures_relaxed = pd.DataFrame(get_structures(battery_relaxed_file))
print(f"\t{len(battery_structures_relaxed)} read")
print(f"Reading {batt_energy_file}")
batt_energy = pd.read_csv(Path(batt_energy_file))
print(f"{len(batt_energy)} structures with energy")
batt_data = batt_energy.merge(battery_structures_relaxed, on="id", how="inner")
print(f"{len(batt_data)} after merge")

icsd_structures = pd.DataFrame(get_structures(icsd_strcs_file))
print(f"\t{len(icsd_structures)} read")
print(f"Reading {icsd_energy_file}")
icsd_energy = pd.read_csv(icsd_energy_file)
print(f"\t{len(icsd_energy)} structures with energy")
icsd_data = icsd_energy.merge(icsd_structures, on="id", how="inner")
print(f"{len(icsd_data)} after merge")


icsd_energy["comp_type"] = icsd_energy.composition.apply(
            lambda comp: int(
                        "".join(str(x) for x in sorted((int(x) for x in re.findall("(\d+)", comp))))
                            )
            )

#batt_data.to_pickle(Path(out_dir, "batt_data.p"))
#icsd_data.to_pickle(Path(out_dir, "icsd_data.p"))
#vol_data.to_pickle(Path(out_dir, "volunrelax_data.p"))
batt_data['data_type'] = 'battery_relaxed'
vol_data['data_type'] = 'battery_volume_unrelaxed'
icsd_data['data_type'] = 'icsd'
all_data = pd.concat([batt_data, vol_data, icsd_data])
out_file = Path(out_dir, "all_data.p")
print(f"Writing {out_file}")
all_data.to_pickle(out_file)
preprocessor.to_json(Path(out_dir, "preprocessor.json"))
