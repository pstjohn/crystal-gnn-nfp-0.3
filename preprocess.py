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

structure_dir = Path("/projects/rlmolecule/jlaw/inputs/structures")
battery_relaxed_file = Path(structure_dir, "volrelax/battery_relaxed_vol_structures.json.gz")
#battery_relaxed_file = Path(structure_dir, "battery_relaxed_vol_structures.json.gz")
#inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")
energy_file = Path(structure_dir, "volrelax/battery_volrelaxed_energies.csv")
out_dir = Path("outputs/20220309_volrelax_no_norm")
#out_dir = Path("outputs/20220309_batt_and_volrelax")
os.makedirs(out_dir, exist_ok=True)


preprocessor = PymatgenPreprocessor()


def preprocess_structure(structure, normalize_to_min_dist=False):
    scaled_struct = structure.copy()
    inputs = preprocessor(scaled_struct, train=True)

    # scale structures to a minimum of 1A interatomic distance
    min_distance = inputs["distance"].min()
    if np.isclose(min_distance, 0):
        raise RuntimeError(f"Error with {structure}")

    if normalize_to_min_dist:
        inputs["distance"] /= inputs["distance"].min()
    return inputs


def get_structures(filename):
    with gzip.open(Path(structure_dir, filename), "r") as f:
        for key, structure_dict in tqdm(json.loads(f.read().decode()).items()):
            structure = Structure.from_dict(structure_dict)
            try:
                # for the volume-only relaxation, 
                # try the unnormalized structures
                inputs = preprocess_structure(
                        structure, 
                        normalize_to_min_dist=False)
                yield {"id": key, "inputs": inputs}
            except RuntimeError:
                print(f"Failed to load {key}")
                continue


#icsd_structures = pd.DataFrame(get_structures("icsd_structures.json.gz"))
#icsd_energy = pd.read_csv(Path(structure_dir, "icsd_energies.csv"))

print(f"Reading {battery_relaxed_file}")
battery_structures_relaxed = pd.DataFrame(
    get_structures(battery_relaxed_file)
)
print(f"\t{len(battery_structures_relaxed)} read")

print(f"Reading {energy_file}")
calc_energy = pd.read_csv(Path(energy_file))
print(f"{len(calc_energy)} structures with energy")


#icsd_energy["comp_type"] = icsd_energy.composition.apply(
#    lambda comp: int(
#        "".join(str(x) for x in sorted((int(x) for x in re.findall("(\d+)", comp))))
#    )
#)
#
#data = calc_energy.append(icsd_energy)
#structures = battery_structures_relaxed.append(icsd_structures)
data = calc_energy
structures = battery_structures_relaxed

data = data.merge(structures, on="id", how="inner")
print(f"{len(data)} after merge")
data.to_pickle(Path(out_dir, "all_data.p"))
preprocessor.to_json(Path(out_dir, "preprocessor.json"))
