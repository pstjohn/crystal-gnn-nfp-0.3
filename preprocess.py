import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen import Structure
from tqdm.auto import tqdm

tqdm.pandas()

structure_dir = Path("/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures")
inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")


preprocessor = PymatgenPreprocessor()


def preprocess_structure(structure):
    scaled_struct = structure.copy()
    inputs = preprocessor(scaled_struct, train=True)

    # scale structures to a minimum of 1A interatomic distance
    min_distance = inputs["distance"].min()
    if np.isclose(min_distance, 0):
        raise RuntimeError(f"Error with {structure}")

    inputs["distance"] /= inputs["distance"].min()
    return inputs


def get_structures(filename):
    with gzip.open(Path(structure_dir, filename), "r") as f:
        for key, structure_dict in tqdm(json.loads(f.read().decode()).items()):
            structure = Structure.from_dict(structure_dict)
            inputs = preprocess_structure(structure)
            yield {"id": key, "inputs": inputs}


icsd_structures = pd.DataFrame(get_structures("icsd_structures.json.gz"))

battery_structures_relaxed = pd.DataFrame(
    get_structures("battery_relaxed_structures.json.gz")
)

calc_energy = pd.read_csv(
    Path(inputs_dir, "20211223_deduped_energies_matminer_0.01.csv")
)

icsd_energy = pd.read_csv(Path(structure_dir, "icsd_energies.csv"))

icsd_energy["comp_type"] = icsd_energy.composition.apply(
    lambda comp: int(
        "".join(str(x) for x in sorted((int(x) for x in re.findall("(\d+)", comp))))
    )
)

data = calc_energy.append(icsd_energy)
structures = battery_structures_relaxed.append(icsd_structures)

data = data.merge(structures, on="id", how="inner")
data.to_pickle(Path(inputs_dir, "20211227_all_data.p"))
preprocessor.to_json(Path(inputs_dir, "20211227_preprocessor.json"))
