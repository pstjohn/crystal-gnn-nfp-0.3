from distutils.log import warn
from pathlib import Path

import numpy as np
import pandas as pd
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen.core.periodic_table import Element
from tqdm.auto import tqdm

tqdm.pandas()

structure_dir = Path("/projects/rlmolecule/jlaw/inputs/structures")
inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")
volrelax_dir = Path("/projects/rlmolecule/pstjohn/volume_relaxation_outputs/")


class AtomicNumberPreprocessor(PymatgenPreprocessor):
    def __init__(self, max_atomic_num=83, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_tokenizer = lambda x: Element(x).Z
        self._max_atomic_num = max_atomic_num

    @property
    def site_classes(self):
        return self._max_atomic_num


def preprocess_structure(row):
    inputs = preprocessor(row.structure, train=True)

    # scale structures to a minimum of 1A interatomic distance
    min_distance = inputs["distance"].min()
    if np.isclose(min_distance, 0):
        warn(f"Error with {row.id}")
        return None

    scale_factor = 1.0 / inputs["distance"].min()
    inputs["distance"] *= scale_factor

    radii = np.array([site.specie.atomic_radius for site in row.structure.sites])
    r0 = radii[inputs["connectivity"][:, 0]]
    r1 = radii[inputs["connectivity"][:, 1]]
    d = inputs["distance"]
    radii_scale_factor = (d / (r0 + r1)).min()

    return pd.Series(
        {
            "inputs": inputs,
            "scale_factor": scale_factor,
            "radii_scale_factor": radii_scale_factor,
        }
    )


preprocessor = AtomicNumberPreprocessor()

if __name__ == "__main__":

    data = pd.read_pickle(Path(inputs_dir, "20220510_all_structures.p")).reset_index(
        drop=True
    )
    preprocessed = data.progress_apply(preprocess_structure, axis=1).dropna()
    preprocessed = data.drop(["structure"], axis=1).join(preprocessed, how="right")

    preprocessed.to_pickle(Path(inputs_dir, "20220511_scaled_inputs.p"))
