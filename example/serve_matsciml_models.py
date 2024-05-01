import sys

sys.path.append("/store/code/ai4science/matsciml")
import argparse

import numpy as np
import torch
from ase import Atoms
from dgl import DGLGraph
from kusp import KUSPServer
from matsciml.datasets import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    DistancesTransform,
    FrameAveraging,
    GraphVariablesTransform,
    MGLDataTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.datasets.utils import concatenate_keys, element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import FAENet, M3GNet
from matsciml.models.base import ForceRegressionTask, ScalarRegressionTask
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data as PyGGraph


class PyMatGenDataset(MaterialsProjectDataset):
    def data_converter(self, config):
        pymatgen_structure = AseAtomsAdaptor.get_structure(config)
        data = {"structure": pymatgen_structure}
        return_dict = {}
        self._parse_structure(data, return_dict)
        for transform in self.transforms:
            return_dict = transform(return_dict)
        return_dict = self.collate_fn([return_dict])
        return return_dict


def raw_data_to_atoms(species, pos, contributing, cell, elem_map):
    contributing = contributing.astype(int)
    pos_contributing = pos[contributing == 1]
    species = np.array(list(map(lambda x: elem_map[x], species)))
    species = species[contributing == 1]
    atoms = Atoms(species, positions=pos_contributing, cell=cell, pbc=[1, 1, 1])
    return atoms


#########################################################################
#### Server
#########################################################################


class MatSciMLModelServer(KUSPServer):
    def __init__(self, model, dataset, configuration):
        super().__init__(model, configuration)
        self.cutoff = self.global_information.get("cutoff", 6.0)
        self.elem_map = self.global_information.get("elements")
        self.graph_in = None
        self.cell = self.global_information.get(
            "cell",
            np.array(
                [[10.826 * 2, 0.0, 0.0], [0.0, 10.826 * 2, 0.0], [0.0, 0.0, 10.826 * 2]]
            ),
        )
        if not isinstance(self.cell, np.ndarray):
            self.cell = np.array(self.cell)
        self.n_atoms = -1
        self.config = None
        self.dataset = generic_dataset

    def prepare_model_inputs(self, atomic_numbers, positions, contributing_atoms):
        self.n_atoms = atomic_numbers.shape[0]

        config = raw_data_to_atoms(
            atomic_numbers, positions, contributing_atoms, self.cell, self.elem_map
        )
        data = self.dataset.data_converter(config)
        self.batch_in = data
        self.config = config
        return {"batch": self.batch_in}

    def prepare_model_outputs(self, outputs):
        energy = outputs["energy"].double().squeeze().detach().numpy()
        force = outputs["force"].double().squeeze().detach().numpy()
        return {"energy": energy, "forces": force}


if __name__ == "__main__":
    import sys

    sys.path.append("/store/code/ai4science/matsciml")
    from matsciml_configs.models import available_models
    from matsciml_configs.data import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="m3gnet", help="Model to run")
    parser.add_argument(
        "--checkpoint-path",
        "-c",
        default=None,
        help="Model to run",
    )
    
    args = parser.parse_args()
    model = ForceRegressionTask(**available_models[args.model])
    if args.checkpoint_path is None:
        args.checkpoint_path = f"./checkpoints/{args.model}.ckpt"
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=torch.device("cpu")), strict=False
    )

    transforms = transforms[args.model]
    # Need to change cutoff radius due to how lattice is configured in KUSP
    for transform in transforms:
        if isinstance(transform, PeriodicPropertiesTransform):
            transforms[0].cutoff_radius = 1e5

    generic_dataset = PyMatGenDataset("./empty_lmdb", transforms=transforms)

    server = MatSciMLModelServer(
        model=model, dataset=generic_dataset, configuration="kusp_config.yaml"
    )
    server.serve()
