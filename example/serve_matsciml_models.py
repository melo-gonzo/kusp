import sys

sys.path.append("/store/code/open-catalyst/public-repo/matsciml")

import numpy as np
import torch
from ase import Atoms
from dgl import DGLGraph
from kusp import KUSPServer
from matsciml.datasets import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    MGLDataTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.datasets.utils import concatenate_keys, element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import FAENet, M3GNet
from matsciml.models.base import ScalarRegressionTask
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
        if isinstance(self.batch_in["graph"], DGLGraph):
            self.batch_in["graph"].ndata["pos"].requires_grad_(True)
        elif isinstance(self.batch_in["graph"], PyGGraph):
            self.batch_in["graph"].pos.requires_grad_(True)
        else:
            raise TypeError(
                f"This graph typ is not supported {type(self.batch_in['graph'])}."
            )
        return {"batch": self.batch_in}

    def prepare_model_outputs(self, energies):
        energy = energies["energy_total"]
        import pdb

        pdb.set_trace()
        if isinstance(self.batch_in["graph"], DGLGraph):
            pos = self.batch_in["graph"].ndata["pos"]
        elif isinstance(self.batch_in["graph"], PyGGraph):
            pos = self.batch_in["graph"].pos
        forces_contributing = -1 * pos.grad
        forces = np.zeros((self.n_atoms, 3))
        forces[: forces_contributing.shape[0], :] = (
            forces_contributing.double().detach().numpy()
        )
        energy = energy.double().squeeze().detach().numpy()
        return {"energy": energy, "forces": forces}


if __name__ == "__main__":
    model = ScalarRegressionTask(
        encoder_class=M3GNet,
        encoder_kwargs={
            "element_types": element_types(),
            "return_all_layer_output": True,
        },
        output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
        task_keys=["energy_total"],
    )

    model.load_state_dict(
        torch.load("m3gnet_2.pt", map_location=torch.device("cpu")), strict=False
    )

    generic_dataset = PyMatGenDataset(
        "./empty_lmdb",
        transforms=[
            PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "dgl",
                cutoff_dist=20.0,
                node_keys=["pos", "atomic_numbers"],
            ),
            MGLDataTransform(),
        ],
    )

    server = MatSciMLModelServer(
        model=model, dataset=generic_dataset, configuration="kusp_config.yaml"
    )
    server.serve()
