import sys

sys.path.append("/store/code/open-catalyst/public-repo/matsciml")
import dgl
import numpy as np
import torch
from ase import Atoms
from kusp import KUSPServer
from matsciml.datasets import S2EFDataset
from matsciml.datasets.transforms import (MGLDataTransform,
                                          PeriodicPropertiesTransform,
                                          PointCloudToGraphTransform)
from matsciml.datasets.utils import element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import M3GNet
from matsciml.models.base import ScalarRegressionTask
from pymatgen.io.ase import AseAtomsAdaptor

#########################################################################
#### Server
#########################################################################


class M3GNetServer(KUSPServer):
    def __init__(self, model, configuration):
        super().__init__(model, configuration)
        self.cutoff = self.global_information.get("cutoff", 6.0)
        self.n_atoms = -1
        self.config = None

    def prepare_model_inputs(self, atomic_numbers, positions, contributing_atoms):
        import pdb; pdb.set_trace()
        atomic_numbers = self.graph_in['graph'].ndata['atomic_numbers']
        self.n_atoms = atomic_numbers.shape[0]
        self.graph_in['graph'].ndata["node_type"] = torch.tensor(
            list(map(lambda x: x - 1, atomic_numbers))
        ).to(int)
        self.graph_in['graph'].ndata['pos'].requires_grad_(True)
        return {"batch": self.graph_in}

    def prepare_model_outputs(self, energies):
        energy = energies["energy_total"]
        energy.backward()
        forces_contributing = -1 * self.graph_in['graph'].ndata["pos"].grad
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

    server = M3GNetServer(model=model, configuration="kusp_config.yaml")
    server.serve()
