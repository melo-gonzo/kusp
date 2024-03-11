import sys

sys.path.append("/store/code/open-catalyst/public-repo/matsciml")
import numpy as np
from ase.calculators.kim import KIM

from ase import Atoms, io


from matsciml.datasets import S2EFDataset, MaterialsProjectDataset
from matsciml.datasets.transforms import (
    MGLDataTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

# Initialize KIM Model
model = KIM("KUSP__MO_000000000000_000")

dset = MaterialsProjectDataset.from_devset(
    transforms=[
        PeriodicPropertiesTransform(cutoff_radius=6.5),
        PointCloudToGraphTransform(backend="dgl"),
        MGLDataTransform(),
    ],
)
for idx in range(len(dset)):
    sample = dset.__getitem__(idx)
    config = Atoms(
        sample["graph"].ndata["node_type"], positions=sample["graph"].ndata["pos"]
    )
    # Setting this manually - matsciml PeriodicPropertiesTransform enforces pbc. Not sure if needed
    config.pbc = (True, True, True)
    # Set it as calculator
    config.set_calculator(model)
    # Compute energy/forces
    energy = config.get_potential_energy()
    forces = config.get_forces()

    print(f"Forces: {forces}")
    print(f"Energy: {energy}")
