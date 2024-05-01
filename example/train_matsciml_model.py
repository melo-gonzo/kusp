from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.utils import element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import M3GNet
from matsciml.models.base import ForceRegressionTask

from matsciml.datasets.transforms import MGLDataTransform
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

# construct a scalar regression task with SchNet encoder
task = ForceRegressionTask(
    encoder_class=M3GNet,
    encoder_kwargs={"element_types": element_types(), "return_all_layer_output": True},
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    task_keys=["force"],
)
dm = MatSciMLDataModule.from_devset(
    "S2EFDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
            PointCloudToGraphTransform(backend="dgl"),
            MGLDataTransform(),
        ]
    },
    num_workers=0,
    batch_size=4,
)
# run a quick training loop
trainer = pl.Trainer(max_epochs=1, accelerator="cpu")
trainer.fit(task, datamodule=dm)

