from typing import Any, Dict, Optional, Callable

from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from .components.mepgraph import MEPGraph
from src.utils.transforms import NormalizeExtent


class MEPGraphLinkPredDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/MEPGraph",
        val_fold: int = 0,
        test_fold: int = 1,
        batch_size: int = 1,
        transform: Callable = None,
        transform_once_train: Callable = None,
        transform_once_val: Callable = None,
        node_dropout_prob: float = 0.0,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.pre_transform = T.Compose(
            [
                NormalizeExtent(),
            ]
        )
        self.transform = transform

        self.data_train: Optional[InMemoryDataset] = None
        self.data_val: Optional[InMemoryDataset] = None
        self.data_test: Optional[InMemoryDataset] = None

        self.train_loader: Optional[DataLoader] = None

    def prepare_data(self):
        MEPGraph(
            self.hparams.data_dir,
            transform=self.transform,
            pre_transform=self.pre_transform,
        )

    def setup(self, stage: Optional[str] = None):
        dataset = MEPGraph(
            self.hparams.data_dir,
            transform=self.transform,
            pre_transform=self.pre_transform,
        )

        folds = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21],
            [22, 23, 24, 25, 26],
        ]

        val_indices = folds[self.hparams.val_fold]
        test_indices = folds[self.hparams.test_fold]

        train_indices = []
        for i, f in enumerate(folds):
            if i == self.hparams.val_fold or i == self.hparams.test_fold:
                continue
            train_indices += f

        if self.hparams.transform_once_train:
            t = self.hparams.transform_once_train
            self.data_train = [t(dataset[i]) for i in train_indices]
        else:
            self.data_train = [dataset[i] for i in train_indices]

        if self.hparams.transform_once_val:
            t = self.hparams.transform_once_val
            self.data_val = [t(dataset[i]) for i in val_indices]
            self.data_test = [t(dataset[i]) for i in test_indices]
        else:
            self.data_val = [dataset[i] for i in val_indices]
            self.data_test = [dataset[i] for i in test_indices]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass
