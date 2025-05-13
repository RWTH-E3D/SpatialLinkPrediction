import torch
import torch.utils
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader, NeighborLoader
from typing import Union, List, Optional
from torch_geometric.data.collate import collate
from torch_geometric.loader import NeighborLoader


class InductiveLinkLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.graph_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=shuffle,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )


class MEPSystemSubgraphLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: List[BaseData],
        batch_size: int = 8,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.data, slice_dict, inc_dict = collate(dataset[0].__class__, dataset)
