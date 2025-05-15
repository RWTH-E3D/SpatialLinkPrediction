from typing import List
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T


@functional_transform("concat_features")
class ConcatFeatures(BaseTransform):
    def __init__(self, features: List[str], dim=-1):
        self.features = features
        self.dim = dim

    def __call__(self, data: Data) -> Data:
        if len(self.features) == 1:
            data.x = data[self.features[0]]
        else:
            tensors = []
            for f in self.features:
                tensors.append(data[f])
            data.x = torch.concat(tensors, dim=self.dim)
        data.x = data.x.float()
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.features}, dim={self.dim})"


@functional_transform("remove_features")
class RemoveFeatures(BaseTransform):
    def __init__(self, features: List[str]):
        self.features = features

    def __call__(self, data: Data) -> Data:
        for f in self.features:
            data[f] = None
        return data


@functional_transform("generate_fake_feature")
class GenerateFakeFeature(BaseTransform):
    def __init__(self, channels: int):
        self.channels = channels

    def __call__(self, data: Data) -> Data:
        data.x = torch.rand((data.num_nodes, self.channels))
        return data


@functional_transform("to_one_hot_feature")
class ToOneHotFeature(BaseTransform):
    def __init__(self, feature: str, num_classes: int, save_as: str):
        self.feature = feature
        self.num_classes = num_classes
        self.save_as = save_as

    def __call__(self, data: Data) -> Data:
        data[self.save_as] = torch.nn.functional.one_hot(
            data[self.feature], self.num_classes
        )
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(feature={self.feature}, num_classes={self.num_classes}, save_as={self.save_as})"


@functional_transform("rename_feature")
class RenameFeature(BaseTransform):
    def __init__(self, feature: str, new_name: str):
        self.feature = feature
        self.new_name = new_name

    def __call__(self, data: Data) -> Data:
        data[self.new_name] = data[self.feature]
        data[self.feature] = None
        return data


@functional_transform("normalize_extent")
class NormalizeExtent(BaseTransform):
    def __init__(self):
        self.center = T.Center()

    def __call__(self, data: Data) -> Data:
        data = self.center(data)

        scale = (1 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale
        data.extent = data.extent * scale

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("negative_edge_sampling")
class NegativeEdgeSampling(BaseTransform):
    def __init__(
        self,
        sample_neg_edges: bool = True,
        neg_edge_ratio: float = 1.0,
    ):
        self.sample_neg_edges = sample_neg_edges
        self.neg_edge_ratio = neg_edge_ratio

    def __call__(self, data: Data) -> Data:
        data = data.clone()

        device = data.edge_index.device

        if self.sample_neg_edges:
            pos_edge_index = data.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=int(pos_edge_index.size(1) * self.neg_edge_ratio),
                method="sparse",
            )
            data.edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            data.edge_label = torch.cat(
                [
                    torch.ones(pos_edge_index.size(1), device=device),
                    torch.zeros(neg_edge_index.size(1), device=device),
                ],
                dim=-1,
            )
            data.edge_index = data.edge_label_index
        else:
            data.edge_label_index = data.edge_index
            data.edge_label = torch.ones(data.edge_index.size(1), device=device)

        # perm = torch.randperm(data.edge_label.size(0))
        # data.edge_label_index = data.edge_label_index[:, perm]
        # data.edge_label = data.edge_label[perm]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class OnlyKeepClasses(BaseTransform):
    """
    Filters out data samples that contain none of the desired classes.
    Samples that contain at least one of the desired classes will have
    the labels of points that belong to other classes set to 0.
    """

    def __init__(self, class_indices: List[int]):
        self.class_indices = torch.tensor(class_indices, dtype=torch.long)

    def __call__(self, data: Data) -> bool:
        has_class = torch.any(sum(data.y == i for i in self.class_indices).bool())
        if not has_class:
            return False

        max_class = torch.max(torch.concat([data.y, self.class_indices]))
        remap_tensor = torch.zeros(max_class + 1, dtype=torch.long)
        remap_tensor[self.class_indices] = torch.arange(
            1, self.class_indices.size(0) + 1
        )

        data.y = remap_tensor[data.y]
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.class_indices})"
