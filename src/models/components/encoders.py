from torch import nn
from torch_geometric.nn import (
    MLP,
    EdgeConv,
    GCNConv,
    SAGEConv,
)

from src import utils


log = utils.get_pylogger(__name__)


class EdgeConvEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        embedding_channels,
        aggr="max",
    ):
        super().__init__()
        self.conv1 = EdgeConv(MLP([2 * in_channels, 2 * embedding_channels]), aggr)
        self.mlp = MLP(
            [embedding_channels * 2, embedding_channels * 2, embedding_channels],
            dropout=0.5,
        )

    def forward(self, data):
        x0 = data.x
        x1 = self.conv1(x0, data.edge_index)
        out = self.mlp(x1)
        return out


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, embedding_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, embedding_channels)
        self.conv2 = GCNConv(embedding_channels, embedding_channels)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index)
        return x


class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, embedding_channels):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, embedding_channels * 2)
        self.lin1 = MLP(
            [embedding_channels * 2, embedding_channels * 2, embedding_channels],
            dropout=0.5,
        )

    def forward(self, data):
        x = data.x
        x = self.conv1(x, data.edge_index).relu()
        x = self.lin1(x)
        return x


class SAGE2HopEncoder(nn.Module):
    def __init__(self, in_channels, embedding_channels):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, embedding_channels * 2)
        self.conv2 = SAGEConv(2 * embedding_channels, 2 * embedding_channels)
        self.lin1 = MLP(
            [embedding_channels * 2, embedding_channels * 2, embedding_channels],
            dropout=0.5,
        )

    def forward(self, data):
        x = data.x
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = self.lin1(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(self, in_channels, embedding_channels):
        super().__init__()
        self.lin1 = MLP(
            [in_channels, embedding_channels * 2, embedding_channels], dropout=0.5
        )
        self.lin2 = MLP(
            [embedding_channels, embedding_channels * 2, embedding_channels],
            dropout=0.5,
        )
        self.lin3 = MLP(
            [embedding_channels, embedding_channels * 2, embedding_channels],
            dropout=0.5,
        )

    def forward(self, data):
        x = self.lin1(data.x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x)
        return x
