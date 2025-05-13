import torch
from torch import nn
from torch_geometric.nn import MLP


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, edge_label_index):
        head = batch.z[edge_label_index[0]]
        tail = batch.z[edge_label_index[1]]

        return (head * tail).sum(dim=-1)


class DistMultDecoder(nn.Module):
    def __init__(self, embedding_channels):
        super().__init__()
        self.rel_emb = nn.Embedding(1, embedding_channels)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, batch, edge_label_index):
        head = batch.z[edge_label_index[0]]
        tail = batch.z[edge_label_index[1]]
        rel = self.rel_emb.weight[0]

        return (head * rel * tail).sum(dim=-1)


class SpatialDistMultDecoder(nn.Module):
    def __init__(self, embedding_channels, output_channels=1):
        super().__init__()
        self.rel_emb = nn.Embedding(1, embedding_channels)
        self.lin = MLP([embedding_channels + 1, embedding_channels, output_channels])
        self.link_lin = MLP([4, 16, 32, 16, 1])
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        for m in self.link_lin.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data, edge_label_index):
        head = data.z[edge_label_index[0]]
        tail = data.z[edge_label_index[1]]
        rel = self.rel_emb.weight[0]

        start, end = data.pos[edge_label_index[0]], data.pos[edge_label_index[1]]
        min_xyz = torch.min(data.pos, dim=0)[0]
        max_xyz = torch.max(data.pos, dim=0)[0]

        directions = end - start
        directions_sq = directions**2
        distances = torch.sqrt(torch.sum(directions_sq, dim=1)).view(-1, 1)
        directions_normed = directions / (distances + 1e-8)

        max_distance = torch.sqrt(torch.sum((max_xyz - min_xyz) ** 2))
        distances /= max_distance

        link_feat = torch.cat([distances, directions_normed], dim=-1)
        link_emb = self.link_lin(link_feat)

        x = torch.cat([(head * rel * tail), link_emb], dim=-1)
        x = self.lin(x)

        # return (head * rel * tail).sum(dim=-1)
        return x


class MEPLinkDecoder(nn.Module):
    def __init__(self, embedding_channels, output_channels=1):
        super().__init__()
        self.lin = MLP(
            [2 * embedding_channels + 1, embedding_channels, output_channels]
        )
        self.link_lin = MLP([4, 16, 32, 16, 1])

    def forward(self, data, edge_label_index):
        start, end = data.pos[edge_label_index[0]], data.pos[edge_label_index[1]]
        min_xyz = torch.min(data.pos, dim=0)[0]
        max_xyz = torch.max(data.pos, dim=0)[0]

        directions = end - start
        directions_sq = directions**2
        distances = torch.sqrt(torch.sum(directions_sq, dim=1)).view(-1, 1)
        directions_normed = directions / (distances + 1e-8)

        max_distance = torch.sqrt(torch.sum((max_xyz - min_xyz) ** 2))
        distances /= max_distance

        link_feat = torch.cat([distances, directions_normed], dim=-1)
        # link_feat = directions_normed
        # link_feat = distances
        # link_feat = torch.randn_like(distances)
        link_emb = self.link_lin(link_feat)
        # fake_link_emb = torch.zeros_like(link_emb)

        z = data.z
        x = torch.cat(
            [z[edge_label_index[0]], z[edge_label_index[1]], link_emb], dim=-1
        )
        # TODO: Try sum, mean, max pooling
        x = self.lin(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, embedding_channels, output_channels=1):
        super().__init__()
        self.lin = MLP([2 * embedding_channels, embedding_channels, output_channels])

    def forward(self, data, edge_label_index):
        z = data.z
        x = torch.cat([z[edge_label_index[0]], z[edge_label_index[1]]], dim=-1)
        # TODO: Try sum, mean, max pooling
        x = self.lin(x)
        return x


class NoOpDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
