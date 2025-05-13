import torch
from torch import nn


class GenericLinkPrediction(nn.Module):
    # Using this model to bundle encoder and decoder
    # performs better than GAE for some reason.
    # Seems to be an issue of parameter initialization.

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, data):
        return self.encoder(data)

    def decode(self, z, edge_label_index):
        return self.decoder(z, edge_label_index)

    def decode_all(self, z):
        edge_label_index = torch.combinations(
            torch.arange(z.size(0)), r=2, with_replacement=True
        ).T
        return self.decoder(z, edge_label_index)
