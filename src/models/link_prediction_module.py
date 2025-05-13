from typing import Any, List

import torch
import json
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAccuracy,
)
from torch_geometric.utils import negative_sampling
from pathlib import Path


class LinkPredictionLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        visual_test_output_path: str = None,
        use_collision_edges=True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_auc = BinaryAUROC()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.train_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()
        self.test_auc = BinaryAUROC()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_auc_best = MaxMetric()

    def step(self, batch):
        z = self.net.encode(batch)
        return z

    def on_train_start(self):
        self.val_auc_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        z = self.step(batch)

        if self.hparams.use_collision_edges:
            # rsat case
            pos_edge_index = batch.true_connections_index
            neg_edge_index = batch.extra_connections_index
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            edge_label = torch.cat(
                [
                    torch.ones(pos_edge_index.size(1), device=pos_edge_index.device),
                    torch.zeros(neg_edge_index.size(1), device=neg_edge_index.device),
                ],
                dim=0,
            )
            batch.edge_index = edge_label_index
        else:
            # negative sampling case
            # Sample new negative edges each time during training
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1),
                method="sparse",
            )

            edge_label_index = torch.cat(
                [batch.edge_label_index, neg_edge_index], dim=-1
            )
            edge_label = torch.cat(
                [batch.edge_label, batch.edge_label.new_zeros(neg_edge_index.size(1))],
                dim=0,
            )

        # Shuffle edges
        perm = torch.randperm(edge_label.size(0), device=edge_label.device)
        edge_label_index = edge_label_index[:, perm]
        edge_label = edge_label[perm]

        batch.z = z
        out = self.net.decode(batch, edge_label_index).view(-1)
        preds = out.sigmoid().detach()

        loss = self.criterion(out, edge_label)

        if self.hparams.use_collision_edges:
            # We need to incorporate the edges that rsat did not
            # find into the metric calculation, pretending that
            # the model predicted them as negative
            preds = torch.cat(
                [
                    preds,
                    torch.zeros(
                        batch.missing_connections_index.size(1),
                        device=batch.missing_connections_index.device,
                    ),
                ],
                dim=0,
            )
            edge_label = torch.cat(
                [
                    edge_label,
                    torch.ones(
                        batch.missing_connections_index.size(1),
                        device=batch.missing_connections_index.device,
                    ),
                ],
                dim=0,
            )

        self.train_loss(loss)
        self.train_auc(preds, edge_label)
        self.train_precision(preds, edge_label)
        self.train_recall(preds, edge_label)
        self.train_f1(preds, edge_label)
        self.train_acc(preds, edge_label)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/precision",
            self.train_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/recall",
            self.train_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds, "targets": edge_label}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        if self.hparams.use_collision_edges:
            edge_label_index = torch.cat(
                [batch.true_connections_index, batch.extra_connections_index], dim=-1
            )
            edge_label = torch.cat(
                [
                    torch.ones(
                        batch.true_connections_index.size(1),
                        device=batch.true_connections_index.device,
                    ),
                    torch.zeros(
                        batch.extra_connections_index.size(1),
                        device=batch.extra_connections_index.device,
                    ),
                ],
                dim=0,
            )
            batch.edge_index = edge_label_index
        else:
            edge_label_index = batch.edge_label_index
            edge_label = batch.edge_label

        z = self.step(batch)
        batch.z = z

        out = self.net.decode(batch, edge_label_index).view(-1)
        preds = out.sigmoid()

        loss = self.criterion(out, edge_label)

        if self.hparams.use_collision_edges:
            preds = torch.cat(
                [
                    preds,
                    torch.zeros(
                        batch.missing_connections_index.size(1),
                        device=batch.missing_connections_index.device,
                    ),
                ],
                dim=0,
            )
            edge_label = torch.cat(
                [
                    edge_label,
                    torch.ones(
                        batch.missing_connections_index.size(1),
                        device=batch.missing_connections_index.device,
                    ),
                ],
                dim=0,
            )

        self.val_loss(loss)
        self.val_auc(preds, edge_label)
        self.val_precision(preds, edge_label)
        self.val_recall(preds, edge_label)
        self.val_f1(preds, edge_label)
        self.val_acc(preds, edge_label)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": edge_label}

    def validation_epoch_end(self, outputs: List[Any]):
        auc = self.val_auc.compute()
        self.val_auc_best(auc)
        self.log("val/auc_best", self.val_auc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        if self.hparams.use_collision_edges:
            edge_label_index = torch.cat(
                [batch.true_connections_index, batch.extra_connections_index], dim=-1
            )
            edge_label = torch.cat(
                [
                    torch.ones(
                        batch.true_connections_index.size(1),
                        device=batch.true_connections_index.device,
                    ),
                    torch.zeros(
                        batch.extra_connections_index.size(1),
                        device=batch.extra_connections_index.device,
                    ),
                ],
                dim=0,
            )
            batch.edge_index = edge_label_index
        else:
            edge_label_index = batch.edge_label_index
            edge_label = batch.edge_label

        z = self.step(batch)
        batch.z = z

        out = self.net.decode(batch, edge_label_index).view(-1)
        preds = out.sigmoid()

        loss = self.criterion(out, edge_label)

        if self.hparams.use_collision_edges:
            preds = torch.cat(
                [
                    preds,
                    torch.zeros(
                        batch.missing_connections_index.size(1),
                        device=batch.missing_connections_index.device,
                    ),
                ],
                dim=0,
            )
            edge_label = torch.cat(
                [
                    edge_label,
                    torch.ones(
                        batch.missing_connections_index.size(1),
                        device=batch.missing_connections_index.device,
                    ),
                ],
                dim=0,
            )

        self.test_loss(loss)
        self.test_auc(preds, edge_label)
        self.test_precision(preds, edge_label)
        self.test_recall(preds, edge_label)
        self.test_f1(preds, edge_label)
        self.test_acc(preds, edge_label)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.visual_test_output_path:
            batch.edge_label = edge_label
            batch.edge_label_index = edge_label_index
            self.save_test_visuals(batch, preds)

        return {"loss": loss, "preds": preds, "targets": edge_label}

    def save_test_visuals(self, batch, probs):
        graph = {"nodes": [], "edges": []}

        for i in range(batch.num_nodes):
            pos = [round(x.item(), 4) for x in batch.pos[i]]
            rotation = (
                np.array([round(x.item(), 4) for x in batch.rotation[i]])
                .reshape(3, 3)
                .tolist()
            )
            extent = [round(x.item(), 4) for x in batch.extent[i]]
            class_id = batch.ifc_class[i].item()
            node = {
                "id": i,
                "global_id": None,
                "name": None,
                "ifc_class": class_id,
                "obb": {"center": pos, "extent": extent, "rotation": rotation},
            }
            graph["nodes"].append(node)

        for i in range(batch.edge_label_index.size(1)):
            e = batch.edge_label_index[:, i]
            label = int(batch.edge_label[i].item())
            pred = (probs[i] > 0.5).int().item()
            if label == 0 and pred == 0:
                # Do not export true negatives, as there are too many of them to properly visualize
                continue
            edge = {
                "from": e[0].item(),
                "to": e[1].item(),
                "label": label,
                "probability": round(probs[i].item(), 4),
                "flow_direction": None,
            }
            graph["edges"].append(edge)

        path = Path(self.hparams.visual_test_output_path)
        path.mkdir(exist_ok=True)
        output_count = len(list(path.glob("*.predictions.json")))
        with (path / f"graph_{output_count:02d}.predictions.json").open("w") as f:
            json.dump(graph, f)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        layer = ["point_cloud_encoder"]
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(l in n for l in layer)
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(l in n for l in layer)
                ],
                "lr": 1e-3,
            },
        ]
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
