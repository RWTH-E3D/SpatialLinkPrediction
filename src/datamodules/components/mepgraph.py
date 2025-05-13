import torch
import json
import numpy as np
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.collate import collate
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T


class MEPGraph(InMemoryDataset):
    class_names = [
        "IfcAirTerminal",
        "IfcAirTerminalBox",
        "IfcAirToAirHeatRecovery",
        "IfcAlarm",
        "IfcBoiler",
        "IfcBuildingElementProxy",
        "IfcCableCarrierFitting",
        "IfcCableCarrierSegment",
        "IfcChiller",
        "IfcCoil",
        "IfcController",
        "IfcCoolingTower",
        "IfcDamper",
        "IfcDistributionChamberElement",
        "IfcDistributionControlElement",
        "IfcDuctFitting",
        "IfcDuctSegment",
        "IfcDuctSilencer",
        "IfcElectricAppliance",
        "IfcElectricMotor",
        "IfcFan",
        "IfcFilter",
        "IfcFireSuppressionTerminal",
        "IfcFlowController",
        "IfcFlowFitting",
        "IfcFlowMeter",
        "IfcFlowSegment",
        "IfcFlowTerminal",
        "IfcHeatExchanger",
        "IfcHumidifier",
        "IfcInterceptor",
        "IfcJunctionBox",
        "IfcLightFixture",
        "IfcOutlet",
        "IfcPipeFitting",
        "IfcPipeSegment",
        "IfcPump",
        "IfcSanitaryTerminal",
        "IfcSensor",
        "IfcSpaceHeater",
        "IfcSwitchingDevice",
        "IfcTank",
        "IfcTransformer",
        "IfcUnitaryEquipment",
        "IfcValve",
        "IfcWasteTerminal",
    ]

    class_mapping = {
        "IfcAirTerminal": "IfcFlowTerminal",
        "IfcAirTerminalBox": "IfcFlowController",
        "IfcAirToAirHeatRecovery": "IfcEnergyConversionDevice",
        "IfcAlarm": "IfcDistributionControlElement",
        "IfcBoiler": "IfcEnergyConversionDevice",
        "IfcBuildingElementProxy": "IfcBuildingElementProxy",
        "IfcCableCarrierFitting": "IfcFlowFitting",
        "IfcCableCarrierSegment": "IfcFlowSegment",
        "IfcCableSegment": "IfcFlowSegment",
        "IfcChiller": "IfcEnergyConversionDevice",
        "IfcCoil": "IfcEnergyConversionDevice",
        "IfcController": "IfcDistributionControlElement",
        "IfcCoolingTower": "IfcEnergyConversionDevice",
        "IfcDamper": "IfcFlowController",
        "IfcDistributionChamberElement": "IfcDistributionFlowElement",
        "IfcDistributionControlElement": "IfcDistributionControlElement",
        "IfcDuctFitting": "IfcFlowFitting",
        "IfcDuctSegment": "IfcFlowSegment",
        "IfcDuctSilencer": "IfcFlowTreatmentDevice",
        "IfcElectricAppliance": "IfcFlowTerminal",
        "IfcElectricDistributionBoard": "IfcFlowController",
        "IfcElectricMotor": "IfcEnergyConversionDevice",
        "IfcFan": "IfcFlowMovingDevice",
        "IfcFilter": "IfcFlowTreatmentDevice",
        "IfcFireSuppressionTerminal": "IfcFlowTerminal",
        "IfcFlowController": "IfcFlowController",
        "IfcFlowFitting": "IfcFlowFitting",
        "IfcFlowMeter": "IfcFlowController",
        "IfcFlowSegment": "IfcFlowSegment",
        "IfcFlowTerminal": "IfcFlowTerminal",
        "IfcHeatExchanger": "IfcEnergyConversionDevice",
        "IfcHumidifier": "IfcEnergyConversionDevice",
        "IfcInterceptor": "IfcFlowTreatmentDevice",
        "IfcJunctionBox": "IfcFlowFitting",
        "IfcLightFixture": "IfcFlowTerminal",
        "IfcOutlet": "IfcFlowTerminal",
        "IfcPipeFitting": "IfcFlowFitting",
        "IfcPipeSegment": "IfcFlowSegment",
        "IfcPump": "IfcFlowMovingDevice",
        "IfcSanitaryTerminal": "IfcFlowTerminal",
        "IfcSensor": "IfcDistributionControlElement",
        "IfcSpaceHeater": "IfcFlowTerminal",
        "IfcSwitchingDevice": "IfcFlowController",
        "IfcTank": "IfcFlowStorageDevice",
        "IfcTransformer": "IfcEnergyConversionDevice",
        "IfcUnitaryEquipment": "IfcEnergyConversionDevice",
        "IfcValve": "IfcFlowController",
        "IfcWasteTerminal": "IfcFlowTerminal",
    }

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        map_classes_to_parent=True,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        data, slices = torch.load(self.processed_paths[0])

        if map_classes_to_parent:
            parent_classes = np.unique(list(self.class_mapping.values())).tolist()
            parent_class_indices = []
            for child_class_index in data.ifc_class:
                child_class_name = self.class_names[child_class_index]
                parent_class_name = self.class_mapping[child_class_name]
                parent_class_index = parent_classes.index(parent_class_name)
                parent_class_indices.append(parent_class_index)

            data.ifc_class = torch.tensor(parent_class_indices, dtype=int)

        self.data, self.slices = data, slices

    @property
    def raw_file_names(self):
        return [f"model_{str(i).zfill(2)}" for i in range(27)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process_directory(self, path):
        path = Path(path)
        print(f"Processing {path.stem}")

        graph_file = path / "graph.json"
        with open(graph_file, "r") as f:
            content = json.load(f)

        data = Data()
        nodes = content["nodes"]
        edges = content["edges"]
        edge_index = []
        flow_direction = []
        node_dict = {n["id"]: i for i, n in enumerate(nodes)}
        available_directions = ["sink", "source", "source_and_sink"]
        for e in edges:
            source_id = node_dict[e["from"]]
            target_id = node_dict[e["to"]]
            fd_index = available_directions.index(e["flow_direction"])
            flow_direction.append(fd_index)
            edge_index.append([source_id, target_id])
        data.edge_index = torch.tensor(edge_index, dtype=int).T
        data.flow_direction = torch.tensor(flow_direction, dtype=int)
        class2id = {c: i for i, c in enumerate(self.class_names)}
        ifc_class = []
        pos = []
        extents = []
        rotations = []

        for n in nodes:
            obb = n["obb"]
            class_id = class2id[n["ifc_class"]]
            position = torch.tensor(obb["center"])
            extent = torch.tensor(obb["extent"])
            rotation = torch.tensor(obb["rotation"]).flatten()

            ifc_class.append(class_id)
            pos.append(position)
            extents.append(extent)
            rotations.append(rotation)

        data.ifc_class = torch.tensor(ifc_class)
        data.pos = torch.vstack(pos)
        data.extent = torch.vstack(extents)
        data.rotation = torch.vstack(rotations)

        rsat_file = path / "extent_factor_1.1.json"
        if rsat_file.exists():
            with rsat_file.open("r") as f:
                rsat_data = json.load(f)
            data.extra_connections_index = self._map_edges(
                rsat_data["extra_connections"], node_dict
            )
            data.missing_connections_index = self._map_edges(
                rsat_data["missing_connections"], node_dict
            )
            data.true_connections_index = self._map_edges(
                rsat_data["true_connections"], node_dict
            )
        return data

    def _map_edges(self, edges, node_dict):
        result = []
        for con in edges:
            source_id = node_dict[con[0]]
            target_id = node_dict[con[1]]
            result.append([source_id, target_id])

        if len(result) == 0:
            return torch.tensor(result, dtype=int)

        return torch.tensor(result, dtype=int).T

    def process(self):
        data_list = []
        for path in self.raw_paths:
            data = self.process_directory(path)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
