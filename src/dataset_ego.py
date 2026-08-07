from pathlib import Path

import networkx as nx
from loguru import logger
import typer

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (
    from_networkx,
    to_networkx,
)

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_utils import to_dense, get_root_node, estimate_marginal_distributions


EGO_DATASET = "Ego-small"
BASE_EGO_DATASET = "Citeseer"

app = typer.Typer()

class EgoSmallDataset(list):
    """List-like container holding attributed Ego-small graphs."""

    def __init__(
        self,
        graphs,
        num_node_classes: int,
        num_node_features: int,
    ):
        super().__init__(graphs)

        self.num_node_classes = num_node_classes
        self.num_node_features = num_node_features
        self.num_edge_classes = 2

def get_ego_data(
    path: Path,
    radius: int = 1,
    min_nodes: int = 4,
    max_nodes: int = 18,
    num_graphs: int = 200,
):
    logger.info(
        f"Constructing attributed {EGO_DATASET} from {BASE_EGO_DATASET}: "
        f"radius={radius}, nodes=[{min_nodes}, {max_nodes}], "
        f"num_graphs={num_graphs}."
    )

    dataset = Planetoid(
        root=path,
        name=BASE_EGO_DATASET,
        split="full",
        transform=T.RemoveIsolatedNodes(),
    )

    citation_data = dataset[0]

    citation_graph = to_networkx(
        citation_data,
        to_undirected=True,
        remove_self_loops=True,
    )

    # Keep only the largest connected component
    largest_component_nodes = max(
        nx.connected_components(citation_graph),
        key=len,
    )

    citation_graph = citation_graph.subgraph(
        largest_component_nodes
    ).copy()

    ego_graphs = []

    for root_node in sorted(citation_graph.nodes()):
        ego_graph = nx.ego_graph(
            citation_graph,
            root_node,
            radius=radius,
            center=True,
            undirected=True,
        )

        num_nodes = ego_graph.number_of_nodes()

        if not min_nodes <= num_nodes <= max_nodes:
            continue

        ego_graph.remove_edges_from(nx.selfloop_edges(ego_graph))

        # Keep the original CiteSeer node IDs in a stable order
        original_node_ids = torch.tensor(
            sorted(ego_graph.nodes()),
            dtype=torch.long,
        )

        # Relabel nodes locally as 0, ..., num_nodes - 1.
        node_mapping = {
            original_node_id: local_node_id
            for local_node_id, original_node_id
            in enumerate(original_node_ids.tolist())
        }

        local_ego_graph = nx.relabel_nodes(
            ego_graph,
            node_mapping,
            copy=True,
        )

        root_local_id = node_mapping[root_node]
        graph_data = from_networkx(local_ego_graph)

        # Root node as feature
        graph_data.x = get_root_node(
            graph=local_ego_graph,
            root_node=root_local_id,
        )

        # Original CiteSeer paper-topic labels.
        graph_data.y = citation_data.y[
            original_node_ids
        ].long()

        graph_data.original_node_ids = original_node_ids

        graph_data.graph_id = torch.tensor(
            [len(ego_graphs)],
            dtype=torch.long,
        )

        ego_graphs.append(graph_data)

        if len(ego_graphs) == num_graphs:
            break

    if len(ego_graphs) < num_graphs:
        raise ValueError(
            f"Only found {len(ego_graphs)} valid ego graphs, "
            f"but {num_graphs} were requested."
        )

    node_counts = torch.tensor(
        [graph.num_nodes for graph in ego_graphs],
        dtype=torch.long,
    )

    logger.info(
        f"Constructed {len(ego_graphs)} attributed graphs. "
        f"Node counts: min={node_counts.min().item()}, "
        f"max={node_counts.max().item()}, "
        f"mean={node_counts.float().mean().item():.2f}."
    )

    logger.info(
        f"Node feature dimension: {dataset.num_node_features}."
    )

    logger.info(
        f"Node classes: {dataset.num_classes}."
    )

    return EgoSmallDataset(
        graphs=ego_graphs,
        num_node_classes=dataset.num_classes,
        num_node_features=4,
    )
    
def construct_ego_dataloader(
    data,
    seed: int = 0,
    batch_size: int = 32,
    shuffle: bool = True,
):
    if len(data) == 0:
        raise ValueError("The Ego-small dataset contains no graphs.")

    generator = torch.Generator().manual_seed(seed)

    permutation = torch.randperm(
        len(data),
        generator=generator,
    ).tolist()

    shuffled_graphs = [data[index] for index in permutation]

    num_graphs = len(shuffled_graphs)
    num_train = int(0.8 * num_graphs)
    num_val = int(0.1 * num_graphs)

    train_graphs = shuffled_graphs[:num_train]
    val_graphs = shuffled_graphs[
        num_train:num_train + num_val
    ]
    test_graphs = shuffled_graphs[
        num_train + num_val:
    ]

    if not train_graphs or not val_graphs or not test_graphs:
        raise ValueError(
            "The requested split produced an empty train, validation, "
            "or test dataset."
        )

    node_marginals, edge_marginals = (
    estimate_marginal_distributions(
            graphs=train_graphs,
            num_node_classes=data.num_node_classes,
            num_edge_classes=data.num_edge_classes,
        )
    )

    logger.info(
        f"Training node marginals: {node_marginals.tolist()}"
    )

    logger.info(
        f"Training edge marginals: {edge_marginals.tolist()}"
    )

    logger.info(
        f"Ego-small split: train={len(train_graphs)}, "
        f"val={len(val_graphs)}, test={len(test_graphs)}."
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, node_marginals, edge_marginals


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / BASE_EGO_DATASET,
):
    data = get_ego_data(
        path=input_path,
        radius=1,
        min_nodes=4,
        max_nodes=18,
        num_graphs=200,
    )

    train_loader, val_loader, test_loader, _, _ = construct_ego_dataloader(
        data=data,
        batch_size=32,
        shuffle=True,
        seed=42,
    )

    batch = next(iter(train_loader))

    node_features, node_labels, adj, node_mask = to_dense(
        x=batch.x,
        y=batch.y,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, "edge_attr", None),
        batch=batch.batch,
        min_nodes=1,
        max_nodes=18,
    )

    print("node_features:", node_features.shape)
    print("node_labels:", node_labels.shape)
    print("adj:", adj.shape)
    print("mask:", node_mask.shape)

    print("Number of graphs:", len(data))
    print("Node classes:", data.num_node_classes)
    print("Node feature dimension:", data.num_node_features)
    print("Feature values:", torch.unique(node_features))
    print("Label values:", torch.unique(node_labels[node_mask]))
    print("Edge classes:", data.num_edge_classes)

    print(f"Train graphs: {len(train_loader.dataset)}")
    print(f"Validation graphs: {len(val_loader.dataset)}")
    print(f"Test graphs: {len(test_loader.dataset)}")

    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("Done testing!")


if __name__ == "__main__":
    app()
