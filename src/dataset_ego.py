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
    remove_self_loops,
    to_dense_adj,
    to_dense_batch,
    to_networkx,
)

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


DATASET = "Ego-small"
BASE_DATASET = "Citeseer"

app = typer.Typer()

class EgoSmallDataset(list):
    """List-like container holding the generated Ego-small graphs."""

    def __init__(self, graphs):
        super().__init__(graphs)
        self.num_node_classes = 1
        self.num_edge_classes = 2

def get_data(
    path: Path,
    radius: int = 1,
    min_nodes: int = 4,
    max_nodes: int = 18,
    num_graphs: int = 200,
):
    logger.info(
        f"Constructing {DATASET} from {BASE_DATASET}: "
        f"radius={radius}, nodes=[{min_nodes}, {max_nodes}], "
        f"num_graphs={num_graphs}."
    )

    dataset = Planetoid(
        root=path,
        name=BASE_DATASET,
        split="full",
        transform=T.RemoveIsolatedNodes(),
    )

    citation_data = dataset[0]

    citation_graph = to_networkx(
        citation_data,
        to_undirected=True,
        remove_self_loops=True,
    )

    # Work with the largest connected component, as commonly done for
    # Ego-small preprocessing.
    largest_component_nodes = max(
        nx.connected_components(citation_graph),
        key=len,
    )

    citation_graph = citation_graph.subgraph(
        largest_component_nodes
    ).copy()

    citation_graph = nx.convert_node_labels_to_integers(
        citation_graph,
        ordering="sorted",
    )

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

        # Give every graph local node indices 0, ..., n - 1.
        ego_graph = nx.convert_node_labels_to_integers(
            ego_graph,
            ordering="sorted",
        )

        graph_data = from_networkx(ego_graph)

        # Standard Ego-small generation is topology-only. Every valid node
        # therefore receives the same dummy feature and class.
        graph_data.x = torch.ones(
            (graph_data.num_nodes, 1),
            dtype=torch.float,
        )
        graph_data.y = torch.zeros(
            graph_data.num_nodes,
            dtype=torch.long,
        )

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
        f"Constructed {len(ego_graphs)} graphs. "
        f"Node counts: min={node_counts.min().item()}, "
        f"max={node_counts.max().item()}, "
        f"mean={node_counts.float().mean().item():.2f}."
    )

    return EgoSmallDataset(ego_graphs)
    
def construct_dataloader(
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

    return train_loader, val_loader, test_loader


def to_dense(
    x,
    y,
    edge_index,
    edge_attr,
    batch,
    min_nodes: int = 1,
    max_nodes: int | None = None,
):
    num_graphs = int(batch.max().item()) + 1
    node_counts = torch.bincount(batch, minlength=num_graphs)
    keep_graph = node_counts >= min_nodes

    if keep_graph.sum() == 0:
        raise ValueError(
            f"No graphs in this batch have at least {min_nodes} nodes. "
            f"Node counts were: {node_counts.tolist()}"
        )

    graph_id_map = torch.full(
        size=(num_graphs,),
        fill_value=-1,
        dtype=torch.long,
        device=batch.device,
    )
    graph_id_map[keep_graph] = torch.arange(
        keep_graph.sum(),
        dtype=torch.long,
        device=batch.device,
    )

    keep_node = keep_graph[batch]
    old_to_new_node = torch.full(
        size=(x.size(0),),
        fill_value=-1,
        dtype=torch.long,
        device=x.device,
    )
    old_to_new_node[keep_node] = torch.arange(
        keep_node.sum(),
        dtype=torch.long,
        device=x.device,
    )

    x = x[keep_node]
    y = y[keep_node]
    batch = graph_id_map[batch[keep_node]]

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    keep_edge = keep_node[edge_index[0]] & keep_node[edge_index[1]]
    edge_index = old_to_new_node[edge_index[:, keep_edge]]

    if edge_attr is not None:
        edge_attr = edge_attr[keep_edge]

    node_features, node_mask = to_dense_batch(
        x=x,
        batch=batch,
        max_num_nodes=max_nodes,
    )
    node_features = node_features.float()

    node_labels, label_mask = to_dense_batch(
        x=y,
        batch=batch,
        max_num_nodes=node_features.size(1),
        fill_value=0,
    )
    node_labels = node_labels.long()

    if not torch.equal(node_mask, label_mask):
        raise RuntimeError("Feature and label masks do not match after densification.")

    adj = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=node_features.size(1),
    )

    adj = (adj > 0).long()
    adj = torch.maximum(adj, adj.transpose(1, 2))

    return node_features, node_labels, adj, node_mask


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / BASE_DATASET,
):
    data = get_data(
        path=input_path,
        radius=1,
        min_nodes=4,
        max_nodes=18,
        num_graphs=200,
    )

    train_loader, val_loader, test_loader = construct_dataloader(
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
