from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops, k_hop_subgraph, subgraph

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

DATASET = "Citeseer"

app = typer.Typer()


def connected_node_subset(edge_index, center_local: int, num_nodes: int, max_nodes: int):
    if num_nodes <= max_nodes:
        return torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)

    neighbors = [[] for _ in range(num_nodes)]
    src_nodes = edge_index[0].detach().cpu().tolist()
    dst_nodes = edge_index[1].detach().cpu().tolist()

    for src, dst in zip(src_nodes, dst_nodes):
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    visited = {int(center_local)}
    queue = [int(center_local)]
    selected = []

    while queue and len(selected) < max_nodes:
        node = queue.pop(0)
        selected.append(node)

        for neighbor in neighbors[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return torch.tensor(selected, dtype=torch.long, device=edge_index.device)


def largest_connected_component(edge_index, num_nodes: int):
    neighbors = [[] for _ in range(num_nodes)]
    src_nodes = edge_index[0].detach().cpu().tolist()
    dst_nodes = edge_index[1].detach().cpu().tolist()

    for src, dst in zip(src_nodes, dst_nodes):
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    visited = set()
    components = []

    for start in range(num_nodes):
        if start in visited:
            continue

        queue = [start]
        visited.add(start)
        component = []

        while queue:
            node = queue.pop(0)
            component.append(node)

            for neighbor in neighbors[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    largest = max(components, key=len)
    return torch.tensor(largest, dtype=torch.long, device=edge_index.device)


def get_data(path: Path):
    logger.info(f"Loading {DATASET}.")

    transform = T.Compose([
        T.RandomNodeSplit(num_val=100, num_test=100),
        T.RemoveIsolatedNodes(),
    ])

    dataset = Planetoid(
        root=path,
        name=DATASET,
        split="full",
        transform=transform,
    )

    data = dataset[0]
    data.num_node_classes = dataset.num_classes
    data.num_edge_classes = 2

    return data


def extract_k_hop(data, root_node, num_hops=2, max_nodes: int = 64, min_nodes: int = 2):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=int(root_node),
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    if subset.numel() > max_nodes:
        keep_local = connected_node_subset(
            edge_index=edge_index,
            center_local=int(mapping.item()),
            num_nodes=subset.numel(),
            max_nodes=max_nodes,
        )
        subset = subset[keep_local]

        edge_index, edge_mask = subgraph(
            subset=subset,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
    else:
        edge_attr_mask = edge_mask

    if subset.numel() < min_nodes:
        return None

    component_local = largest_connected_component(
        edge_index=edge_index,
        num_nodes=subset.numel(),
    )

    if component_local.numel() < min_nodes:
        return None

    if component_local.numel() < subset.numel():
        subset = subset[component_local]
        edge_index, edge_mask = subgraph(
            subset=subset,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

    node_features = data.x[subset].float()
    node_labels = data.y[subset].long()

    sub_data = Data(
        x=node_features,
        edge_index=edge_index,
        y=node_labels,
        original_node_ids=subset,
        num_nodes=subset.numel(),
    )

    if data.edge_attr is not None:
        sub_data.edge_attr = data.edge_attr[edge_mask]

    return sub_data

    
def construct_dataloader(
    data,
    num_hops: int = 2,
    max_nodes: int = 64,
    min_nodes: int = 8,
    seed: int = 0,
    batch_size: int = 32,
    shuffle: bool = True,
):
    split_nodes = {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "val": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }

    generator = torch.Generator().manual_seed(seed)
    loaders = {}

    for split_name, nodes in split_nodes.items():
        permutation = torch.randperm(nodes.numel(), generator=generator)
        nodes = nodes[permutation]

        subgraphs = []
        for root_node in nodes:
            subgraph_data = extract_k_hop(
                data=data,
                root_node=root_node,
                num_hops=num_hops,
                max_nodes=max_nodes,
                min_nodes=min_nodes,
            )

            if subgraph_data is not None:
                subgraphs.append(subgraph_data)

        if len(subgraphs) == 0:
            raise ValueError(
                f"No connected {split_name} k-hop subgraphs found with "
                f"min_nodes={min_nodes} and max_nodes={max_nodes}."
            )

        loaders[split_name] = DataLoader(
            subgraphs,
            batch_size=batch_size,
            shuffle=shuffle if split_name == "train" else False,
            drop_last=split_name == "train",
        )

    return loaders["train"], loaders["val"], loaders["test"]


def estimate_class_distributions(
    data,
    loader,
    max_nodes: int | None = None,
    min_nodes: int = 1,
):
    node_counts = torch.zeros(data.num_node_classes, dtype=torch.long)
    edge_counts = torch.zeros(data.num_edge_classes, dtype=torch.long)

    for batch in loader:
        node_features, node_labels, adj, node_mask = to_dense(
            x=batch.x,
            y=batch.y,
            edge_index=batch.edge_index,
            edge_attr=getattr(batch, "edge_attr", None),
            batch=batch.batch,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )

        valid_x = node_labels[node_mask]
        node_counts += torch.bincount(
            valid_x.long(),
            minlength=data.num_node_classes,
        ).cpu()

        _, num_nodes, _ = adj.shape
        valid_pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        upper_mask = torch.triu(
            torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=adj.device),
            diagonal=1,
        )
        edge_values = adj[valid_pair_mask & upper_mask.unsqueeze(0)]

        edge_counts += torch.bincount(
            edge_values.long(),
            minlength=data.num_edge_classes,
        ).cpu()

    node_probs = node_counts.float() / node_counts.sum().clamp_min(1)
    edge_probs = edge_counts.float() / edge_counts.sum().clamp_min(1)

    return {
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "node_probs": node_probs,
        "edge_probs": edge_probs,
    }


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
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / DATASET,
    output_path: Path = PROCESSED_DATA_DIR / DATASET,
    # ----------------------------------------------
):
    data = get_data(output_path)
    train_loader, val_loader, test_loader = construct_dataloader(
        data=data,
        num_hops=3,
        max_nodes=16,
        min_nodes=4,
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
        min_nodes=2,
        max_nodes=None,
    )

    print("node_features:", node_features.shape)
    print("node_labels:", node_labels.shape)
    print("adj:", adj.shape)
    print("mask:", node_mask.shape)

    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("Done Testing!")

if __name__ == "__main__":
    app()
