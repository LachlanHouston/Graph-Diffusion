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
        T.RandomNodeSplit(num_val=500, num_test=500),
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

    print(f"Raw data statistics for: {DATASET}")
    print("-"*25)
    print(data)
    print("num_nodes:", data.num_nodes)
    print("num_edges:", data.num_edges)
    print("num_features:", data.num_features)
    print("num_classes:", dataset.num_classes)
    print("discrete node feature:", data.y)
    print("num_node_classes:", data.num_node_classes)
    print("num_edge_classes:", data.num_edge_classes)

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

    node_labels = data.y[subset].long()

    sub_data = Data(
        x=node_labels,
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
    num_samples: int = 10_000,
    num_hops: int = 2,
    max_nodes: int = 64,
    min_nodes: int = 8,
    seed: int = 0,
    batch_size: int = 32,
    shuffle: bool = False,
):
    train_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(train_nodes.numel(), generator=generator)

    sample_nodes = train_nodes[perm]

    subgraphs = []
    for root_node in sample_nodes:
        subgraph_data = extract_k_hop(
            data=data,
            root_node=root_node,
            num_hops=num_hops,
            max_nodes=max_nodes,
            min_nodes=min_nodes,
        )

        if subgraph_data is None:
            continue

        subgraphs.append(subgraph_data)

        if len(subgraphs) >= num_samples:
            break

    if len(subgraphs) == 0:
        raise ValueError(
            f"No connected k-hop subgraphs found with min_nodes={min_nodes} and max_nodes={max_nodes}."
        )

    logger.info(
        f"Using KHopSubgraphDataset with num_samples={num_samples}, "
        f"num_hops={num_hops}, batch_size={batch_size}, "
        f"min_nodes={min_nodes}, max_nodes={max_nodes}."
    )

    return DataLoader(
        subgraphs,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def to_dense(
    x,
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
    batch = graph_id_map[batch[keep_node]]

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    keep_edge = keep_node[edge_index[0]] & keep_node[edge_index[1]]
    edge_index = old_to_new_node[edge_index[:, keep_edge]]

    if edge_attr is not None:
        edge_attr = edge_attr[keep_edge]

    X, node_mask = to_dense_batch(
        x=x,
        batch=batch,
        max_num_nodes=max_nodes,
    )
    X = X.long()

    A = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=X.size(1),
    )

    A = (A > 0).long()
    A = torch.maximum(A, A.transpose(1, 2))

    return X, A, node_mask


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / DATASET,
    output_path: Path = PROCESSED_DATA_DIR / DATASET,
    # ----------------------------------------------
):
    data = get_data(output_path)
    loader = construct_dataloader(
        data=data,
        num_samples=10_000,
        num_hops=3,
        max_nodes=64,
        min_nodes=2,
        batch_size=32,
        shuffle=True,
        seed=42,
    )

    batch = next(iter(loader))

    print(batch)
    print(batch.x.shape)
    print(batch.edge_index.shape)

    x, adj, node_mask = to_dense(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, "edge_attr", None),
        batch=batch.batch,
        min_nodes=2,
        max_nodes=None,
    )

    print("x:", x.shape)
    print("adj:", adj.shape)
    print("mask:", node_mask.shape)
    print("x dtype:", x.dtype)
    print("adj dtype:", adj.dtype)
    print("unique node classes in batch:", torch.unique(x[node_mask]).tolist())
    print("unique edge classes in batch:", torch.unique(adj).tolist())

    num_batches = 0
    for i, _ in enumerate(loader):
        num_batches += 1
    print(num_batches)

    print("Done Testing!")

if __name__ == "__main__":
    app()
