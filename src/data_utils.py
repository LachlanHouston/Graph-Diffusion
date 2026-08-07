import torch
import networkx as nx
from torch_geometric.utils import (
    remove_self_loops,
    to_dense_adj,
    to_dense_batch,
    dense_to_sparse
)

from torch_geometric.data import Batch, Data

def get_root_node(
    graph: nx.Graph,
    root_node: int,
) -> torch.Tensor:
    roots = []

    for node in range(graph.number_of_nodes()):
        is_root = float(node == root_node)

        roots.append(
            [
                is_root,
            ]
        )

    return torch.tensor(
        roots,
        dtype=torch.float,
    )

def estimate_marginal_distributions(
    graphs,
    num_node_classes: int,
    num_edge_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate node- and edge-class marginal distributions.

    Edge class convention:
        0 = no edge
        1 = edge

    Only unordered, non-diagonal node pairs are counted.
    """

    node_counts = torch.zeros(
        num_node_classes,
        dtype=torch.float64,
    )

    edge_counts = torch.zeros(
        num_edge_classes,
        dtype=torch.float64,
    )

    for graph in graphs:
        labels = graph.y.long()

        node_counts += torch.bincount(
            labels,
            minlength=num_node_classes,
        ).to(torch.float64)

        num_nodes = graph.num_nodes

        possible_edges = (
            num_nodes * (num_nodes - 1) // 2
        )

        # PyG normally stores both directions for an undirected edge.
        num_edges = graph.edge_index.size(1) // 2

        num_non_edges = possible_edges - num_edges

        edge_counts[0] += num_non_edges
        edge_counts[1] += num_edges

    if node_counts.sum() == 0:
        raise ValueError(
            "Cannot estimate node marginals from an empty graph collection."
        )

    if edge_counts.sum() == 0:
        raise ValueError(
            "Cannot estimate edge marginals from graphs "
            "with no valid node pairs."
        )

    node_marginals = (
        node_counts / node_counts.sum()
    ).float()

    edge_marginals = (
        edge_counts / edge_counts.sum()
    ).float()

    return node_marginals, edge_marginals

def compute_noisy_structural_features(
    noisy_adj: torch.Tensor,
    node_mask: torch.Tensor,
    root_indicator: torch.Tensor,
) -> torch.Tensor:
    """
    Compute structural node features from the current noisy adjacency.

    Args:
        noisy_adj:
            Tensor of shape [batch_size, num_nodes, num_nodes].
            Edge class 0 is assumed to mean no edge, while values > 0
            are treated as edges.

        node_mask:
            Boolean tensor of shape [batch_size, num_nodes].

        root_indicator:
            Tensor of shape [batch_size, num_nodes] or
            [batch_size, num_nodes, 1], with exactly one root node
            marked per graph.

    Returns:
        Tensor of shape [batch_size, num_nodes, 4] containing:

        0: normalized noisy degree
        1: noisy clustering coefficient
        2: normalized noisy distance from the root
        3: fixed root indicator
    """
    if root_indicator.ndim == 3:
        root_indicator = root_indicator.squeeze(-1)

    node_mask = node_mask.bool()

    batch_size, num_nodes, _ = noisy_adj.shape

    pair_mask = (
        node_mask.unsqueeze(1)
        & node_mask.unsqueeze(2)
    )

    adjacency = (noisy_adj > 0).float()

    adjacency = torch.maximum(
        adjacency,
        adjacency.transpose(1, 2),
    )

    adjacency = adjacency * pair_mask.float()

    diagonal_mask = torch.eye(
        num_nodes,
        device=adjacency.device,
        dtype=torch.bool,
    ).unsqueeze(0)

    adjacency = adjacency.masked_fill(
        diagonal_mask,
        0.0,
    )

    # ---------------------------------------------------------
    # Degree
    # ---------------------------------------------------------

    degrees = adjacency.sum(dim=-1)

    valid_node_counts = (
        node_mask.sum(dim=-1, keepdim=True).float()
    )

    degree_denominator = (
        valid_node_counts - 1
    ).clamp(min=1)

    normalized_degree = (
        degrees / degree_denominator
    )

    # ---------------------------------------------------------
    # Clustering coefficient
    # ---------------------------------------------------------

    adjacency_squared = torch.bmm(
        adjacency,
        adjacency,
    )

    triangles_per_node = (
        adjacency_squared * adjacency
    ).sum(dim=-1) / 2.0

    possible_neighbor_pairs = (
        degrees * (degrees - 1)
    ) / 2.0

    clustering = torch.where(
        possible_neighbor_pairs > 0,
        triangles_per_node
        / possible_neighbor_pairs.clamp(min=1),
        torch.zeros_like(triangles_per_node),
    )

    # ---------------------------------------------------------
    # Root indicator
    # ---------------------------------------------------------

    root_indicator = (
        root_indicator.float()
        * node_mask.float()
    )

    root_indices = root_indicator.argmax(dim=-1)

    # ---------------------------------------------------------
    # Shortest-path distance from root
    # ---------------------------------------------------------

    distances = torch.full(
        size=(batch_size, num_nodes),
        fill_value=-1,
        dtype=torch.long,
        device=adjacency.device,
    )

    frontier = torch.zeros(
        size=(batch_size, num_nodes),
        dtype=torch.bool,
        device=adjacency.device,
    )

    frontier.scatter_(
        dim=1,
        index=root_indices.unsqueeze(1),
        value=True,
    )

    frontier = frontier & node_mask
    visited = frontier.clone()
    distances[frontier] = 0
    adjacency_bool = adjacency.bool()

    for distance in range(1, num_nodes):
        next_frontier = torch.bmm(
            frontier.float().unsqueeze(1),
            adjacency_bool.float(),
        ).squeeze(1) > 0

        next_frontier = (
            next_frontier
            & node_mask
            & ~visited
        )

        if not next_frontier.any():
            break

        distances[next_frontier] = distance

        visited = visited | next_frontier
        frontier = next_frontier

    normalized_distance = torch.zeros(
        size=(batch_size, num_nodes),
        dtype=adjacency.dtype,
        device=adjacency.device,
    )

    for graph_idx in range(batch_size):
        reachable = (
            distances[graph_idx] >= 0
        ) & node_mask[graph_idx]

        if reachable.any():
            maximum_distance = (
                distances[graph_idx, reachable]
                .max()
                .clamp(min=1)
            )

            normalized_distance[
                graph_idx,
                reachable,
            ] = (
                distances[
                    graph_idx,
                    reachable,
                ].float()
                / maximum_distance.float()
            )

        unreachable = (
            node_mask[graph_idx]
            & ~reachable
        )

        normalized_distance[
            graph_idx,
            unreachable,
        ] = 1.0

    features = torch.stack(
        [
            normalized_degree,
            clustering,
            normalized_distance,
            root_indicator,
        ],
        dim=-1,
    )

    features = (
        features
        * node_mask.unsqueeze(-1).float()
    )

    return features

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


def sampled_tensors_to_batch(
    sampled_x: torch.Tensor,
    sampled_e: torch.Tensor,
    node_mask: torch.Tensor,
) -> Batch:
    """
    Convert dense sampled node labels and adjacency matrices into a PyG Batch.

    Args:
        sampled_x:
            Node labels with shape [batch_size, num_nodes], or one-hot/logit
            node values with shape [batch_size, num_nodes, num_classes].

        sampled_e:
            Edge classes with shape [batch_size, num_nodes, num_nodes].

        node_mask:
            Boolean mask with shape [batch_size, num_nodes].

    Returns:
        A PyG Batch containing one Data object per sampled graph.
    """
    sampled_x = sampled_x.detach().cpu()
    sampled_e = sampled_e.detach().cpu()
    node_mask = node_mask.detach().cpu().bool()

    if sampled_x.ndim == 3:
        sampled_x = sampled_x.argmax(dim=-1)

    data_list = []

    for graph_idx in range(sampled_e.size(0)):
        valid_mask = node_mask[graph_idx]
        num_valid_nodes = int(valid_mask.sum().item())

        node_labels = sampled_x[
            graph_idx,
            valid_mask,
        ].long()

        adjacency = sampled_e[
            graph_idx,
            :num_valid_nodes,
            :num_valid_nodes,
        ]

        adjacency = (adjacency > 0).long()

        adjacency = torch.maximum(
            adjacency,
            adjacency.transpose(0, 1),
        )

        adjacency.fill_diagonal_(0)

        edge_index, _ = dense_to_sparse(adjacency)

        graph = Data(
            y=node_labels,
            edge_index=edge_index,
            num_nodes=num_valid_nodes,
        )

        data_list.append(graph)

    return Batch.from_data_list(data_list)