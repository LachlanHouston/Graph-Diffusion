import torch
import networkx as nx
from torch_geometric.utils import (
    remove_self_loops,
    to_dense_adj,
    to_dense_batch,
)

def compute_structural_features(
    graph: nx.Graph,
    root_node: int,
    max_nodes: int,
) -> torch.Tensor:
    degrees = dict(graph.degree())
    clustering = nx.clustering(graph)
    distances = nx.single_source_shortest_path_length(
        graph,
        root_node,
    )

    features = []

    for node in range(graph.number_of_nodes()):
        normalized_degree = (
            degrees[node] / max(max_nodes - 1, 1)
        )

        clustering_coefficient = clustering[node]

        normalized_distance = (
            distances[node] / max(max(distances.values()), 1)
        )

        is_root = float(node == root_node)

        features.append(
            [
                normalized_degree,
                clustering_coefficient,
                normalized_distance,
                is_root,
            ]
        )

    return torch.tensor(
        features,
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