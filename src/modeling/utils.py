import math

import torch.nn.functional as F

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch
from torch import Tensor

# -----------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------

def masked_upper_mse(pred: Tensor, target: Tensor, node_mask: Tensor | None = None):
    """
    MSE over the upper triangular adjacency entries.

    pred: [B, N, N]
    target: [B, N, N]
    node_mask: optional [B, N]
    """
    _, N, _ = pred.shape
    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=pred.device),
        diagonal=1,
    )

    if node_mask is None:
        mask = upper_mask.unsqueeze(0).expand_as(pred)
    else:
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        mask = pair_mask & upper_mask.unsqueeze(0)

    return torch.nn.functional.mse_loss(pred[mask], target[mask])


def symmetric_noise_like(adj: Tensor):
    """Sample Gaussian noise with the same symmetry as an undirected adjacency."""
    noise = torch.randn_like(adj)
    noise = torch.triu(noise, diagonal=1)
    return noise + noise.transpose(1, 2)


def symmetrize(adj: Tensor):
    return 0.5 * (adj + adj.transpose(1, 2))


def remove_diagonal(adj: Tensor):
    _, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device).unsqueeze(0)
    return adj * (1.0 - eye)


def apply_node_mask(adj: Tensor, node_mask: Tensor):
    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    return adj * pair_mask.float()


def binarize_samples(samples: Tensor, threshold: float):
    samples = symmetrize(samples)
    adj = (samples > threshold).float()
    adj = remove_diagonal(adj)
    adj = symmetrize(adj)
    return (adj > 0.5).float()


# -----------------------------------------------------------------------------
# NetworkX conversion and graph statistics
# -----------------------------------------------------------------------------

def graph_from_adjacency(x, e, node_mask):
    """
    Convert node features and adjacency into a masked NetworkX graph.

    Accepts either torch.Tensor or numpy.ndarray inputs.
    """

    def to_numpy(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    x = to_numpy(x)
    e = to_numpy(e)
    node_mask = to_numpy(node_mask).astype(bool)

    valid_nodes = np.where(node_mask)[0]

    feats = None
    if x is not None:
        feats = x[valid_nodes]

    adj = e[valid_nodes][:, valid_nodes]

    adj = np.maximum(adj, adj.T)
    adj = (adj > 0.5).astype(np.int32)

    graph = nx.from_numpy_array(adj)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return feats, graph


def graph_only_from_adjacency(e, node_mask):
    _, graph = graph_from_adjacency(x=None, e=e, node_mask=node_mask)
    return graph


def graph_degree_histogram(graph: nx.Graph) -> Tensor:
    degrees = [degree for _, degree in graph.degree()]
    if len(degrees) == 0:
        return torch.zeros(1)

    hist = torch.bincount(torch.tensor(degrees, dtype=torch.long)).float()
    return hist / hist.sum().clamp_min(1.0)


def graph_clustering_histogram(graph: nx.Graph, bins: int = 20) -> Tensor:
    if graph.number_of_nodes() == 0:
        return torch.zeros(bins)

    coeffs = torch.tensor(list(nx.clustering(graph).values()), dtype=torch.float)
    hist = torch.histc(coeffs, bins=bins, min=0.0, max=1.0)
    return hist / hist.sum().clamp_min(1.0)


def graph_orbit_features(graph: nx.Graph) -> Tensor:
    """
    Lightweight graphlet/orbit-style proxy features.

    This is not a full ORCA orbit count. It tracks simple local motifs that are
    cheap to compute with NetworkX: degree, isolated nodes, leaves, wedges,
    triangles, and 3-stars.
    """
    if graph.number_of_nodes() == 0:
        return torch.zeros(6)

    triangles = nx.triangles(graph)
    features = []

    for node in graph.nodes():
        degree = graph.degree(node)
        triangle_count = triangles[node]
        wedge_count = max(math.comb(degree, 2) - triangle_count, 0) if degree >= 2 else 0
        three_star_count = math.comb(degree, 3) if degree >= 3 else 0
        leaf = int(degree == 1)
        isolated = int(degree == 0)

        features.append(
            [
                float(degree),
                float(isolated),
                float(leaf),
                float(wedge_count),
                float(triangle_count),
                float(three_star_count),
            ]
        )

    return torch.tensor(features, dtype=torch.float).mean(dim=0)


def pad_stat_vectors(stats: list[Tensor]) -> Tensor:
    if len(stats) == 0:
        return torch.empty(0, 1)

    max_len = max(stat.numel() for stat in stats)
    padded = []

    for stat in stats:
        stat = stat.flatten().float()
        if stat.numel() < max_len:
            stat = torch.nn.functional.pad(stat, (0, max_len - stat.numel()))
        padded.append(stat)

    return torch.stack(padded, dim=0)


def gaussian_kernel_matrix(x: Tensor, y: Tensor, sigma: float | None = None):
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(x.size(0), y.size(0))

    dist = torch.cdist(x, y, p=2).pow(2)

    if sigma is None:
        all_dist = dist.detach().flatten()
        positive_dist = all_dist[all_dist > 0]
        sigma = positive_dist.median().sqrt().item() if positive_dist.numel() > 0 else 1.0

    gamma = 1.0 / (2.0 * max(sigma, 1e-6) ** 2)
    return torch.exp(-gamma * dist)


def mmd_from_stats(real_stats: list[Tensor], sampled_stats: list[Tensor]) -> float:
    """Biased MMD estimator using one shared Gaussian-kernel bandwidth."""
    x = pad_stat_vectors(real_stats)
    y = pad_stat_vectors(sampled_stats)

    if x.size(0) == 0 or y.size(0) == 0:
        return float("nan")

    max_dim = max(x.size(1), y.size(1))
    if x.size(1) < max_dim:
        x = torch.nn.functional.pad(x, (0, max_dim - x.size(1)))
    if y.size(1) < max_dim:
        y = torch.nn.functional.pad(y, (0, max_dim - y.size(1)))

    combined = torch.cat([x, y], dim=0)
    combined_dist = torch.cdist(combined, combined, p=2).pow(2)
    positive_dist = combined_dist[combined_dist > 0]
    sigma = (
        positive_dist.median().sqrt().item()
        if positive_dist.numel() > 0
        else 1.0
    )

    k_xx = gaussian_kernel_matrix(x, x, sigma=sigma)
    k_yy = gaussian_kernel_matrix(y, y, sigma=sigma)
    k_xy = gaussian_kernel_matrix(x, y, sigma=sigma)

    return (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).item()


def graph_adjacency_signature(graph: nx.Graph) -> str:
    """Permutation-invariant structural signature for an unlabeled graph."""
    return nx.weisfeiler_lehman_graph_hash(graph)


def graph_uniqueness(graphs: list[nx.Graph]) -> float:
    if len(graphs) == 0:
        return float("nan")

    signatures = [graph_adjacency_signature(graph) for graph in graphs]
    return len(set(signatures)) / len(signatures)


def graph_num_nodes(graph: nx.Graph) -> float:
    return float(graph.number_of_nodes())


def graph_num_edges(graph: nx.Graph) -> float:
    return float(graph.number_of_edges())


def graph_density(graph: nx.Graph) -> float:
    if graph.number_of_nodes() < 2:
        return 0.0
    return float(nx.density(graph))


def graph_average_degree(graph: nx.Graph) -> float:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return 0.0
    return float(2.0 * graph.number_of_edges() / num_nodes)


def graph_max_degree(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(max(degree for _, degree in graph.degree()))


def graph_average_clustering(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(nx.average_clustering(graph))


def graph_num_components(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(nx.number_connected_components(graph))


def graph_largest_component_size(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(max(len(component) for component in nx.connected_components(graph)))


def graph_largest_component_fraction(graph: nx.Graph) -> float:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return 0.0
    return graph_largest_component_size(graph) / num_nodes


def graph_connected_fraction(graphs: list[nx.Graph]) -> float:
    if len(graphs) == 0:
        return float("nan")
    connected = [
        float(graph.number_of_nodes() > 0 and nx.is_connected(graph))
        for graph in graphs
    ]
    return sum(connected) / len(connected)


def mean_graph_stat(graphs: list[nx.Graph], statistic) -> float:
    if len(graphs) == 0:
        return float("nan")
    values = [float(statistic(graph)) for graph in graphs]
    return sum(values) / len(values)


def std_graph_stat(graphs: list[nx.Graph], statistic) -> float:
    if len(graphs) == 0:
        return float("nan")
    values = torch.tensor(
        [float(statistic(graph)) for graph in graphs],
        dtype=torch.float,
    )
    return values.std(unbiased=False).item()


def evaluate_generated_graphs(real_e, sampled_e, node_mask):
    """
    Evaluate generated undirected graphs using direct graph statistics and
    distributional metrics commonly reported for graph generative models.

    Node labels are intentionally ignored here. Each graph is first restricted
    to valid nodes from node_mask.
    """
    num_graphs = min(real_e.size(0), sampled_e.size(0), node_mask.size(0))
    real_e = real_e[:num_graphs]
    sampled_e = sampled_e[:num_graphs]
    node_mask = node_mask[:num_graphs]

    real_graphs = [
        graph_only_from_adjacency(real_e[i], node_mask[i])
        for i in range(num_graphs)
    ]
    sampled_graphs = [
        graph_only_from_adjacency(sampled_e[i], node_mask[i])
        for i in range(num_graphs)
    ]

    real_degree_stats = [graph_degree_histogram(graph) for graph in real_graphs]
    sampled_degree_stats = [graph_degree_histogram(graph) for graph in sampled_graphs]

    real_cluster_stats = [graph_clustering_histogram(graph) for graph in real_graphs]
    sampled_cluster_stats = [graph_clustering_histogram(graph) for graph in sampled_graphs]

    real_orbit_stats = [graph_orbit_features(graph) for graph in real_graphs]
    sampled_orbit_stats = [graph_orbit_features(graph) for graph in sampled_graphs]

    metrics = {
        # Distributional metrics used in graph-generation evaluation.
        "degree_mmd": mmd_from_stats(real_degree_stats, sampled_degree_stats),
        "cluster_mmd": mmd_from_stats(real_cluster_stats, sampled_cluster_stats),
        "orbit_mmd": mmd_from_stats(real_orbit_stats, sampled_orbit_stats),

        # Sample diversity.
        "uniqueness": graph_uniqueness(sampled_graphs),

        # Graph-size statistics.
        "real_num_nodes_mean": mean_graph_stat(real_graphs, graph_num_nodes),
        "sampled_num_nodes_mean": mean_graph_stat(sampled_graphs, graph_num_nodes),
        "real_num_nodes_std": std_graph_stat(real_graphs, graph_num_nodes),
        "sampled_num_nodes_std": std_graph_stat(sampled_graphs, graph_num_nodes),
        "real_num_edges_mean": mean_graph_stat(real_graphs, graph_num_edges),
        "sampled_num_edges_mean": mean_graph_stat(sampled_graphs, graph_num_edges),
        "real_num_edges_std": std_graph_stat(real_graphs, graph_num_edges),
        "sampled_num_edges_std": std_graph_stat(sampled_graphs, graph_num_edges),

        # Density and degree statistics.
        "real_density_mean": mean_graph_stat(real_graphs, graph_density),
        "sampled_density_mean": mean_graph_stat(sampled_graphs, graph_density),
        "real_density_std": std_graph_stat(real_graphs, graph_density),
        "sampled_density_std": std_graph_stat(sampled_graphs, graph_density),
        "real_avg_degree_mean": mean_graph_stat(real_graphs, graph_average_degree),
        "sampled_avg_degree_mean": mean_graph_stat(sampled_graphs, graph_average_degree),
        "real_avg_degree_std": std_graph_stat(real_graphs, graph_average_degree),
        "sampled_avg_degree_std": std_graph_stat(sampled_graphs, graph_average_degree),
        "real_max_degree_mean": mean_graph_stat(real_graphs, graph_max_degree),
        "sampled_max_degree_mean": mean_graph_stat(sampled_graphs, graph_max_degree),

        # Local structure.
        "real_avg_clustering_mean": mean_graph_stat(real_graphs, graph_average_clustering),
        "sampled_avg_clustering_mean": mean_graph_stat(sampled_graphs, graph_average_clustering),
        "real_avg_clustering_std": std_graph_stat(real_graphs, graph_average_clustering),
        "sampled_avg_clustering_std": std_graph_stat(sampled_graphs, graph_average_clustering),

        # Connectivity.
        "real_num_components_mean": mean_graph_stat(real_graphs, graph_num_components),
        "sampled_num_components_mean": mean_graph_stat(sampled_graphs, graph_num_components),
        "real_num_components_std": std_graph_stat(real_graphs, graph_num_components),
        "sampled_num_components_std": std_graph_stat(sampled_graphs, graph_num_components),
        "real_largest_component_mean": mean_graph_stat(real_graphs, graph_largest_component_size),
        "sampled_largest_component_mean": mean_graph_stat(sampled_graphs, graph_largest_component_size),
        "real_largest_component_fraction_mean": mean_graph_stat(
            real_graphs,
            graph_largest_component_fraction,
        ),
        "sampled_largest_component_fraction_mean": mean_graph_stat(
            sampled_graphs,
            graph_largest_component_fraction,
        ),
        "real_connected_fraction": graph_connected_fraction(real_graphs),
        "sampled_connected_fraction": graph_connected_fraction(sampled_graphs),
    }

    return metrics


def masked_node_cross_entropy(logits, target, node_mask=None):
    """
    logits: [B, N, X_classes]
    target: [B, N]
    node_mask: [B, N]
    """
    if node_mask is not None:
        logits = logits[node_mask]
        target = target[node_mask]
    else:
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)

    return F.cross_entropy(logits, target.long())


def masked_upper_edge_cross_entropy(logits, target, node_mask=None):
    """
    logits: [B, N, N, E_classes]
    target: [B, N, N]
    node_mask: [B, N]
    """
    B, N, _, _ = logits.shape

    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=logits.device),
        diagonal=1,
    ).unsqueeze(0).expand(B, N, N)

    if node_mask is not None:
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        upper_mask = upper_mask & pair_mask

    logits = logits[upper_mask]
    target = target[upper_mask]

    return F.cross_entropy(logits, target.long())

def masked_multiclass_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    num_classes: int,
    prefix: str,
):
    pred = pred[node_mask].detach()
    target = target[node_mask].detach()

    if target.numel() == 0:
        return {
            f"{prefix}/accuracy": 0.0,
            f"{prefix}/macro_precision": 0.0,
            f"{prefix}/macro_recall": 0.0,
            f"{prefix}/macro_f1": 0.0,
        }

    eps = 1e-8
    accuracy = (pred == target).float().mean().item()

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for class_idx in range(num_classes):
        pred_is_class = pred == class_idx
        target_is_class = target == class_idx

        true_positive = (pred_is_class & target_is_class).float().sum()
        false_positive = (pred_is_class & ~target_is_class).float().sum()
        false_negative = (~pred_is_class & target_is_class).float().sum()
        support = target_is_class.float().sum()

        precision = true_positive / (true_positive + false_positive + eps)
        recall = true_positive / (true_positive + false_negative + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class_metrics[f"{prefix}/class_{class_idx}_precision"] = precision.item()
        per_class_metrics[f"{prefix}/class_{class_idx}_recall"] = recall.item()
        per_class_metrics[f"{prefix}/class_{class_idx}_f1"] = f1.item()
        per_class_metrics[f"{prefix}/class_{class_idx}_support"] = support.item()

    macro_precision = torch.stack(precisions).mean().item()
    macro_recall = torch.stack(recalls).mean().item()
    macro_f1 = torch.stack(f1s).mean().item()

    metrics = {
        f"{prefix}/accuracy": accuracy,
        f"{prefix}/macro_precision": macro_precision,
        f"{prefix}/macro_recall": macro_recall,
        f"{prefix}/macro_f1": macro_f1,
    }
    metrics.update(per_class_metrics)
    return metrics

def node_colours_from_features(x, num_nodes: int, cmap):
    if x is None:
        return None

    classes = x.long().flatten().tolist()
    colours = [cmap(node_class % cmap.N) for node_class in classes]

    if len(colours) != num_nodes:
        return None

    return colours

def make_sample_figure(
    real_e,
    sampled_e,
    node_mask,
    num_graphs: int = 2,
    real_x=None,
    sampled_x=None,
    show_node_labels: bool = True,
):
    num_graphs = min(num_graphs, real_e.size(0), sampled_e.size(0), node_mask.size(0))
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_graphs,
        figsize=(3.5 * num_graphs, 6.0),
        squeeze=False,
    )

    for graph_idx in range(num_graphs):
        real_feats, real_graph = graph_from_adjacency(
            real_x[graph_idx] if real_x is not None else None,
            real_e[graph_idx],
            node_mask[graph_idx],
        )
        sampled_feats, sampled_graph = graph_from_adjacency(
            sampled_x[graph_idx] if sampled_x is not None else None,
            sampled_e[graph_idx],
            node_mask[graph_idx],
        )

        real_pos = nx.spring_layout(real_graph, seed=42)
        sampled_pos = nx.spring_layout(sampled_graph, seed=42)

        ax = axes[0, graph_idx]
        real_labels = None
        if show_node_labels and real_feats is not None:
            real_labels = {
                node_idx: str(int(node_class))
                for node_idx, node_class in enumerate(real_feats.long().flatten().tolist())
            }
        nx.draw_networkx(
            real_graph,
            pos=real_pos,
            node_color=node_colours_from_features(real_feats, real_graph.number_of_nodes(), cmap),
            ax=ax,
            node_size=45,
            labels=real_labels,
            with_labels=real_labels is not None,
            font_size=6,
            font_color="black",
            width=0.7,
            alpha=0.8,
        )
        ax.set_title(f"Real {graph_idx} | E={real_graph.number_of_edges()}")
        ax.set_axis_off()

        ax = axes[1, graph_idx]
        sampled_labels = None
        if show_node_labels and sampled_feats is not None:
            sampled_labels = {
                node_idx: str(int(node_class))
                for node_idx, node_class in enumerate(sampled_feats.long().flatten().tolist())
            }
        nx.draw_networkx(
            sampled_graph,
            pos=sampled_pos,
            node_color=node_colours_from_features(sampled_feats, sampled_graph.number_of_nodes(), cmap),
            ax=ax,
            node_size=45,
            labels=sampled_labels,
            with_labels=sampled_labels is not None,
            font_size=6,
            font_color="black",
            width=0.7,
            alpha=0.8,
        )
        ax.set_title(f"Sampled {graph_idx} | E={sampled_graph.number_of_edges()}")
        ax.set_axis_off()

    fig.suptitle("Real vs sampled graphs during training", fontsize=14)
    fig.tight_layout()
    return fig

def node_label_distribution(
    labels,
    node_mask,
    num_classes: int,
):
    """
    Compute the node-label distribution across a batch of graphs.

    Args:
        labels:
            NumPy array with shape [batch_size, num_nodes], or one-hot
            labels with shape [batch_size, num_nodes, num_classes].

        node_mask:
            NumPy array with shape [batch_size, num_nodes].

        num_classes:
            Number of node classes.

    Returns:
        NumPy array with shape [num_classes].
    """
    labels = np.asarray(labels)
    node_mask = np.asarray(node_mask, dtype=bool)

    if labels.ndim == 3:
        labels = labels.argmax(axis=-1)

    valid_labels = labels[node_mask].astype(np.int64)

    counts = np.bincount(
        valid_labels,
        minlength=num_classes,
    ).astype(np.float64)

    total = counts.sum()

    if total == 0:
        return np.zeros(
            num_classes,
            dtype=np.float64,
        )

    return counts / total


def total_variation_distance(
    real_distribution,
    generated_distribution,
) -> float:
    real_distribution = np.asarray(
        real_distribution,
        dtype=np.float64,
    )

    generated_distribution = np.asarray(
        generated_distribution,
        dtype=np.float64,
    )

    if real_distribution.shape != generated_distribution.shape:
        raise ValueError(
            "The real and generated distributions must have the same shape."
        )

    return float(
        0.5
        * np.abs(
            real_distribution - generated_distribution
        ).sum()
    )

def plot_node_label_distribution(
    real_distribution,
    generated_distribution,
    conditioning,
    save_path=None,
):
    real_distribution = np.asarray(real_distribution)
    generated_distribution = np.asarray(generated_distribution)

    if conditioning:
        experiment = "DDM"
    else:
        experiment = "DDMC"

    classes = np.arange(len(real_distribution))
    width = 0.4

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        classes - width / 2,
        real_distribution,
        width,
        label="Real",
    )

    ax.bar(
        classes + width / 2,
        generated_distribution,
        width,
        label="Generated",
    )

    ax.set_title(f"Node label distribution for {experiment}.", fontsize=18)
    ax.set_xlabel("Node label", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.set_xticks(classes)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig