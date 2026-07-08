import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
import wandb

import math

def adjacency_mask(node_mask):
    """
    node_mask: [B, N]
    returns:   [B, N, N]
    """
    return node_mask.unsqueeze(1) & node_mask.unsqueeze(2)


def masked_upper_mse(pred, target, node_mask: Tensor | None = None):
    _, N, _ = pred.shape

    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=pred.device),
        diagonal=1,
    )

    if node_mask is None:
        return torch.nn.functional.mse_loss(pred, target)
    else:
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        mask = pair_mask & upper_mask.unsqueeze(0)

    pred = pred[mask]
    target = target[mask]

    return torch.nn.functional.mse_loss(pred, target)


def masked_upper_bce_with_logits(logits, target, node_mask: Tensor | None = None, pos_weight=None):
    _, N, _ = logits.shape

    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=logits.device),
        diagonal=1,
    )

    if node_mask is None:
        mask = upper_mask.unsqueeze(0).expand(logits.size(0), N, N)
    else:
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        mask = pair_mask & upper_mask.unsqueeze(0)

    logits = logits[mask]
    target = target[mask]

    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )


def symmetric_noise_like(adj):
    """
    Sample Gaussian noise with the same symmetry as an undirected adjacency matrix.
    """
    noise = torch.randn_like(adj)
    noise = torch.triu(noise, diagonal=1)
    noise = noise + noise.transpose(1, 2)
    return noise


def symmetrize(adj):
    return 0.5 * (adj + adj.transpose(1, 2))


def remove_diagonal(adj):
    _, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device).unsqueeze(0)
    return adj * (1.0 - eye)


def apply_node_mask(adj, node_mask):
    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    return adj * pair_mask.float()


def binarize_samples(samples, threshold):
    samples = symmetrize(samples)
    adj = (samples > threshold).float()
    adj = remove_diagonal(adj)
    adj = symmetrize(adj)
    return (adj > 0.5).float()



def graph_from_adjacency(adj, node_mask):
    adj = adj.detach().cpu()
    node_mask = node_mask.detach().cpu().bool()

    valid_nodes = torch.where(node_mask)[0]
    adj = adj[valid_nodes][:, valid_nodes]
    adj = (adj > 0.5).int().numpy()

    graph = nx.from_numpy_array(adj)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


# --- Graph statistics and evaluation functions ---

def graph_degree_histogram(graph: nx.Graph) -> torch.Tensor:
    degrees = [degree for _, degree in graph.degree()]
    if len(degrees) == 0:
        return torch.zeros(1)

    hist = torch.bincount(torch.tensor(degrees, dtype=torch.long)).float()
    return hist / hist.sum().clamp_min(1.0)


def graph_clustering_histogram(graph: nx.Graph, bins: int = 20) -> torch.Tensor:
    if graph.number_of_nodes() == 0:
        return torch.zeros(bins)

    coeffs = torch.tensor(
        list(nx.clustering(graph).values()),
        dtype=torch.float,
    )
    hist = torch.histc(coeffs, bins=bins, min=0.0, max=1.0)
    return hist / hist.sum().clamp_min(1.0)


def graph_orbit_features(graph: nx.Graph) -> torch.Tensor:
    if graph.number_of_nodes() == 0:
        return torch.zeros(6)

    triangles = nx.triangles(graph)
    features = []

    for node in graph.nodes():
        degree = graph.degree(node)
        triangle_count = triangles[node]
        wedge_count = max(math.comb(degree, 2) - triangle_count, 0) if degree >= 2 else 0
        three_star_count = math.comb(degree, 3) if degree >= 3 else 0
        leaf = 1 if degree == 1 else 0
        isolated = 1 if degree == 0 else 0

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

    features = torch.tensor(features, dtype=torch.float)
    return features.mean(dim=0)


def pad_stat_vectors(stats: list[torch.Tensor]) -> torch.Tensor:
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


def gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None):
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(x.size(0), y.size(0))

    dist = torch.cdist(x, y, p=2).pow(2)

    if sigma is None:
        all_dist = dist.detach().flatten()
        positive_dist = all_dist[all_dist > 0]
        sigma = positive_dist.median().sqrt().item() if positive_dist.numel() > 0 else 1.0

    gamma = 1.0 / (2.0 * max(sigma, 1e-6) ** 2)
    return torch.exp(-gamma * dist)


def mmd_from_stats(real_stats: list[torch.Tensor], sampled_stats: list[torch.Tensor]) -> float:
    x = pad_stat_vectors(real_stats)
    y = pad_stat_vectors(sampled_stats)

    if x.size(0) == 0 or y.size(0) == 0:
        return float("nan")

    max_dim = max(x.size(1), y.size(1))
    if x.size(1) < max_dim:
        x = torch.nn.functional.pad(x, (0, max_dim - x.size(1)))
    if y.size(1) < max_dim:
        y = torch.nn.functional.pad(y, (0, max_dim - y.size(1)))

    k_xx = gaussian_kernel_matrix(x, x)
    k_yy = gaussian_kernel_matrix(y, y)
    k_xy = gaussian_kernel_matrix(x, y)

    return (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).item()


def graph_adjacency_signature(graph: nx.Graph) -> bytes:
    nodes = sorted(graph.nodes())
    adj = nx.to_numpy_array(graph, nodelist=nodes, dtype="int8")
    return adj.tobytes()


def graph_uniqueness(graphs: list[nx.Graph]) -> float:
    if len(graphs) == 0:
        return float("nan")

    signatures = [graph_adjacency_signature(graph) for graph in graphs]
    return len(set(signatures)) / len(signatures)


def evaluate_generated_graphs(real_adj, sampled_adj, node_mask):
    real_graphs = [
        graph_from_adjacency(real_adj[i], node_mask[i])
        for i in range(real_adj.size(0))
    ]
    sampled_graphs = [
        graph_from_adjacency(sampled_adj[i], node_mask[i])
        for i in range(sampled_adj.size(0))
    ]

    real_degree_stats = [graph_degree_histogram(graph) for graph in real_graphs]
    sampled_degree_stats = [graph_degree_histogram(graph) for graph in sampled_graphs]

    real_cluster_stats = [graph_clustering_histogram(graph) for graph in real_graphs]
    sampled_cluster_stats = [graph_clustering_histogram(graph) for graph in sampled_graphs]

    real_orbit_stats = [graph_orbit_features(graph) for graph in real_graphs]
    sampled_orbit_stats = [graph_orbit_features(graph) for graph in sampled_graphs]

    return {
        "degree_mmd": mmd_from_stats(real_degree_stats, sampled_degree_stats),
        "cluster_mmd": mmd_from_stats(real_cluster_stats, sampled_cluster_stats),
        "orbit_mmd": mmd_from_stats(real_orbit_stats, sampled_orbit_stats),
        "uniqueness": graph_uniqueness(sampled_graphs),
        "real_edges_mean": sum(graph.number_of_edges() for graph in real_graphs) / max(len(real_graphs), 1),
        "sampled_edges_mean": sum(graph.number_of_edges() for graph in sampled_graphs) / max(len(sampled_graphs), 1),
    }


def make_sample_figure(real_adj, sampled_adj, node_mask, num_graphs=2):
    num_graphs = min(num_graphs, real_adj.size(0))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_graphs,
        figsize=(3.5 * num_graphs, 6.0),
        squeeze=False,
    )

    for graph_idx in range(num_graphs):
        real_graph = graph_from_adjacency(real_adj[graph_idx], node_mask[graph_idx])
        sampled_graph = graph_from_adjacency(sampled_adj[graph_idx], node_mask[graph_idx])

        real_pos = nx.spring_layout(real_graph, seed=42)
        sampled_pos = nx.spring_layout(sampled_graph, seed=42)

        ax = axes[0, graph_idx]
        nx.draw_networkx(
            real_graph,
            pos=real_pos,
            ax=ax,
            node_size=45,
            with_labels=False,
            width=0.7,
            alpha=0.8,
        )
        ax.set_title(f"Real {graph_idx} | E={real_graph.number_of_edges()}")
        ax.set_axis_off()

        ax = axes[1, graph_idx]
        nx.draw_networkx(
            sampled_graph,
            pos=sampled_pos,
            ax=ax,
            node_size=45,
            with_labels=False,
            width=0.7,
            alpha=0.8,
        )
        ax.set_title(f"Sampled {graph_idx} | E={sampled_graph.number_of_edges()}")
        ax.set_axis_off()

    fig.suptitle("Real vs sampled graphs during training", fontsize=14)
    fig.tight_layout()
    return fig


def sample_adjacency_for_plotting(
    denoiser,
    diffusion,
    x,
    real_adj,
    node_mask,
    device,
):
    B, N, _ = real_adj.shape

    # if hasattr(diffusion, "sample_joint"):
    #     try:
    #         samples = diffusion.sample_joint(
    #             model=denoiser,
    #             shape_X=tuple(x.shape),
    #             shape_E=(B, N, N),
    #             node_mask=node_mask,
    #             device=device,
    #         )
    #         return samples["E"]
    #     except Exception:
    #         pass

    samples = diffusion.sample(
        model=denoiser,
        x=x,
        adj_shape=[B, N, N],
        node_mask=node_mask,
        device=device,
    )

    if isinstance(samples, dict):
        return samples["E"]

    return samples


@torch.no_grad()
def log_samples(wandb_mode, denoiser, diffusion, x, real_adj, node_mask, epoch, global_step, threshold, num_graphs, device, figure_path):
    was_training = denoiser.training
    denoiser.eval()

    num_graphs = min(num_graphs, x.size(0))
    x = x[:num_graphs].to(device)
    real_adj = real_adj[:num_graphs].to(device)
    node_mask = node_mask[:num_graphs].to(device)

    samples = sample_adjacency_for_plotting(
        denoiser=denoiser,
        diffusion=diffusion,
        x=x,
        real_adj=real_adj,
        node_mask=node_mask,
        device=device,
    )

    sampled_adj = binarize_samples(samples, threshold=threshold)
    sampled_adj = apply_node_mask(sampled_adj, node_mask)

    metrics = evaluate_generated_graphs(
        real_adj=real_adj,
        sampled_adj=sampled_adj,
        node_mask=node_mask,
    )

    print(metrics)

    fig = make_sample_figure(
        real_adj=real_adj,
        sampled_adj=sampled_adj,
        node_mask=node_mask,
        num_graphs=num_graphs,
    )

    wandb.log(
        {
            "samples/real_vs_sampled": wandb.Image(
                fig,
                caption=f"Epoch {epoch}, threshold={threshold}",
            ),
            "samples/threshold": threshold,
            "eval/degree_mmd": metrics["degree_mmd"],
            "eval/cluster_mmd": metrics["cluster_mmd"],
            "eval/orbit_mmd": metrics["orbit_mmd"],
            "eval/uniqueness": metrics["uniqueness"],
            "eval/real_edges_mean": metrics["real_edges_mean"],
            "eval/sampled_edges_mean": metrics["sampled_edges_mean"],
        },
        step=global_step,
    )

    if wandb_mode != "online":
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    if was_training:
        denoiser.train()
