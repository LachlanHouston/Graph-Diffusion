from pathlib import Path
import io
import math
import tempfile
from loguru import logger
import torch.nn.functional as F

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import Tensor
import wandb


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
    Convert node labels/features and adjacency into a masked NetworkX graph.

    x may be None. If present, it is masked to the valid nodes and returned as the
    first output. This is used for node-coloured visualisations.
    """
    adj = e.detach().cpu()
    node_mask = node_mask.detach().cpu().bool()
    valid_nodes = torch.where(node_mask)[0]

    feats = None
    if x is not None:
        feats = x.detach().cpu()[valid_nodes]

    adj = adj[valid_nodes][:, valid_nodes]
    adj = (adj > 0.5).int().numpy()

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


def evaluate_generated_graphs(real_e, sampled_e, node_mask):
    """Evaluate structural graph statistics, ignoring node colours/types."""
    num_graphs = min(real_e.size(0), sampled_e.size(0), node_mask.size(0))
    real_e = real_e[:num_graphs]
    sampled_e = sampled_e[:num_graphs]
    node_mask = node_mask[:num_graphs]

    real_graphs = [graph_only_from_adjacency(real_e[i], node_mask[i]) for i in range(num_graphs)]
    sampled_graphs = [graph_only_from_adjacency(sampled_e[i], node_mask[i]) for i in range(num_graphs)]

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


# -----------------------------------------------------------------------------
# Plotting and logging
# -----------------------------------------------------------------------------

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
        nx.draw_networkx(
            real_graph,
            pos=real_pos,
            node_color=node_colours_from_features(real_feats, real_graph.number_of_nodes(), cmap),
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
            node_color=node_colours_from_features(sampled_feats, sampled_graph.number_of_nodes(), cmap),
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


def sample_adjacency_for_plotting(denoiser, diffusion, x, real_adj, node_mask, device):
    B, N, _ = real_adj.shape

    samples = diffusion.sample(
        model=denoiser,
        x=x,
        adj_shape=[B, N, N],
        node_mask=node_mask,
        device=device,
    )

    if isinstance(samples, dict):
        return samples["E"]

    return ((samples + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def log_samples(
    wandb_mode,
    denoiser,
    diffusion,
    x,
    real_adj,
    node_mask,
    epoch,
    global_step,
    threshold,
    num_graphs,
    device,
    figure_path,
):
    """Legacy continuous/Gaussian sample logger."""
    was_training = denoiser.training
    denoiser.eval()

    num_graphs = min(num_graphs, x.size(0), real_adj.size(0), node_mask.size(0))
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
        real_e=real_adj,
        sampled_e=sampled_adj,
        node_mask=node_mask,
    )

    fig = make_sample_figure(
        real_e=real_adj,
        sampled_e=sampled_adj,
        node_mask=node_mask,
        num_graphs=num_graphs,
    )

    if wandb_mode != "disabled":
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


def visualize_chain(
    chain,
    node_mask,
    gif_path: Path | None = None,
    duration: int = 20,
    wandb_mode: str = "disabled",
    global_step: int | None = None,
    wandb_key: str = "samples/sampling_chain",
):
    """Render a sampled discrete reverse chain as an in-memory GIF."""
    x_chain = chain["X_chain"][:, 0].detach().cpu()
    e_chain = chain["E_chain"][:, 0].detach().cpu()

    node_mask = node_mask[0].detach().cpu().bool()
    valid_nodes = torch.where(node_mask)[0]

    frames = []
    graph_pos = None
    cmap = plt.get_cmap("tab10")

    for i, (x_frame, e_frame) in enumerate(zip(x_chain, e_chain)):
        e_frame = e_frame[valid_nodes][:, valid_nodes]
        x_frame = x_frame[valid_nodes]

        graph = nx.from_numpy_array(e_frame.int().numpy())
        graph.remove_edges_from(nx.selfloop_edges(graph))

        if graph_pos is None:
            graph_pos = nx.spring_layout(graph, seed=42)

        node_colors = node_colours_from_features(x_frame, graph.number_of_nodes(), cmap)

        fig = plt.figure(figsize=(6, 6))
        nx.draw_networkx(
            graph,
            pos=graph_pos,
            node_color=node_colors,
            node_size=70,
            with_labels=False,
            width=0.7,
            alpha=0.9,
        )
        plt.title(f"Sampled graph at timestep: {i}")
        plt.axis("off")

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    if len(frames) == 0:
        raise ValueError("Cannot visualise an empty chain.")

    frames.extend([frames[-1]] * 10)
    gif_log_path = gif_path

    if gif_path is not None:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, duration=duration)
    elif wandb_mode != "disabled":
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
            gif_log_path = Path(tmp_file.name)
        imageio.mimsave(gif_log_path, frames, duration=duration)

    if wandb_mode != "disabled" and gif_log_path is not None:
        wandb.log(
            {wandb_key: wandb.Video(str(gif_log_path), format="gif")},
            step=global_step,
        )

    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, frames, format="GIF", duration=duration)
    gif_buffer.seek(0)
    return gif_buffer

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


@torch.no_grad()
def log_discrete_samples(
    samples,
    real,
    node_mask,
    epoch,
    global_step,
    wandb_mode,
    device,
    figure_path,
    num_graphs: int = 6,
):  
    num_graphs = min(num_graphs, real[1].size(0), samples["E"].size(0), node_mask.size(0))

    real_x = real[0][:num_graphs].to(device).float()
    real_e = real[1][:num_graphs].to(device).float()
    node_mask = node_mask[:num_graphs].to(device)

    sampled_x = samples["X"][:num_graphs].to(device).float()
    sampled_e = samples["E"][:num_graphs].to(device).float()

    metrics = evaluate_generated_graphs(
        real_e=real_e,
        sampled_e=sampled_e,
        node_mask=node_mask,
    )

    logger.info(f"Discrete sample metrics at epoch {epoch}: {metrics}")

    fig = make_sample_figure(
        real_x=real_x,
        real_e=real_e,
        sampled_x=sampled_x,
        sampled_e=sampled_e,
        node_mask=node_mask,
        num_graphs=num_graphs,
    )

    if wandb_mode != "disabled":
        wandb.log(
            {
                "samples/real_vs_sampled": wandb.Image(
                    fig,
                    caption=f"Epoch {epoch}",
                ),
                "eval/degree_mmd": metrics["degree_mmd"],
                "eval/cluster_mmd": metrics["cluster_mmd"],
                "eval/orbit_mmd": metrics["orbit_mmd"],
                "eval/uniqueness": metrics["uniqueness"],
                "eval/real_edges_mean": metrics["real_edges_mean"],
                "eval/sampled_edges_mean": metrics["sampled_edges_mean"],
            },
            step=global_step,
        )

    else:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")

    plt.close(fig)