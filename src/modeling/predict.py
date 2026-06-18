from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
from src.dataset import get_data, construct_dataloader, batch_to_dense
from src.modeling.model import GaussianDiffusion, Linear_Denoiser, GAT_Denoiser

app = typer.Typer()

def symmetrize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    adj: [B, N, N]
    """
    adj = 0.5 * (adj + adj.transpose(1, 2))
    return adj


def remove_self_loops(adj: torch.Tensor) -> torch.Tensor:
    """
    adj: [B, N, N]
    """
    B, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device).unsqueeze(0)
    return adj * (1.0 - eye)


def threshold_samples(samples: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    samples = symmetrize_adj(samples)
    adj = (samples > threshold).float()
    adj = remove_self_loops(adj)
    adj = symmetrize_adj(adj)
    adj = (adj > 0.5).float()
    return adj


def mask_adj(adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Remove padded nodes from adjacency matrices.
    """
    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    return adj * pair_mask.float()


def count_upper_edges(adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Count undirected non-self-loop edges for each graph using only valid nodes.

    adj:       [B, N, N]
    node_mask: [B, N]
    returns:   [B]
    """
    _, N, _ = adj.shape

    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=adj.device),
        diagonal=1,
    )

    valid_pairs = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    valid_upper = valid_pairs & upper_mask.unsqueeze(0)

    return (adj * valid_upper.float()).sum(dim=(1, 2))

def select_graphs_by_real_edge_fractions(
    real_adj: torch.Tensor,
    node_mask: torch.Tensor,
    max_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select graph indices whose real edge counts are closest to evenly spaced
    fractions of the maximum real edge count.

    Selection is based only on the real graphs. The sampled graphs are then
    plotted using the same selected indices.
    """
    real_edge_counts = count_upper_edges(real_adj, node_mask)
    num_available = real_edge_counts.numel()
    num_selected = min(max_graphs, num_available)

    max_edges = real_edge_counts.max().clamp(min=1.0)
    target_fractions = torch.arange(
        1,
        num_selected + 1,
        device=real_edge_counts.device,
        dtype=torch.float,
    ) / float(num_selected)

    target_fractions[0] = 0.0
    target_fractions[-1] = 1.0
    target_edges = target_fractions * max_edges

    selected_indices: list[int] = []
    available_indices = set(range(num_available))

    for target in target_edges:
        best_idx = min(
            available_indices,
            key=lambda idx: abs(float(real_edge_counts[idx].item()) - float(target.item())),
        )
        selected_indices.append(best_idx)
        available_indices.remove(best_idx)

    selected_indices_tensor = torch.tensor(
        selected_indices,
        device=real_edge_counts.device,
        dtype=torch.long,
    )

    return selected_indices_tensor, target_edges, real_edge_counts


def dense_adj_to_nx(adj: torch.Tensor, node_mask: torch.Tensor | None = None) -> nx.Graph:
    """
    Convert one dense adjacency matrix to a NetworkX graph.
    """
    adj = adj.detach().cpu()

    if node_mask is not None:
        node_mask = node_mask.detach().cpu().bool()
        valid_idx = torch.where(node_mask)[0]
        adj = adj[valid_idx][:, valid_idx]

    adj = (adj > 0.5).int().numpy()

    G = nx.from_numpy_array(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def graph_statistics(adj: torch.Tensor, node_mask: torch.Tensor) -> pd.DataFrame:
    """
    Compute graph-level statistics for a batch of dense adjacency matrices.

    adj:       [B, N, N]
    node_mask: [B, N]
    """
    rows = []

    B = adj.size(0)

    for i in range(B):
        G = dense_adj_to_nx(adj[i], node_mask[i])

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        if num_nodes > 1:
            density = nx.density(G)
        else:
            density = 0.0

        degrees = [deg for _, deg in G.degree()]
        avg_degree = float(sum(degrees) / max(len(degrees), 1))
        max_degree = float(max(degrees)) if degrees else 0.0

        if num_nodes > 0:
            avg_clustering = nx.average_clustering(G)
        else:
            avg_clustering = 0.0

        num_components = nx.number_connected_components(G) if num_nodes > 0 else 0
        largest_component = max((len(c) for c in nx.connected_components(G)), default=0)

        rows.append(
            {
                "graph_idx": i,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": density,
                "avg_degree": avg_degree,
                "max_degree": max_degree,
                "avg_clustering": avg_clustering,
                "num_components": num_components,
                "largest_component": largest_component,
            }
        )

    return pd.DataFrame(rows)


def summarize_stats(real_stats: pd.DataFrame, sample_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compare mean graph statistics between real and sampled graphs.
    """
    metric_cols = [
        "num_nodes",
        "num_edges",
        "density",
        "avg_degree",
        "max_degree",
        "avg_clustering",
        "num_components",
        "largest_component",
    ]

    rows = []

    for metric in metric_cols:
        real_mean = real_stats[metric].mean()
        sample_mean = sample_stats[metric].mean()

        rows.append(
            {
                "metric": metric,
                "real_mean": real_mean,
                "sample_mean": sample_mean,
                "absolute_difference": abs(real_mean - sample_mean),
            }
        )

    return pd.DataFrame(rows)


def plot_real_vs_sampled_batch(
    real_adj: torch.Tensor,
    sample_adj: torch.Tensor,
    node_mask: torch.Tensor,
    output_path: Path,
    max_graphs: int = 6,
    seed: int = 42,
):
    """
    Plot real and generated graphs side by side.

    Top row: real graphs
    Bottom row: sampled graphs
    """
    selected_indices, target_edges, real_edge_counts = select_graphs_by_real_edge_fractions(
        real_adj=real_adj,
        node_mask=node_mask,
        max_graphs=max_graphs,
    )

    num_graphs = selected_indices.numel()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_graphs,
        figsize=(3.4 * num_graphs, 6.5),
        squeeze=False,
    )

    for plot_idx, graph_idx_tensor in enumerate(selected_indices):
        graph_idx = int(graph_idx_tensor.item())

        real_G = dense_adj_to_nx(real_adj[graph_idx], node_mask[graph_idx])
        sample_G = dense_adj_to_nx(sample_adj[graph_idx], node_mask[graph_idx])

        pos_real = nx.spring_layout(real_G, seed=seed)
        pos_sample = nx.spring_layout(sample_G, seed=seed)

        ax = axes[0, plot_idx]
        nx.draw_networkx(
            real_G,
            pos=pos_real,
            ax=ax,
            node_size=45,
            with_labels=False,
            width=0.7,
            alpha=0.8,
        )
        real_edges = real_G.number_of_edges()
        ax.set_title(f"Real {graph_idx} | E={real_edges}")
        ax.set_axis_off()

        ax = axes[1, plot_idx]
        nx.draw_networkx(
            sample_G,
            pos=pos_sample,
            ax=ax,
            node_size=45,
            with_labels=False,
            width=0.7,
            alpha=0.8,
        )
        sampled_edges = sample_G.number_of_edges()
        ax.set_title(f"Sampled {graph_idx} | E={sampled_edges}")
        ax.set_axis_off()

    fig.suptitle(
        "Real vs sampled Cora subgraphs at increasing fractions of max real edge count",
        fontsize=14,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "cora",
    model_path: Path = MODELS_DIR / "model.pt",
    output_dir: Path = FIGURES_DIR / "sampling",
    batch_size: int = 32,
    max_nodes: int = 64,
    num_samples: int = 10_000,
    num_hops: int = 2,
    min_nodes: int = 8,
    threshold: float = 0.5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Performing inference for model...")

    data = get_data(data_path)

    loader = construct_dataloader(
        data,
        num_samples=num_samples,
        num_hops=num_hops,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        seed=0,
        batch_size=batch_size,
        shuffle=True,
    )

    diffusion = GaussianDiffusion(num_steps=1000).to(device)
    denoiser = GAT_Denoiser(
        max_nodes=64,
        feature_dim=1433,
        hidden_dim=128,
        time_emb_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    denoiser.load_state_dict(ckpt)
    denoiser.eval()

    batch = next(iter(loader))
    x, real_adj, node_mask = batch_to_dense(batch, max_nodes=max_nodes)

    x = x.to(device).float()
    real_adj = real_adj.to(device).float()
    node_mask = node_mask.to(device)

    real_adj = symmetrize_adj(real_adj)
    real_adj = remove_self_loops(real_adj)
    real_adj = mask_adj(real_adj, node_mask)
    real_adj = torch.maximum(real_adj, real_adj.transpose(1, 2))

    B, N, _ = real_adj.shape

    logger.info(f"x shape: {x.shape}")
    logger.info(f"real adj shape: {real_adj.shape}")
    logger.info(f"node mask shape: {node_mask.shape}")

    with torch.no_grad():
        samples = diffusion.sample(
            model=denoiser,
            x=x,
            adj_shape=[B, N, N],
            node_mask=node_mask,
            device=device,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in test_thresholds:
        sample_adj = threshold_samples(samples, threshold=threshold)
        sample_adj = mask_adj(sample_adj, node_mask)

        logger.debug(f"Real adjacency matrix: \n{real_adj.detach()[0, :3, :3]}")
        logger.debug(f"Sampled adjacency matrix: \n{sample_adj.detach()[0, :3, :3]}")

        real_stats = graph_statistics(real_adj, node_mask)
        sample_stats = graph_statistics(sample_adj, node_mask)
        summary = summarize_stats(real_stats, sample_stats)

        threshold_tag = str(threshold).replace(".", "_")
        real_stats_path = output_dir / f"real_graph_stats_threshold_{threshold_tag}.csv"
        sample_stats_path = output_dir / f"sampled_graph_stats_threshold_{threshold_tag}.csv"
        summary_path = output_dir / f"real_vs_sampled_summary_threshold_{threshold_tag}.csv"
        figure_path = output_dir / f"real_vs_sampled_batch_threshold_{threshold_tag}.png"

        plot_real_vs_sampled_batch(
            real_adj=real_adj,
            sample_adj=sample_adj,
            node_mask=node_mask,
            output_path=figure_path,
            max_graphs=6,
        )

        logger.info("Real vs sampled graph statistics:")
        logger.info(f"\n{summary}")

        logger.success(f"Saved real stats to {real_stats_path}")
        logger.success(f"Saved sampled stats to {sample_stats_path}")
        logger.success(f"Saved summary to {summary_path}")
        logger.success(f"Saved visualization to {figure_path}")
        logger.success("Inference complete.")

if __name__ == "__main__":
    app()