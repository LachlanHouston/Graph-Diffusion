from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import wandb
import typer
import imageio
import io
import tempfile
from sklearn.manifold import TSNE
import seaborn as sns

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
from src.dataset_discrete import get_data, to_dense, construct_dataloader, DATASET
from src.modeling.utils import graph_from_adjacency, evaluate_generated_graphs, binarize_samples, apply_node_mask

app = typer.Typer()

CORA_LABEL_NAMES = {
    0: "Theory",
    1: "Reinforcement Learning",
    2: "Genetic Algorithms",
    3: "Neural Networks",
    4: "Probabilistic Methods",
    5: "Case Based",
    6: "Rule Learning",
}

PUBMED_LABEL_NAMES = {
    0: "Diabetes Mellitus, Experimental",
    1: "Diabetes Mellitus, Type 1",
    2: "Diabetes Mellitus, Type 2",
}


LABEL_NAMES_BY_DATASET = {
    "Cora": CORA_LABEL_NAMES,
    "PubMed": PUBMED_LABEL_NAMES,
}


def label_names() -> dict[int, str]:
    return LABEL_NAMES_BY_DATASET.get(DATASET, {})


def label_color(label: int):
    cmap = plt.get_cmap("tab10")
    return cmap(int(label) % 10)


def graph_title(graph: Data, graph_idx: int) -> str:
    """Create a compact title for one sampled subgraph."""
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.size(1)

    # PyG often stores undirected graphs with both directions.
    approx_undirected_edges = num_edges // 2

    return f"Graph {graph_idx + 1} | nodes={num_nodes}, edges≈{approx_undirected_edges}"


def draw_subgraph(ax: plt.Axes, graph: Data, graph_idx: int, seed: int = 42) -> None:
    """Draw one PyG graph on a matplotlib axis."""
    graph = graph.cpu()

    nx_graph = to_networkx(
        graph,
        to_undirected=True,
        remove_self_loops=True,
    )

    pos = nx.spring_layout(nx_graph, k=0.25, seed=seed)

    node_colors = None
    if hasattr(graph, "y") and graph.y is not None:
        node_colors = [label_color(int(label)) for label in graph.y.cpu().tolist()]

    nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax,
        alpha=0.35,
        width=0.8,
    )

    nx.draw_networkx_nodes(
        nx_graph,
        pos=pos,
        ax=ax,
        node_size=55,
        node_color=node_colors,
        linewidths=0.4,
        edgecolors="black",
    )

    ax.set_title(graph_title(graph, graph_idx), fontsize=9)
    ax.set_axis_off()



def graphs_from_loader_output(batch: Batch | Data) -> list[Data]:
    """
    Convert a loader output into a list of graphs for visualization.

    Standard PyG DataLoader batches can be reconstructed with `to_data_list()`.
    ShaDowKHopSampler returns one sampled Data/Batch-like object that was not
    created by `Batch.from_data_list()`, so `to_data_list()` cannot be used.
    In that case, visualize the sampled object as one merged subgraph.
    """
    try:
        return batch.to_data_list()
    except (RuntimeError, AttributeError):
        graph = Data(
            x=batch.x,
            edge_index=batch.edge_index,
            y=batch.y if hasattr(batch, "y") else None,
            num_nodes=batch.num_nodes,
        )
        return [graph]


def crop_graph(data: Data, max_nodes: int) -> Data:
    if data.num_nodes <= max_nodes:
        return data

    keep_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.edge_index.device)
    keep_mask[:max_nodes] = True

    edge_mask = keep_mask[data.edge_index[0]] & keep_mask[data.edge_index[1]]
    edge_index = data.edge_index[:, edge_mask]

    y = data.y[:max_nodes] if hasattr(data, "y") and data.y is not None else None

    return Data(
        x=data.x[:max_nodes],
        edge_index=edge_index,
        y=y,
        num_nodes=max_nodes,
    )


def dense_from_loader_output(batch: Batch | Data, min_nodes: int, max_nodes: int):
    if hasattr(batch, "batch") and batch.batch is not None:
        return to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch, min_nodes=min_nodes, max_nodes=max_nodes)

    graph = Data(
        x=batch.x,
        edge_index=batch.edge_index,
        y=batch.y if hasattr(batch, "y") else None,
        num_nodes=batch.num_nodes,
    )
    graph = crop_graph(graph, max_nodes=max_nodes)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.x.device)

    return to_dense(graph.x, max_nodes=max_nodes, batch_size=1)


def dense_graph_from_adj(adj: torch.Tensor, node_mask: torch.Tensor) -> nx.Graph:
    valid_nodes = torch.where(node_mask.detach().cpu().bool())[0]
    adj = adj.detach().cpu()[valid_nodes][:, valid_nodes]
    adj = (adj > 0.5).int().numpy()

    graph = nx.from_numpy_array(adj)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def visualize_dense_batch(
    adj: torch.Tensor,
    node_mask: torch.Tensor,
    output_path: Path,
    max_graphs: int = 9,
    seed: int = 42,
) -> None:
    num_graphs = min(max_graphs, adj.size(0))

    ncols = min(3, num_graphs)
    nrows = (num_graphs + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for graph_idx in range(num_graphs):
        graph = dense_graph_from_adj(adj[graph_idx], node_mask[graph_idx])
        pos = nx.spring_layout(graph, k=0.25, seed=seed)

        ax = flat_axes[graph_idx]
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, alpha=0.35, width=0.8)
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            ax=ax,
            node_size=55,
            linewidths=0.4,
            edgecolors="black",
        )
        ax.set_title(
            f"Dense {graph_idx + 1} | nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}",
            fontsize=9,
        )
        ax.set_axis_off()

    for ax in flat_axes[num_graphs:]:
        ax.set_axis_off()

    fig.suptitle(f"{DATASET} after batch_to_dense", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def add_label_legend(fig: plt.Figure, batch: Batch | Data) -> None:
    if not hasattr(batch, "y") or batch.y is None:
        return

    labels_present = sorted(batch.y.detach().cpu().unique().tolist())
    names = label_names()

    handles = []
    for label in labels_present:
        label_int = int(label)
        label_name = names.get(label_int, f"Class {label_int}")

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=label_color(label_int),
                markeredgecolor="black",
                markersize=7,
                label=f"{label_int}: {label_name}",
            )
        )

    fig.legend(
        handles=handles,
        title=f"{DATASET} paper class",
        loc="lower center",
        ncol=max(1, len(handles)),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def visualize_batch(
    batch: Batch,
    output_path: Path,
    max_graphs: int = 9,
    seed: int = 42,
) -> None:
    """Visualize a PyG batch as a grid of sampled subgraphs."""
    graphs = graphs_from_loader_output(batch)
    graphs = graphs[:max_graphs]

    labels_in_plotted_graphs = []
    for graph in graphs:
        if hasattr(graph, "y") and graph.y is not None:
            labels_in_plotted_graphs.extend(graph.y.detach().cpu().tolist())

    if len(graphs) == 0:
        raise ValueError("Batch did not contain any graphs to visualize.")

    ncols = min(3, len(graphs))
    nrows = (len(graphs) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for graph_idx, graph in enumerate(graphs):
        draw_subgraph(flat_axes[graph_idx], graph, graph_idx=graph_idx, seed=seed)

    for ax in flat_axes[len(graphs):]:
        ax.set_axis_off()

    fig.suptitle(f"Sampled {DATASET} subgraphs colored by paper class", fontsize=14)
    if labels_in_plotted_graphs:
        plotted_labels = torch.tensor(labels_in_plotted_graphs, dtype=torch.long)
        legend_batch = Batch(y=plotted_labels)
        add_label_legend(fig, legend_batch)
    else:
        add_label_legend(fig, batch)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Train specific plotting functions
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

    return samples


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

    # sampled_adj = binarize_samples(samples, threshold=threshold)
    sampled_adj = apply_node_mask(samples, node_mask)

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

    fig = make_sample_figure(
        real_x=real_x,
        real_e=real_e,
        sampled_x=sampled_x,
        sampled_e=sampled_e,
        node_mask=node_mask,
        num_graphs=num_graphs,
    )

    if wandb_mode != "disabled":
        wandb_payload = {
            "samples/real_vs_sampled": wandb.Image(
                fig,
                caption=f"Epoch {epoch}",
            ),
        }

        summary_metrics = {
            "eval_summary/degree_mmd": metrics["degree_mmd"],
            "eval_summary/cluster_mmd": metrics["cluster_mmd"],
            "eval_summary/orbit_mmd": metrics["orbit_mmd"],
            "eval_summary/uniqueness": metrics["uniqueness"],
            "eval_summary/density_abs_error": abs(
                metrics["sampled_density_mean"]
                - metrics["real_density_mean"]
            ),
            "eval_summary/avg_degree_abs_error": abs(
                metrics["sampled_avg_degree_mean"]
                - metrics["real_avg_degree_mean"]
            ),
            "eval_summary/avg_clustering_abs_error": abs(
                metrics["sampled_avg_clustering_mean"]
                - metrics["real_avg_clustering_mean"]
            ),
            "eval_summary/num_components_abs_error": abs(
                metrics["sampled_num_components_mean"]
                - metrics["real_num_components_mean"]
            ),
            "eval_summary/largest_component_fraction_abs_error": abs(
                metrics["sampled_largest_component_fraction_mean"]
                - metrics["real_largest_component_fraction_mean"]
            ),
            "eval_summary/connected_fraction_abs_error": abs(
                metrics["sampled_connected_fraction"]
                - metrics["real_connected_fraction"]
            ),
        }

        wandb_payload.update(summary_metrics)

        metric_table = wandb.Table(
            columns=["metric", "value"],
            data=[
                [metric_name, float(metric_value)]
                for metric_name, metric_value in sorted(metrics.items())
            ],
        )

        wandb_payload["eval_details/all_metrics"] = metric_table

        wandb.log(
            wandb_payload,
            step=global_step,
        )

    else:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

def plot_tsne(
    embeddings,
    labels,
    output_dir: Path,
    epoch: int,
    global_step: int,
    wandb_mode: str = "disabled",
    perplexity: float = 40.0,
    wandb_key: str = "embeddings/t_sne",
):
    """
    Plot a t-SNE projection of learned node embeddings and optionally log it to Weights & Biases.

    Args:
        embeddings:
            Tensor or array with shape [num_nodes, hidden_dim].
        labels:
            Tensor or array with shape [num_nodes].
        n_classes:
            Number of node classes.
        output_dir:
            Directory where the figure is saved when W&B is disabled or offline.
        epoch:
            Current training epoch.
        global_step:
            Current optimizer/global step used for W&B logging.
        wandb_mode:
            W&B mode. Expected values are "online", "offline", or "disabled".
        perplexity:
            t-SNE perplexity. Must be smaller than the number of nodes.
        wandb_key:
            W&B metric key for the generated image.
    """
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()

    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels).reshape(-1)

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape [num_nodes, hidden_dim], "
            f"got {embeddings.shape}."
        )

    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "labels and embeddings must contain the same number of nodes, "
            f"got {labels.shape[0]} labels and {embeddings.shape[0]} embeddings."
        )

    num_samples = embeddings.shape[0]

    if num_samples < 3:
        raise ValueError("t-SNE requires at least three samples.")

    effective_perplexity = min(
        perplexity,
        max(2.0, float(num_samples - 1) / 3.0),
    )

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=effective_perplexity,
        random_state=0,
    )

    projected = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))

    class_ids = sorted(
        int(class_id)
        for class_id in np.unique(labels).tolist()
    )
    names = label_names()
    cmap = plt.get_cmap("tab10")

    point_colours = [
        cmap(int(class_id) % cmap.N)
        for class_id in labels
    ]

    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=point_colours,
        s=18,
        alpha=0.75,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=cmap(class_id % 10),
            markeredgecolor="none",
            markersize=7,
            label=names.get(class_id, f"Class {class_id}"),
        )
        for class_id in class_ids
    ]

    ax.legend(
        handles=legend_handles,
        title="Node class",
        loc="best",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"t-SNE of learned node representations — epoch {epoch}")

    fig.tight_layout()

    output_dir = Path(output_dir)
    out_path = output_dir / f"t_SNE_{epoch}.png"

    if wandb_mode != "disabled":
        wandb.log(
            {
                wandb_key: wandb.Image(
                    fig,
                    caption=(
                        f"Epoch {epoch}, samples={num_samples}, "
                        f"perplexity={effective_perplexity:.2f}"
                    ),
                ),
                "embeddings/t_sne_num_samples": num_samples,
                "embeddings/t_sne_perplexity": effective_perplexity,
            },
            step=global_step,
        )

    if wandb_mode != "online":
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=200,
            bbox_inches="tight",
        )

    plt.close(fig)
    return out_path

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / DATASET,
    output_path: Path = FIGURES_DIR / f"{DATASET}_subgraph_batch.png",
    dense_output_path: Path = FIGURES_DIR / f"{DATASET}_dense_batch.png",
    batch_size: int = 32,
    max_graphs: int = 16,
    num_samples: int = 10_000,
    num_hops: int = 3,
    max_nodes: int = 64,
    min_nodes: int = 8,
    seed: int = 0,
):
    logger.info(f"Loading {DATASET} data...")
    data = get_data(input_path)

    logger.info("Constructing subgraph dataloader...")

    loader = construct_dataloader(
        data=data,
        num_samples=num_samples,
        num_hops=num_hops,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    logger.info("Fetching one batch of sampled subgraphs...")
    batch = next(iter(loader))

    graphs = graphs_from_loader_output(batch)
    logger.info(f"Loader output contains {len(graphs)} visualizable graph object(s)")
    logger.info(f"Total nodes in loader output: {batch.num_nodes}")
    logger.info(f"Total directed edges in loader output: {batch.edge_index.size(1)}")

    labels_present = sorted(batch.y.detach().cpu().unique().tolist())
    label_text = ", ".join(
        f"{int(label)}={label_names().get(int(label), f'Class {int(label)}')}"
        for label in labels_present
    )
    logger.info(f"Paper classes present in batch: {label_text}")

    x_dense, adj_dense, node_mask = dense_from_loader_output(
        batch=batch,
        min_nodes=min_nodes,
        max_nodes=None,
    )

    logger.info(f"Dense x shape: {x_dense.shape}")
    logger.info(f"Dense adj shape: {adj_dense.shape}")
    logger.info(f"Dense mask shape: {node_mask.shape}")

    visualize_batch(
        batch=batch,
        output_path=output_path,
        max_graphs=max_graphs,
        seed=seed,
    )

    visualize_dense_batch(
        adj=adj_dense,
        node_mask=node_mask,
        output_path=dense_output_path,
        max_graphs=max_graphs,
        seed=seed,
    )

    logger.success(f"Saved subgraph visualization to {output_path}")
    logger.success(f"Saved dense visualization to {dense_output_path}")


if __name__ == "__main__":
    app()
