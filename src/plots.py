from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
import wandb
import typer
from sklearn.manifold import TSNE

from src.config import FIGURES_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.dataset_ego import (
    BASE_EGO_DATASET,
    EGO_DATASET,
    construct_ego_dataloader,
    get_ego_data,
)

from src.dataset_community import (
    COMMUNINTY_DATASET,
    construct_community_dataloader,
    get_community_data,
)

from src.data_utils import to_dense

app = typer.Typer()

CITESEER_LABEL_NAMES = {
    0: "Agents",
    1: "Artificial Intelligence",
    2: "Databases",
    3: "Information Retrieval",
    4: "Machine Learning",
    5: "Human-Computer Interaction",
}


LABEL_NAMES_BY_DATASET = {
    "Citeseer": CITESEER_LABEL_NAMES,
    "Community-small": {},
}

DATASET = "community"

if DATASET == "ego":
    DATASET_NAME = EGO_DATASET
    BASE_DATASET = BASE_EGO_DATASET

    get_data = get_ego_data
    construct_dataloader = construct_ego_dataloader

elif DATASET == "community":
    DATASET_NAME = COMMUNINTY_DATASET
    BASE_DATASET = "Community-small"

    get_data = get_community_data
    construct_dataloader = construct_community_dataloader

else:
    raise ValueError(f"Unknown dataset: {DATASET}")

def label_names():
    return LABEL_NAMES_BY_DATASET.get(BASE_DATASET, {})


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
        alpha=0.7,
        width=1.6,
    )

    nx.draw_networkx_nodes(
        nx_graph,
        pos=pos,
        ax=ax,
        node_size=160,
        node_color=node_colors,
        linewidths=1.0,
        edgecolors="black",
    )

    # ax.set_title(graph_title(graph, graph_idx), fontsize=16)
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


def dense_from_loader_output(
    batch: Batch | Data,
    min_nodes: int,
    max_nodes: int | None,
):
    if not hasattr(batch, "batch") or batch.batch is None:
        batch.batch = torch.zeros(
            batch.num_nodes,
            dtype=torch.long,
            device=batch.x.device,
        )

    return to_dense(
        x=batch.x,
        y=batch.y,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, "edge_attr", None),
        batch=batch.batch,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )

def add_label_legend(fig: plt.Figure, batch: Batch | Data) -> None:
    if not hasattr(batch, "y") or batch.y is None:
        return

    labels_present = sorted(batch.y.detach().cpu().unique().tolist())

    if len(labels_present) <= 1:
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
                markersize=20,
                label=f"{label_int}: {label_name}",
            )
        )

    fig.legend(
        handles=handles,
        title=f"{DATASET} paper class",
        loc="lower center",
        ncol=max(1, len(handles)),
        fontsize=16,
        title_fontsize=0,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def visualize_batch(
    batch: Batch,
    output_path: Path,
    max_graphs: int = 9,
    seed: int = 42,
    graph_type: str = "Sampled",
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

    nrows = min(3, len(graphs))
    ncols = (len(graphs) + nrows - 1) // nrows

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

    if DATASET == "ego":
        fig.suptitle(
            f"{graph_type} {DATASET} subgraphs colored by paper class",
            fontsize=64,
        )
    else:
        fig.suptitle(
            f"{graph_type} {DATASET} subgraphs",
            fontsize=64,
        )
    
    if labels_in_plotted_graphs:
        plotted_labels = torch.tensor(labels_in_plotted_graphs, dtype=torch.long)
        legend_batch = Batch(y=plotted_labels)
        add_label_legend(fig, legend_batch)
    else:
        add_label_legend(fig, batch)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
    batch_size: int = 32,
    max_graphs: int = 15,
    min_nodes: int = 4,
    max_nodes: int = 18,
    seed: int = 0,
):
    logger.info(f"Loading {DATASET_NAME} data...")

    output_path: Path = FIGURES_DIR / f"{DATASET_NAME}_subgraph_batch.png"

    if DATASET == "ego":
        DEFAULT_INPUT = RAW_DATA_DIR / BASE_DATASET
    else:
        DEFAULT_INPUT = PROCESSED_DATA_DIR / "Community" / "community_small.pkl"

    if DATASET == "ego":
        data = get_data(
            path=DEFAULT_INPUT,
            radius=1,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            num_graphs=200,
        )
    else:
        data = get_data(path=DEFAULT_INPUT)

    logger.info("Constructing dataloaders...")

    train_loader, _, _, _, _ = construct_dataloader(
        data=data,
        seed=seed,
        batch_size=batch_size,
        shuffle=True,
    )

    logger.info("Fetching one training batch...")

    batch = next(iter(train_loader))

    labels_present = sorted(
        batch.y.detach().cpu().unique().tolist()
    )

    label_text = ", ".join(
        f"{int(label)}="
        f"{label_names().get(int(label), f'Class {int(label)}')}"
        for label in labels_present
    )

    logger.info(
        f"Paper classes present in batch: {label_text}"
    )

    visualize_batch(
        batch=batch,
        output_path=output_path,
        max_graphs=max_graphs,
        seed=seed,
    )

    logger.success(
        f"Saved subgraph visualization to {output_path}"
    )

if __name__ == "__main__":
    app()
