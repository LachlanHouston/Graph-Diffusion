from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader, ShaDowKHopSampler, LinkNeighborLoader
from torch_geometric.utils import to_networkx
import typer

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
from src.dataset import batch_to_dense, construct_dataloader, get_data, DATASET

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


def cora_label_color(label: int):
    """Return the fixed matplotlib color for a Cora class label."""
    cmap = plt.get_cmap("tab10")
    return cmap(int(label) % 4)


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
        node_colors = [cora_label_color(int(label)) for label in graph.y.cpu().tolist()]

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
        # node_color=node_colors,
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

def add_cora_label_legend(fig: plt.Figure, batch: Batch) -> None:
    """Add a legend explaining what the Cora node colors represent."""
    if not hasattr(batch, "y") or batch.y is None:
        return

    labels_present = sorted(batch.y.detach().cpu().unique().tolist())

    handles = []
    for label in labels_present:
        label_int = int(label)
        label_name = CORA_LABEL_NAMES.get(label_int, f"Class {label_int}")

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=cora_label_color(label_int),
                markeredgecolor="black",
                markersize=7,
                label=f"{label_int+1}: {label_name}",
            )
        )

    fig.legend(
        handles=handles,
        title="Cora paper class",
        loc="lower center",
        ncol=len(handles),
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

    fig.suptitle("Sampled Cora subgraphs colored by paper class", fontsize=14)
    if labels_in_plotted_graphs:
        plotted_labels = torch.tensor(labels_in_plotted_graphs, dtype=torch.long)
        legend_batch = Batch(y=plotted_labels)
        add_cora_label_legend(fig, legend_batch)
    else:
        add_cora_label_legend(fig, batch)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / DATASET,
    output_path: Path = FIGURES_DIR / f"{DATASET}_subgraph_batch.png",
    dense_output_path: Path = FIGURES_DIR / f"{DATASET}_dense_batch.png",
    batch_size: int = 9,
    max_graphs: int = 9,
    num_samples: int = 128,
    num_hops: int = 2,
    max_nodes: int = 32,
    min_nodes: int = 8,
    seed: int = 0,
):
    logger.info("Loading Cora data...")
    data = get_data(input_path)

    logger.info("Constructing subgraph dataloader...")

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[5],
        batch_size=batch_size,
        edge_label_index=data.edge_index,
    )

    logger.info("Fetching one batch of sampled subgraphs...")
    batch = next(iter(loader))

    graphs = graphs_from_loader_output(batch)
    logger.info(f"Loader output contains {len(graphs)} visualizable graph object(s)")
    logger.info(f"Total nodes in loader output: {batch.num_nodes}")
    logger.info(f"Total directed edges in loader output: {batch.edge_index.size(1)}")

    labels_present = sorted(batch.y.detach().cpu().unique().tolist())
    label_text = ", ".join(
        f"{int(label)}={PUBMED_LABEL_NAMES.get(int(label), f'Class {int(label)}')}"
        for label in labels_present
    )
    logger.info(f"Paper classes present in batch: {label_text}")

    visualize_batch(
        batch=batch,
        output_path=output_path,
        max_graphs=max_graphs,
        seed=seed,
    )

    logger.success(f"Saved subgraph visualization to {output_path}")


if __name__ == "__main__":
    app()
