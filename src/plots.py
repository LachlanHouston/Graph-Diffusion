from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import typer

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
from src.dataset_discreet import get_data, to_dense, construct_dataloader, DATASET

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
