from pathlib import Path
import pickle

import networkx as nx
import torch
import typer
from loguru import logger
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_utils import to_dense, estimate_marginal_distributions


COMMUNINTY_DATASET = "Community-small"

app = typer.Typer()


class CommunitySmallDataset(list):
    """
    List-like container for the Community-small graph collection.
    """

    def __init__(self, graphs):
        super().__init__(graphs)

        self.num_node_classes = 1
        self.num_node_features = 2
        self.num_edge_classes = 2


def get_community_data(
    path: Path = RAW_DATA_DIR
    / "Community-small"
    / "community_small.pkl",
):
    """
    Load the official Community-small NetworkX graph collection and
    convert it to PyTorch Geometric Data objects.

    Only load pickle files from a trusted source.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Community-small dataset was not found at:\n{path}\n\n"
            "Generate or download community_small.pkl first."
        )

    logger.info(f"Loading {COMMUNINTY_DATASET} from {path}.")

    with open(path, "rb") as file:
        nx_graphs = pickle.load(file)

    if not isinstance(nx_graphs, list):
        raise TypeError(
            "Expected community_small.pkl to contain a list "
            "of NetworkX graphs."
        )

    graphs = []

    for graph_index, nx_graph in enumerate(nx_graphs):
        if not isinstance(nx_graph, nx.Graph):
            raise TypeError(
                f"Graph {graph_index} has type {type(nx_graph)}, "
                "but a NetworkX Graph was expected."
            )

        nx_graph = nx.Graph(nx_graph)
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))

        nx_graph = nx.convert_node_labels_to_integers(
            nx_graph,
            ordering="sorted",
        )

        graph_data = from_networkx(nx_graph)

        # Dummy features: Community-small is topology-only.
        graph_data.x = torch.ones(
            (graph_data.num_nodes, 1),
            dtype=torch.float,
        )

        # Dummy node label: every node belongs to one node class.
        graph_data.y = torch.zeros(
            graph_data.num_nodes,
            dtype=torch.long,
        )

        graph_data.graph_id = torch.tensor(
            [graph_index],
            dtype=torch.long,
        )

        graphs.append(graph_data)

    if len(graphs) == 0:
        raise ValueError(
            "The Community-small pickle contained no graphs."
        )

    node_counts = torch.tensor(
        [graph.num_nodes for graph in graphs],
        dtype=torch.long,
    )

    edge_counts = torch.tensor(
        [
            graph.edge_index.size(1) // 2
            for graph in graphs
        ],
        dtype=torch.long,
    )

    logger.info(
        f"Loaded {len(graphs)} {COMMUNINTY_DATASET} graphs."
    )

    logger.info(
        f"Node counts: "
        f"min={node_counts.min().item()}, "
        f"max={node_counts.max().item()}, "
        f"mean={node_counts.float().mean().item():.2f}."
    )

    logger.info(
        f"Edge counts: "
        f"min={edge_counts.min().item()}, "
        f"max={edge_counts.max().item()}, "
        f"mean={edge_counts.float().mean().item():.2f}."
    )

    return CommunitySmallDataset(graphs)


def construct_community_dataloader(
    data,
    seed: int = 42,
    batch_size: int = 32,
    shuffle: bool = True,
):
    """
    Split Community-small into train, validation, and test loaders.
    """
    if len(data) == 0:
        raise ValueError(
            "The Community-small dataset contains no graphs."
        )

    generator = torch.Generator().manual_seed(seed)

    permutation = torch.randperm(
        len(data),
        generator=generator,
    ).tolist()

    shuffled_graphs = [
        data[index]
        for index in permutation
    ]

    num_graphs = len(shuffled_graphs)

    num_train = int(0.8 * num_graphs)
    num_val = int(0.1 * num_graphs)

    train_graphs = shuffled_graphs[:num_train]

    val_graphs = shuffled_graphs[
        num_train:num_train + num_val
    ]

    test_graphs = shuffled_graphs[
        num_train + num_val:
    ]

    if (
        len(train_graphs) == 0
        or len(val_graphs) == 0
        or len(test_graphs) == 0
    ):
        raise ValueError(
            "The split created an empty train, validation, "
            "or test set."
        )

    node_marginals, edge_marginals = estimate_marginal_distributions(train_graphs, num_node_classes=1)

    logger.info(
        f"{COMMUNINTY_DATASET} split: "
        f"train={len(train_graphs)}, "
        f"validation={len(val_graphs)}, "
        f"test={len(test_graphs)}, "
        f"node marginal={node_marginals} "
        f"edge marginal={edge_marginals}."
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, node_marginals, edge_marginals


@app.command()
def main(
    input_path: Path = (
        PROCESSED_DATA_DIR
        / "Community"
        / "community_small.pkl"
    ),
    batch_size: int = 32,
    seed: int = 42,
):
    data = get_community_data(path=input_path)

    train_loader, val_loader, test_loader = (
        construct_community_dataloader(
            data=data,
            seed=seed,
            batch_size=batch_size,
            shuffle=True,
        )
    )

    batch = next(iter(train_loader))

    node_features, node_labels, adjacency, node_mask = to_dense(
        x=batch.x,
        y=batch.y,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, "edge_attr", None),
        batch=batch.batch,
        min_nodes=1,
        max_nodes=20,
    )

    print("node_features:", node_features.shape)
    print("node_labels:", node_labels.shape)
    print("adjacency:", adjacency.shape)
    print("node_mask:", node_mask.shape)

    print("Number of graphs:", len(data))
    print("Node classes:", data.num_node_classes)
    print(
        "Node feature dimension:",
        data.num_node_features,
    )
    print("Edge classes:", data.num_edge_classes)

    print(
        f"Train graphs: {len(train_loader.dataset)}"
    )
    print(
        f"Validation graphs: {len(val_loader.dataset)}"
    )
    print(
        f"Test graphs: {len(test_loader.dataset)}"
    )


if __name__ == "__main__":
    app()