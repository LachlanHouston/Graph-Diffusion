from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch.utils.data import Dataset

DATASET = "PubMed"

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

def get_data(path: Path):
    dataset = Planetoid(
        root=path,
        name=DATASET,
        split="full",
        transform=T.NormalizeFeatures(),
    )

    data = dataset[0]

    print(f"Raw data statistics for: {DATASET}")
    print("-"*25)
    print(data)
    print("num_nodes:", data.num_nodes)
    print("num_edges:", data.num_edges)
    print("num_features:", data.num_features)
    print("num_classes:", dataset.num_classes)

    return data

class KHopSubgraphDataset(Dataset):
    def __init__(
        self,
        data: Data,
        num_samples: int = 10_000,
        num_hops: int = 2,
        max_nodes: int = 64,
        min_nodes: int = 8,
        seed: int = 0,
    ):
        self.data = data
        self.num_samples = num_samples
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes

        generator = torch.Generator()
        generator.manual_seed(seed)

        self.center_nodes = torch.randint(
            low=0,
            high=data.num_nodes,
            size=(num_samples,),
            generator=generator,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        center = int(self.center_nodes[idx])

        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            node_idx=center,
            num_hops=self.num_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )

        # Optional: keep subgraphs small for dense adjacency diffusion.
        if subset.numel() > self.max_nodes:
            perm = torch.randperm(subset.numel())[: self.max_nodes]
            subset = subset[perm]

            # Rebuild induced subgraph from the selected subset.
            subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
                node_idx=subset,
                num_hops=0,
                edge_index=self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes,
            )

        # Avoid extremely tiny samples
        if subset.numel() < self.min_nodes:
            return self.__getitem__((idx + 1) % len(self))

        x_sub = self.data.x[subset]
        y_sub = self.data.y[subset]

        return Data(
            x=x_sub,
            edge_index=edge_index_sub,
            y=y_sub,
            num_nodes=x_sub.size(0),
            center_node=mapping,
        )
    
def construct_dataloader(
    data,
    num_samples: int = 10_000,
    num_hops: int = 2,
    max_nodes: int = 64,
    min_nodes: int = 8,
    seed: int = 0,
    batch_size: int = 32,
    shuffle: bool = False,
):
    train_dataset = KHopSubgraphDataset(
        data=data,
        num_samples=num_samples,
        num_hops=num_hops,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        seed=seed,
    )

    logger.info(
        f"Using KHopSubgraphDataset with num_samples={num_samples}, "
        f"num_hops={num_hops}, batch_size={batch_size}, "
        f"min_nodes={min_nodes}, max_nodes={max_nodes}."
    )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    
def batch_to_dense(batch, max_nodes: int = 64, batch_size: int | None = None):
    if batch_size is None:
        batch_size = int(batch.batch.max().item()) + 1

    x_dense, node_mask = to_dense_batch(
        batch.x,
        batch.batch,
        max_num_nodes=max_nodes,
        batch_size=batch_size,
    )

    adj_dense = to_dense_adj(
        batch.edge_index,
        batch=batch.batch,
        max_num_nodes=max_nodes,
        batch_size=batch_size,
    )

    return x_dense, adj_dense, node_mask

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / DATASET,
    output_path: Path = PROCESSED_DATA_DIR / DATASET,
    # ----------------------------------------------
):
    data = get_data(output_path)
    loader = LinkNeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=4,
        # Use a batch size of 128 for sampling training nodes
        batch_size=32,
        edge_label_index=data.edge_index,
    )

    batch = next(iter(loader))

    print(batch)
    print(batch.x.shape)
    print(batch.edge_index.shape)

    x, adj, node_mask = batch_to_dense(batch, max_nodes=64, batch_size=1)

    print("x:", x.shape)
    print("adj:", adj.shape)
    print("mask:", node_mask.shape)

    num_batches = 0
    for i, _ in enumerate(loader):
        num_batches += 1
    print(num_batches)

    print("Done Testing!")

if __name__ == "__main__":
    app()
