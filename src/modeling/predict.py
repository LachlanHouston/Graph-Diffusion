from pathlib import Path
import os
import subprocess as sp
import copy

from tqdm import tqdm
import torch
import typer
from scipy.linalg import eigvalsh
import numpy as np
import networkx as nx

from src.config import ORCA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.dataset_discrete import get_data, construct_dataloader, to_dense, DATASET
from src.modeling.model_discrete import DiscreteDiffusion, TransformerDenoiser
from mmd import *

app = typer.Typer()

def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list):
    """
    Compute the MMD between the degree distributions of two sets of graphs.
    """

    sample_ref = []
    sample_pred = []

    # Remove empty generated graphs
    graph_pred_list = [
        G for G in graph_pred_list
        if G.number_of_nodes() > 0
    ]

    for G in graph_ref_list:
        sample_ref.append(degree_worker(G))

    for G in graph_pred_list:
        sample_pred.append(degree_worker(G))

    return compute_mmd(
        sample_ref,
        sample_pred,
        kernel=gaussian_emd,
    )


###############################################################################


def spectral_worker(G):
    """
    Compute the normalized Laplacian spectrum histogram.
    """

    eigs = eigvalsh(
        nx.normalized_laplacian_matrix(G).todense()
    )

    spectral_hist, _ = np.histogram(
        eigs,
        bins=200,
        range=(-1e-5, 2),
        density=False,
    )

    spectral_hist = spectral_hist.astype(np.float64)

    if spectral_hist.sum() > 0:
        spectral_hist /= spectral_hist.sum()

    return spectral_hist


def spectral_stats(graph_ref_list, graph_pred_list):
    """
    Compute MMD between graph spectra.
    """

    sample_ref = []
    sample_pred = []

    graph_pred_list = [
        G for G in graph_pred_list
        if G.number_of_nodes() > 0
    ]

    for G in graph_ref_list:
        sample_ref.append(spectral_worker(G))

    for G in graph_pred_list:
        sample_pred.append(spectral_worker(G))

    return compute_mmd(
        sample_ref,
        sample_pred,
        kernel=gaussian_emd,
    )


###############################################################################


def clustering_worker(G, bins=100):
    clustering_coeffs = list(nx.clustering(G).values())

    hist, _ = np.histogram(
        clustering_coeffs,
        bins=bins,
        range=(0.0, 1.0),
        density=False,
    )

    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100):
    """
    Compute MMD between clustering coefficient distributions.
    """

    sample_ref = []
    sample_pred = []

    graph_pred_list = [
        G for G in graph_pred_list
        if G.number_of_nodes() > 0
    ]

    for G in graph_ref_list:
        sample_ref.append(
            clustering_worker(G, bins)
        )

    for G in graph_pred_list:
        sample_pred.append(
            clustering_worker(G, bins)
        )

    return compute_mmd(
        sample_ref,
        sample_pred,
        kernel=gaussian_emd,
        sigma=1.0 / 10,
        distance_scaling=bins,
    )

###############################################################################
# ORCA / Orbit statistics
###############################################################################

motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}

COUNT_START_STR = "orbit counts: \n"


def edge_list_reindexed(G):
    """
    Reindex graph nodes to consecutive integers required by ORCA.
    """
    idx = 0
    id2idx = {}

    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []

    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))

    return edges


def orca(graph):
    """
    Run ORCA and return node orbit counts.
    """

    tmp_file_path = os.path.join(ORCA_DIR, "tmp.txt")

    with open(tmp_file_path, "w") as f:
        f.write(
            f"{graph.number_of_nodes()} {graph.number_of_edges()}\n"
        )

        for u, v in edge_list_reindexed(graph):
            f.write(f"{u} {v}\n")

    orca_path = os.path.join(ORCA_DIR, "orca")
    output = sp.check_output(
        [
            orca_path,
            "node",
            "4",
            tmp_file_path,
            "std",
        ]
    )

    output = output.decode("utf8").strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)

    output = output[idx:]

    node_orbit_counts = np.array(
        [
            list(map(int, line.strip().split(" ")))
            for line in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    """
    Compute MMD over graph orbit count statistics.
    """

    total_counts_ref = []
    total_counts_pred = []

    for G in graph_ref_list:

        try:
            orbit_counts = orca(G)
        except Exception:
            continue

        orbit_counts = (
            np.sum(orbit_counts, axis=0)
            / G.number_of_nodes()
        )

        total_counts_ref.append(orbit_counts)

    for G in graph_pred_list:

        try:
            orbit_counts = orca(G)
        except Exception:
            continue

        orbit_counts = (
            np.sum(orbit_counts, axis=0)
            / G.number_of_nodes()
        )

        total_counts_pred.append(orbit_counts)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    return compute_mmd(
        total_counts_ref,
        total_counts_pred,
        kernel=gaussian,
        is_hist=False,
        sigma=30.0,
    )


###############################################################################
# Graph conversion
###############################################################################


def adjs_to_graphs(adjs, node_flags=None):
    graph_list = []

    if torch.is_tensor(adjs):
        adjs = adjs.detach().cpu().numpy()

    if node_flags is not None and torch.is_tensor(node_flags):
        node_flags = node_flags.detach().cpu().numpy()

    for i, adj in enumerate(adjs):

        if node_flags is not None:
            keep = node_flags[i].astype(bool)
            adj = adj[np.ix_(keep, keep)]

        G = nx.from_numpy_array(adj)

        G.remove_edges_from(nx.selfloop_edges(G))

        if G.number_of_nodes() == 0:
            G.add_node(0)

        graph_list.append(G)

    return graph_list


###############################################################################
# Lobster evaluation
###############################################################################


def is_lobster_graph(G):
    """
    Check whether a graph is a lobster graph.
    """

    G = copy.deepcopy(G)

    if not nx.is_tree(G):
        return False

    leaves = [n for n, d in G.degree() if d == 1]
    G.remove_nodes_from(leaves)

    leaves = [n for n, d in G.degree() if d == 1]
    G.remove_nodes_from(leaves)

    num_nodes = len(G.nodes())

    degree_one = [d for _, d in G.degree() if d == 1]
    degree_two = [d for _, d in G.degree() if d == 2]

    if (
        sum(degree_one) == 2
        and sum(degree_two) == 2 * (num_nodes - 2)
    ):
        return True

    if (
        sum(degree_one) == 0
        and sum(degree_two) == 0
    ):
        return True

    return False


def eval_acc_lobster_graph(graph_list):

    graph_list = [copy.deepcopy(G) for G in graph_list]

    count = 0

    for G in graph_list:
        if is_lobster_graph(G):
            count += 1

    return count / len(graph_list)

###############################################################################
# Evaluation wrappers
###############################################################################

METHOD_NAME_TO_FUNC = {
    "degree": degree_stats,
    "cluster": clustering_stats,
    "orbit": orbit_stats_all,
    "spectral": spectral_stats,
}


def eval_graph_list(graph_ref_list, graph_pred_list, methods=None):
    """
    Evaluate two lists of NetworkX graphs.
    """

    if methods is None:
        methods = [
            "degree",
            "cluster",
            "orbit",
        ]

    results = {}

    for method in methods:
        results[method] = METHOD_NAME_TO_FUNC[method](
            graph_ref_list,
            graph_pred_list,
        )
    return results


def eval_torch_batch(
    ref_batch,
    pred_batch,
    node_mask=None,
    methods=None,
):
    """
    Evaluate batches of adjacency matrices stored as torch tensors.
    """

    graph_ref_list = adjs_to_graphs(
        ref_batch,
        node_mask,
    )

    graph_pred_list = adjs_to_graphs(
        pred_batch,
        node_mask,
    )

    return eval_graph_list(
        graph_ref_list,
        graph_pred_list,
        methods=methods,
    )

@app.command()
def main(
        batch_size: int = 32,
        max_nodes: int = 16,
        num_hops: int = 3,
        min_nodes: int = 4,
        diffusion_steps: int = 1000,
        hidden_dimension: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        time_emb_dim: int = 16,
        dropout: float = 0.1,
        seed: int = 42,
    ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = get_data(PROCESSED_DATA_DIR / DATASET)

    _, _, test_loader = construct_dataloader(
        data=data,
        num_hops=num_hops,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    x_classes = data.num_node_classes
    e_classes = data.num_edge_classes

    diffusion = DiscreteDiffusion(
        x_classes=x_classes,
        e_classes=e_classes,
        num_steps=diffusion_steps,
    ).to(device)
    
    denoiser = TransformerDenoiser(
        max_nodes=max_nodes,
        x_classes=x_classes,
        e_classes=e_classes,
        hidden_dim=hidden_dimension,
        time_emb_dim=time_emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        diffusion=diffusion,
    ).to(device)

    model_path = MODELS_DIR / "model_discrete.pt"
    checkpoint = torch.load(model_path, map_location=device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])

    batch_bar = tqdm(
                test_loader,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

    all_real = []
    all_generated = []
    all_node_masks = []
    for step, batch in enumerate(batch_bar, start=1):
        with torch.no_grad():
            denoiser.eval()

            batch = batch.to(device)

            _, x0, e0, node_mask = to_dense(
                            x=batch.x,
                            y=batch.y,
                            edge_index=batch.edge_index,
                            edge_attr=getattr(batch, "edge_attr", None),
                            batch=batch.batch,
                            min_nodes=min_nodes,
                            max_nodes=max_nodes,
                        )

            real_x = x0.to(device).long()
            real_e = e0.to(device).long()
            node_mask = node_mask.to(device)

            sampled, chain = diffusion.sample(
                            model=denoiser,
                            batch_size=real_x.size(0),
                            num_nodes=real_e.size(1),
                            keep_chain=False,
                            node_mask=node_mask,
                            device=device,
                        )
            
            sampled_x = sampled["X"].to(device).float()
            sampled_e = sampled["E"].to(device).float()

            # Convert edge labels to binary adjacency matrices.
            # Assumes edge class 0 = no edge.
            real_adj = (real_e > 0).float()
            sampled_adj = (sampled_e > 0).float()
            node_mask = (node_mask > 0).float()

            all_real.extend(real_adj)
            all_generated.extend(sampled_adj)
            all_node_masks.extend(node_mask)

    all_real = np.array(all_real)
    all_generated = np.array(all_generated)
    all_node_masks = np.array(all_node_masks)
    metrics = eval_torch_batch(
                    all_real,
                    all_generated,
                    node_mask=all_node_masks,
                    methods=[
                        "degree",
                        "cluster",
                        "spectral",
                        "orbit",
                    ],
                )
  
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

if __name__ == "__main__":
    app()