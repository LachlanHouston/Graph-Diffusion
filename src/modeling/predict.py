from loguru import logger
import os
import subprocess as sp
import random

from tqdm import tqdm
import torch
import typer
from scipy.linalg import eigvalsh
import numpy as np
import networkx as nx

from src.config import ORCA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

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

from src.data_utils import to_dense, sampled_tensors_to_batch
from src.modeling.model_discrete import DiscreteDiffusion, TransformerDenoiser
from src.modeling.utils import make_sample_figure, node_label_distribution, total_variation_distance, plot_node_label_distribution
from src.plots import visualize_batch
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

def data_functions(dataset: str):
    if dataset == "ego":
        DATASET = "ego"
        DATASET_NAME = EGO_DATASET
        BASE_DATASET = BASE_EGO_DATASET

        get_data = get_ego_data
        construct_dataloader = construct_ego_dataloader
        max_nodes = 18

    elif dataset == "community":
        DATASET = "community"
        DATASET_NAME = COMMUNINTY_DATASET
        BASE_DATASET = "Community-small"

        get_data = get_community_data
        construct_dataloader = construct_community_dataloader
        max_nodes = 20

    return DATASET, DATASET_NAME, BASE_DATASET, get_data, construct_dataloader, max_nodes

@app.command()
def main(
        dataset: str = "ego",
        experiment: str = "Final",
        model_checkpoint: str = "ego-cond",
        use_conditioning: bool = True,
        batch_size: int = 32,
        min_nodes: int = 1,
        diffusion_steps: int = 1000,
        hidden_dimension: int = 128,
        feature_hidden_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        time_emb_dim: int = 16,
        dropout: float = 0.1,
        seed: int = 42,
    ):

    DATASET, DATASET_NAME, BASE_DATASET, get_data, construct_dataloader, max_nodes = data_functions(dataset=dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if DATASET == "ego":
        DEFAULT_INPUT = PROCESSED_DATA_DIR / DATASET_NAME
    else:
        DEFAULT_INPUT = PROCESSED_DATA_DIR / "Community" / "community_small.pkl"

    data = get_data(path=DEFAULT_INPUT)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _, _, test_loader, node_marginals, edge_marginals = construct_dataloader(
        data=data,
        seed=seed,
        batch_size=batch_size,
        shuffle=True,
    )

    x_classes = data.num_node_classes
    e_classes = data.num_edge_classes
    f_shape = data.num_node_features

    diffusion = DiscreteDiffusion(
        x_classes=x_classes,
        e_classes=e_classes,
        num_steps=diffusion_steps,
        node_marginals=node_marginals,
        edge_marginals=edge_marginals,
    ).to(device)
    
    denoiser = TransformerDenoiser(
        max_nodes=max_nodes,
        feature_dim=f_shape,
        x_classes=x_classes,
        e_classes=e_classes,
        hidden_dim=hidden_dimension,
        feature_hidden_dim=feature_hidden_dim,
        time_emb_dim=time_emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        diffusion=diffusion,
        use_conditioning=use_conditioning,
    ).to(device)

    model_path = MODELS_DIR / experiment / (model_checkpoint + ".pt")
    checkpoint = torch.load(model_path, map_location=device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])

    batch_bar = tqdm(
                test_loader,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

    all_real_e = []
    all_generated_e = []

    all_real_x = []
    all_generated_x = []

    all_node_masks = []
    for step, batch in enumerate(batch_bar, start=1):
        with torch.no_grad():
            denoiser.eval()

            batch = batch.to(device)

            f0, x0, e0, node_mask = to_dense(
                            x=batch.x,
                            y=batch.y,
                            edge_index=batch.edge_index,
                            edge_attr=getattr(batch, "edge_attr", None),
                            batch=batch.batch,
                            min_nodes=min_nodes,
                            max_nodes=max_nodes,
                        )

            f0 = f0.to(device).float()
            real_x = x0.to(device).long()
            real_e = e0.to(device).long()
            node_mask = node_mask.to(device)

            sampled, chain = diffusion.sample(
                model=denoiser,
                features=f0,
                batch_size=real_e.size(0),
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

            all_real_e.extend(real_adj)
            all_generated_e.extend(sampled_adj)

            all_real_x.extend(real_x)
            all_generated_x.extend(sampled_x)

            all_node_masks.extend(node_mask)

            if step == 1:
                real_batch_to_plot = batch.detach().cpu()
                sampled_batch_to_plot = sampled_tensors_to_batch(
                    sampled_x=sampled_x,
                    sampled_e=sampled_e,
                    node_mask=node_mask,
                )

    all_real_e = np.array(all_real_e)
    all_generated_e = np.array(all_generated_e)

    all_real_x = np.array(all_real_x)
    all_generated_x = np.array(all_generated_x)

    all_node_masks = np.array(all_node_masks)

    metrics = eval_torch_batch(
                    all_real_e,
                    all_generated_e,
                    node_mask=all_node_masks,
                    methods=[
                        "degree",
                        "cluster",
                        "spectral",
                        "orbit",
                    ],
                )

    logger.info(f"Experiment: {model_checkpoint} with conditioning = {use_conditioning}")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.6f}")

    logger.info("Calculating node label distributions...")

    real_distribution = node_label_distribution(
        all_real_x,
        all_node_masks,
        x_classes,
    )

    generated_distribution = node_label_distribution(
        all_generated_x,
        all_node_masks,
        x_classes,
    )

    tv = total_variation_distance(
        real_distribution,
        generated_distribution,
    )

    print(f"Node label TV distance: {tv:.4f}")

    figure_directory = FIGURES_DIR / experiment
    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_node_label_distribution(
        real_distribution,
        generated_distribution,
        conditioning=use_conditioning,
        save_path=figure_directory / f"{model_checkpoint}_node_label_distribution.png",
    )

    logger.info("Generating sample figures...")

    visualize_batch(
        batch=real_batch_to_plot,
        output_path=figure_directory
        / f"{model_checkpoint}_real.png",
        max_graphs=15,
        graph_type="Real",
    )

    visualize_batch(
        batch=sampled_batch_to_plot,
        output_path=figure_directory
        / f"{model_checkpoint}_sampled.png",
        max_graphs=15,
        graph_type="Sampled",
    )

if __name__ == "__main__":
    app()