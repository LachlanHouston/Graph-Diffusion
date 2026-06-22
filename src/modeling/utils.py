import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
import wandb
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
import wandb

def adjacency_mask(node_mask):
    """
    node_mask: [B, N]
    returns:   [B, N, N]
    """
    return node_mask.unsqueeze(1) & node_mask.unsqueeze(2)


def masked_upper_mse(pred, target, node_mask: Tensor | None = None):
    _, N, _ = pred.shape

    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=pred.device),
        diagonal=1,
    )

    if node_mask is None:
        mask = upper_mask.unsqueeze(0).expand(pred.size(0), N, N)
    else:
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        mask = pair_mask & upper_mask.unsqueeze(0)

    pred = pred[mask]
    target = target[mask]

    return torch.nn.functional.mse_loss(pred, target)


def masked_upper_bce_with_logits(logits, target, node_mask: Tensor | None = None, pos_weight=None):
    _, N, _ = logits.shape

    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=logits.device),
        diagonal=1,
    )

    if node_mask is None:
        mask = upper_mask.unsqueeze(0).expand(logits.size(0), N, N)
    else:
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        mask = pair_mask & upper_mask.unsqueeze(0)

    logits = logits[mask]
    target = target[mask]

    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )


def symmetric_noise_like(adj):
    """
    Sample Gaussian noise with the same symmetry as an undirected adjacency matrix.
    """
    noise = torch.randn_like(adj)
    noise = torch.triu(noise, diagonal=1)
    noise = noise + noise.transpose(1, 2)
    return noise


def symmetrize(adj):
    return 0.5 * (adj + adj.transpose(1, 2))


def remove_diagonal(adj):
    _, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device).unsqueeze(0)
    return adj * (1.0 - eye)


def apply_node_mask(adj, node_mask):
    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    return adj * pair_mask.float()


def binarize_samples(samples, threshold):
    samples = symmetrize(samples)
    adj = (samples > threshold).float()
    adj = remove_diagonal(adj)
    adj = symmetrize(adj)
    return (adj > 0.5).float()


def graph_from_adjacency(adj, node_mask):
    adj = adj.detach().cpu()
    node_mask = node_mask.detach().cpu().bool()

    valid_nodes = torch.where(node_mask)[0]
    adj = adj[valid_nodes][:, valid_nodes]
    adj = (adj > 0.5).int().numpy()

    graph = nx.from_numpy_array(adj)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def make_sample_figure(real_adj, sampled_adj, node_mask, num_graphs=2):
    num_graphs = min(num_graphs, real_adj.size(0))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_graphs,
        figsize=(3.5 * num_graphs, 6.0),
        squeeze=False,
    )

    for graph_idx in range(num_graphs):
        real_graph = graph_from_adjacency(real_adj[graph_idx], node_mask[graph_idx])
        sampled_graph = graph_from_adjacency(sampled_adj[graph_idx], node_mask[graph_idx])

        real_pos = nx.spring_layout(real_graph, k=0.25, seed=42)
        sampled_pos = nx.spring_layout(sampled_graph, k=0.25, seed=42)

        ax = axes[0, graph_idx]
        nx.draw_networkx(
            real_graph,
            pos=real_pos,
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


@torch.no_grad()
def log_samples(wandb_mode, denoiser, diffusion, x, real_adj, node_mask, epoch, global_step, threshold, num_graphs, device, figure_path):
    was_training = denoiser.training
    denoiser.eval()

    num_graphs = min(num_graphs, x.size(0))
    x = x[:num_graphs].to(device)
    real_adj = real_adj[:num_graphs].to(device)
    node_mask = node_mask[:num_graphs].to(device)

    B, N, _ = real_adj.shape
    samples = diffusion.sample(
        model=denoiser,
        x=x,
        adj_shape=[B, N, N],
        device=device,
    )

    sampled_adj = binarize_samples(samples, threshold=threshold)
    sampled_adj = apply_node_mask(sampled_adj, node_mask)

    fig = make_sample_figure(
        real_adj=real_adj,
        sampled_adj=sampled_adj,
        node_mask=node_mask,
        num_graphs=num_graphs,
    )

    wandb.log(
        {
            "samples/real_vs_sampled": wandb.Image(
                fig,
                caption=f"Epoch {epoch}, threshold={threshold}",
            ),
            "samples/threshold": threshold,
        },
        step=global_step,
    )

    if wandb_mode != "online":
        plt.savefig(figure_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    if was_training:
        denoiser.train()
