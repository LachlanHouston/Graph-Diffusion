from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
import wandb
import matplotlib.pyplot as plt
import networkx as nx

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.dataset import get_data, construct_dataloader, batch_to_dense
from src.modeling.model import GaussianDiffusion, Linear_Denoiser, GAT_Denoiser, sample_timesteps

app = typer.Typer()

def adjacency_mask(node_mask):
    """
    node_mask: [B, N]
    returns:   [B, N, N]
    """
    return node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

def masked_upper_mse(pred, target, node_mask):
    _, N = node_mask.shape

    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=node_mask.device),
        diagonal=1,
    )

    mask = pair_mask & upper_mask.unsqueeze(0)
    mask = mask.float()

    loss = (pred - target) ** 2
    loss = loss * mask

    return loss.sum() / mask.sum().clamp(min=1.0)

def masked_upper_bce_with_logits(logits, target, node_mask, pos_weight=None):
    _, N = node_mask.shape

    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    upper_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=node_mask.device),
        diagonal=1,
    )

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

        real_pos = nx.spring_layout(real_graph, seed=42)
        sampled_pos = nx.spring_layout(sampled_graph, seed=42)

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
def log_samples(denoiser, diffusion, x, real_adj, node_mask, epoch, global_step, threshold, num_graphs, device):
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
        node_mask=node_mask,
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
        step=epoch,
    )

    plt.close(fig)

    if was_training:
        denoiser.train()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    data_path: Path = PROCESSED_DATA_DIR / "cora",
    model_path: Path = MODELS_DIR / "model.pt",
    max_epochs: int = 10,
    batch_size: int = 32,
    max_nodes: int = 64,
    num_samples: int = 10_000,
    num_hops: int = 2,
    min_nodes: int = 8,
    lr: float = 1e-4,
    x0_scale: float = 0.25,
    wandb_project: str = "graph-diffusion",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "disabled",
    wandb_log_interval: int = 10,
    sample_every: int = 5,
    sample_graphs: int = 2,
    sample_threshold: float = 0.4,
    # -----------------------------------------
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = get_data(data_path)

    train_loader = construct_dataloader(data, 
                                        num_samples=num_samples, 
                                        num_hops=num_hops, 
                                        max_nodes=max_nodes, 
                                        min_nodes=min_nodes, 
                                        seed=0, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        )

    config = {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "device": device,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "max_nodes": max_nodes,
        "num_samples": num_samples,
        "num_hops": num_hops,
        "min_nodes": min_nodes,
        "lr": lr,
        "x0_loss_lambda": x0_scale,
        "diffusion_steps": 1000,
        "encoder_dims": [1024, 512],
        "latent_dim": 256,
        "decoder_dims": [1024, 512],
        "feature_dim": 1433,
        "time_emb_dim": 32,
        "dropout": 0.25,
        "optimizer": "Adam",
        "loss": "masked_upper_mse_epsilon_prediction",
        "sample_every": sample_every,
        "sample_graphs": sample_graphs,
        "sample_threshold": sample_threshold,
    }

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        mode=wandb_mode,
        config=config,
    )

    logger.info("Training some model...")

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

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    wandb.watch(denoiser, log="gradients", log_freq=max(1, wandb_log_interval * 10))
    global_step = 0

    epoch_bar = tqdm(
        range(1, max_epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        denoiser.train()
        total_loss = 0.0
        latest_x = None
        latest_adj = None
        latest_node_mask = None

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{max_epochs:03d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(batch_bar, start=1):
            batch = batch.to(device)

            x, adj, node_mask = batch_to_dense(batch, max_nodes=max_nodes, batch_size=batch_size)
            adj = adj.to(device).float()
            node_mask = node_mask.to(device)

            adj = torch.maximum(adj, adj.transpose(1, 2))

            latest_x = x.detach()
            latest_adj = adj.detach()
            latest_node_mask = node_mask.detach()

            t = sample_timesteps(
                batch_size=adj.shape[0],
                num_steps=diffusion.num_steps,
                device=device,
            )

            noise = symmetric_noise_like(adj)
            adj_noised, noise = diffusion.q_sample(adj, t, noise=noise)

            pred = denoiser(x, adj_noised, t, node_mask)
            loss_noise = masked_upper_mse(pred, noise, node_mask)

            x0_pred = diffusion.predict_x0_from_noise(adj_noised, t, pred)
            loss_x0 = masked_upper_bce_with_logits(
                logits=x0_pred,
                target=adj,
                node_mask=node_mask,
            )

            loss = loss_noise + x0_scale * loss_x0

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value
            running_loss = total_loss / step

            batch_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{running_loss:.4f}",
            )

            global_step += 1
            if global_step % wandb_log_interval == 0:
                wandb.log(
                    {
                        "train/batch_loss": loss_value,
                        "train/running_loss": running_loss,
                        "train/noise_loss": loss_noise,
                        "train/x0_loss": loss_x0,
                        "train/grad_norm": float(grad_norm),
                        "train/pred_noise_mean": pred.detach().mean().item(),
                        "train/pred_noise_std": pred.detach().std().item(),
                        "train/target_noise_mean": noise.detach().mean().item(),
                        "train/target_noise_std": noise.detach().std().item(),
                        "train/timestep_mean": t.float().mean().item(),
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        avg_loss = total_loss / len(train_loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        logger.info(f"Epoch {epoch:03d}/{max_epochs:03d} | loss={avg_loss:.4f}")
        wandb.log(
            {
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch,
            },
            step=global_step,
        )

        if (
            wandb_mode != "disabled"
            and sample_every > 0
            and epoch % sample_every == 0
            and latest_x is not None
            and latest_adj is not None
            and latest_node_mask is not None
        ):
            log_samples(
                denoiser=denoiser,
                diffusion=diffusion,
                x=latest_x,
                real_adj=latest_adj,
                node_mask=latest_node_mask,
                epoch=epoch,
                global_step=global_step,
                threshold=sample_threshold,
                num_graphs=sample_graphs,
                device=device,
            )

    logger.success("Modeling training complete.")

    torch.save(denoiser.state_dict(), model_path)
    wandb.save(str(model_path))
    logger.success(f"Saved model: {str(model_path)}")
    wandb.finish()

if __name__ == "__main__":
    app()