from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
import wandb

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
from src.dataset import get_data, construct_dataloader, to_dense, DATASET
from src.modeling.model import GaussianDiffusion, TransformerDenoiser, sample_timesteps
from src.modeling.utils import (
    log_samples,
    masked_upper_mse,
    symmetric_noise_like,
)

app = typer.Typer()

def masked_node_mse(pred, target, node_mask=None):
    if node_mask is not None:
        pred = pred[node_mask]
        target = target[node_mask]
    return torch.nn.functional.mse_loss(pred, target)

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    data_path: Path = PROCESSED_DATA_DIR / DATASET,
    model_path: Path = MODELS_DIR / "model.pt",
    max_epochs: int = 10,
    batch_size: int = 64,
    max_nodes: int = 32,
    num_samples: int = 10_000,
    num_hops: int = 2,
    min_nodes: int = 3,
    lr: float = 0.0002,
    dropout: float = 0.1,
    x_loss_scale: float = 0.2,
    wandb_project: str = "graph-diffusion",
    wandb_entity: str | None = None,
    wandb_run_name: str = "local_mac_run",
    wandb_mode: str = "disabled",
    wandb_log_interval: int = 10,
    sample_every: int = 1,
    sample_graphs: int = 6,
    sample_threshold: float = 0.5,
    use_node_mask: bool = True,
    # -----------------------------------------
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = get_data(data_path)

    train_loader = construct_dataloader(
        data=data,
        num_samples=num_samples,
        num_hops=num_hops,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        batch_size=batch_size,
        shuffle=True,
    )

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        mode=wandb_mode,
    )

    logger.info("Training some model...")

    diffusion = GaussianDiffusion(num_steps=500).to(device)
    denoiser = TransformerDenoiser(
        max_nodes=max_nodes,
        feature_dim=data.num_features,
        hidden_dim=128,
        time_emb_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr, amsgrad=True)
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

            x, adj, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch, min_nodes=min_nodes, max_nodes=max_nodes)
            x = x.to(device).float()
            adj = adj.to(device).float()
            node_mask = node_mask.to(device)

            if use_node_mask:
                training_node_mask = node_mask
            else:
                training_node_mask = None

            latest_x = x.detach()
            latest_adj = adj.detach()
            latest_node_mask = node_mask.detach()

            t = sample_timesteps(
                batch_size=adj.shape[0],
                num_steps=diffusion.num_steps,
                device=device,
            )

            x_noise = torch.randn_like(x)
            adj_noise = symmetric_noise_like(adj)

            x_noised, _ = diffusion.q_sample(x, t, noise=x_noise)
            adj_noised, _ = diffusion.q_sample(adj, t, noise=adj_noise)

            pred = denoiser(x_noised, adj_noised, t, training_node_mask)
            pred_x = pred["X"]
            pred_adj = pred["E"]

            loss_adj = masked_upper_mse(
                pred=pred_adj,
                target=adj_noise,
                node_mask=training_node_mask,
            )

            loss_x = masked_node_mse(
                pred=pred_x,
                target=x_noise,
                node_mask=training_node_mask,
            )

            loss = loss_adj + x_loss_scale * loss_x

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
                        "train/adj_loss": loss_adj.item(),
                        "train/x_loss": loss_x.item(),
                        "train/grad_norm": float(grad_norm),
                        "train/pred_x_mean": pred_x.detach().mean().item(),
                        "train/pred_x_std": pred_x.detach().std().item(),
                        "train/target_x_mean": x.detach().mean().item(),
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
            sample_every > 0
            and epoch % sample_every == 0
            and latest_x is not None
            and latest_adj is not None
            and latest_node_mask is not None
        ):
            log_samples(
                wandb_mode=wandb_mode,
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
                figure_path=FIGURES_DIR / f"train_output_{epoch}.png"
            )

    logger.success("Modeling training complete.")

    torch.save(denoiser.state_dict(), model_path)
    wandb.save(str(model_path))
    logger.success(f"Saved model: {str(model_path)}")
    wandb.finish()

if __name__ == "__main__":
    app()