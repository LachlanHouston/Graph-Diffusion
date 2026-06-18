from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
import wandb

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.dataset import get_data, construct_dataloader, batch_to_dense
from src.modeling.model import GaussianDiffusion, Linear_Denoiser, GAT_Denoiser, sample_timesteps
from src.modeling.utils import (
    log_samples,
    masked_upper_bce_with_logits,
    masked_upper_mse,
    symmetric_noise_like,
)

app = typer.Typer()

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
    use_node_mask: bool = False,
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
        "use_node_mask": use_node_mask,
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
            x = x.to(device).float()
            adj = adj.to(device).float()
            node_mask = node_mask.to(device)

            adj = torch.maximum(adj, adj.transpose(1, 2))

            if use_node_mask:
                training_node_mask = node_mask
            else:
                training_node_mask = None

            plot_node_mask = node_mask if use_node_mask else torch.ones(
                adj.shape[0],
                adj.shape[1],
                dtype=torch.bool,
                device=device,
            )

            latest_x = x.detach()
            latest_adj = adj.detach()
            latest_node_mask = plot_node_mask.detach()

            t = sample_timesteps(
                batch_size=adj.shape[0],
                num_steps=diffusion.num_steps,
                device=device,
            )

            noise = symmetric_noise_like(adj)
            adj_noised, noise = diffusion.q_sample(adj, t, noise=noise)

            pred = denoiser(x, adj_noised, t, training_node_mask)
            
            loss_noise = masked_upper_mse(pred, noise, training_node_mask)

            x0_pred = diffusion.predict_x0_from_noise(adj_noised, t, pred)
            loss_x0 = masked_upper_bce_with_logits(
                logits=x0_pred,
                target=adj,
                node_mask=training_node_mask,
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