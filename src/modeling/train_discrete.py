from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
import wandb

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.dataset_discreet import get_data, construct_dataloader, to_dense, DATASET
from src.modeling.model_discreet import DiscreteDiffusion, TransformerDenoiser, sample_timesteps
from src.modeling.utils import visualize_chain, masked_node_cross_entropy, masked_upper_edge_cross_entropy, log_discrete_samples

app = typer.Typer()


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / DATASET,
    model_path: Path = MODELS_DIR / "model_discrete.pt",
    max_epochs: int = 10,
    batch_size: int = 64,
    max_nodes: int = 64,
    num_samples: int = 10_000,
    num_hops: int = 3,
    min_nodes: int = 3,
    lr: float = 0.0002,
    dropout: float = 0.1,
    x_loss_scale: float = 0.2,
    wandb_project: str = "graph-diffusion",
    wandb_entity: str | None = None,
    wandb_run_name: str = "local_discrete_run",
    wandb_mode: str = "online",
    wandb_log_interval: int = 10,
    sample_every: int = 1,
    sample_graphs: int = 6,
    visualize_gif_every: int = 5,
    use_node_mask: bool = True,
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

    x_classes = data.num_node_classes
    e_classes = data.num_edge_classes

    logger.info(
        f"Training discrete diffusion model with x_classes={x_classes}, e_classes={e_classes}."
    )

    diffusion = DiscreteDiffusion(
        x_classes=x_classes,
        e_classes=e_classes,
        num_steps=500,
    ).to(device)

    denoiser = TransformerDenoiser(
        max_nodes=max_nodes,
        x_classes=x_classes,
        e_classes=e_classes,
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
        latest_e0 = None
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

            x0, e0, node_mask = to_dense(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch=batch.batch,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
            )

            x0 = x0.to(device).long()
            e0 = e0.to(device).long()
            node_mask = node_mask.to(device)

            training_node_mask = node_mask if use_node_mask else None
            latest_x0 = x0.detach()
            latest_e0 = e0.detach()
            latest_node_mask = node_mask.detach()

            t = sample_timesteps(
                batch_size=e0.shape[0],
                num_steps=diffusion.num_steps,
                device=device,
            )

            noised = diffusion.q_sample(
                x0=x0,
                e0=e0,
                t=t,
                node_mask=training_node_mask,
            )

            pred = denoiser(
                x=noised["X_t"],
                adj_noisy=noised["E_t"],
                t=t,
                node_mask=training_node_mask,
            )

            loss_x = masked_node_cross_entropy(
                logits=pred["X"],
                target=x0,
                node_mask=training_node_mask,
            )

            loss_e = masked_upper_edge_cross_entropy(
                logits=pred["E"],
                target=e0,
                node_mask=training_node_mask,
            )

            loss = loss_e + x_loss_scale * loss_x

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
                        "train/edge_loss": loss_e.item(),
                        "train/x_loss": loss_x.item(),
                        "train/grad_norm": float(grad_norm),
                        "train/timestep_mean": t.float().mean().item(),
                        "train/edge_density": e0.float().mean().item(),
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        avg_loss = total_loss / max(len(train_loader), 1)
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
            and latest_e0 is not None
            and latest_node_mask is not None
        ):
            denoiser.eval()
            sampled, chain = diffusion.sample(
                model=denoiser,
                batch_size=sample_graphs,
                num_nodes=latest_e0.size(1),
                keep_chain=True,
                node_mask=latest_node_mask[:sample_graphs],
                device=device,
            )

            log_discrete_samples(
                samples=sampled,
                real=[latest_x0, latest_e0],
                node_mask=latest_node_mask,
                epoch=epoch,
                global_step=global_step,
                wandb_mode=wandb_mode,
                device=device,
                figure_path=FIGURES_DIR / f"discrete_train_output_{epoch}.png",
                num_graphs=sample_graphs,
            )

            if epoch % visualize_gif_every == 0:
                visualize_chain(
                    chain=chain,
                    node_mask=node_mask,
                    gif_path=FIGURES_DIR / "sampling_chain.gif",
                    duration=10,
                    wandb_mode=wandb_mode,
                    global_step=global_step,
                )
            

    logger.success("Discrete diffusion training complete.")

    torch.save(denoiser.state_dict(), model_path)
    wandb.save(str(model_path))
    logger.success(f"Saved model: {str(model_path)}")
    wandb.finish()


if __name__ == "__main__":
    app()