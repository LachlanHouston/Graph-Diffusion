from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
import torch.nn.functional as F
import wandb

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.dataset_discrete import (
    get_data,
    construct_dataloader,
    to_dense,
    DATASET,
)
from src.modeling.model_discrete import DiscreteDiffusion, TransformerDenoiser, sample_timesteps
from src.modeling.utils import masked_upper_edge_cross_entropy, masked_node_cross_entropy, masked_multiclass_metrics
from src.plots import visualize_chain, log_discrete_samples, plot_tsne

app = typer.Typer()

@app.command()
def main(
    model_prefix: Path = "model_discrete.pt",
    max_epochs: int = 5,
    batch_size: int = 32,
    max_nodes: int = 16,
    num_hops: int = 3,
    min_nodes: int = 4,
    diffusion_steps: int = 1000,
    hidden_dimension: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    time_emb_dim: int = 16,
    lr: float = 0.00002,
    dropout: float = 0.1,
    x_loss_scale: float = 4.0,
    wandb_project: str = "graph-diffusion",
    wandb_entity: str | None = None,
    wandb_run_name: str = "local_discrete_run",
    wandb_mode: str = "disabled",
    wandb_log_interval: int = 10,
    sample_every: int = 1,
    sample_graphs: int = 6,
    visualize_gif_every: int = 10,
    seed: int = 42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = get_data(PROCESSED_DATA_DIR / DATASET)

    train_loader, val_loader, test_loader = construct_dataloader(
        data=data,
        num_hops=num_hops,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        batch_size=batch_size,
        seed=seed,
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

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    wandb.watch(denoiser, log="gradients", log_freq=max(1, wandb_log_interval * 10))
    global_step = 0

    epoch_bar = tqdm(
        range(1, max_epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    validation_batch = None

    for epoch in epoch_bar:
        denoiser.train()
        total_loss = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{max_epochs:03d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(batch_bar, start=1):
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

            x0 = x0.to(device).long()
            e0 = e0.to(device).long()
            node_mask = node_mask.to(device)

            t = sample_timesteps(
                batch_size=e0.shape[0],
                num_steps=diffusion.num_steps,
                device=device,
            )

            noised = diffusion.q_sample(
                x0=x0,
                e0=e0,
                t=t,
                node_mask=node_mask,
            )

            pred = denoiser(
                x=noised["X_t"],
                adj_noisy=noised["E_t"],
                t=t,
                node_mask=node_mask,
            )

            loss_x = masked_node_cross_entropy(
                logits=pred["X"],
                target=x0,
                node_mask=node_mask,
            )

            loss_e = masked_upper_edge_cross_entropy(
                logits=pred["E"],
                target=e0,
                node_mask=node_mask,
            )

            loss = loss_e + x_loss_scale * loss_x

            optimizer.zero_grad()
            loss.backward()
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
                log_dict = {
                    "train/batch_loss": loss_value,
                    "train/running_loss": running_loss,
                    "train/edge_loss": loss_e.item(),
                    "train/x_loss": loss_x.item(),
                    "train/epoch": epoch,
                }
                wandb.log(log_dict, step=global_step)

        avg_loss = total_loss / max(len(train_loader), 1)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        logger.info(f"Epoch {epoch:03d}/{max_epochs:03d} | loss={avg_loss:.4f} | x_loss={x_loss_scale*loss_x:.4f} | e_loss={loss_e:.4f}")

        wandb.log(
            {
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch,
            },
            step=global_step,
        )

        # Get one batch from validation loader for sampling and visualization
        if validation_batch is None:
            try:
                validation_batch = next(iter(val_loader))
                validation_batch = to_dense(
                    x=validation_batch.x,
                    y=validation_batch.y,
                    edge_index=validation_batch.edge_index,
                    edge_attr=getattr(validation_batch, "edge_attr", None),
                    batch=validation_batch.batch,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                )
                validation_batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in validation_batch)
            except StopIteration:
                validation_batch = None

        if sample_every > 0 and epoch % sample_every == 0:
            denoiser.eval()

            if validation_batch is None:
                raise RuntimeError("Validation loader did not produce any batches.")

            _, val_x0, val_e0, val_node_mask = validation_batch
            validation_sample_count = min(sample_graphs, val_x0.size(0))

            with torch.no_grad():
                t = torch.full(
                    (val_x0.size(0),),
                    fill_value=diffusion.num_steps // 2,
                    dtype=torch.long,
                    device=val_x0.device,
                )

                noised = diffusion.q_sample(
                    x0=val_x0,
                    e0=val_e0,
                    t=t,
                    node_mask=val_node_mask,
                )

                h = denoiser.encode_nodes(
                    x=noised["X_t"],
                    adj_noisy=noised["E_t"],
                    t=t,
                    node_mask=val_node_mask,
                )

                embeddings = h[val_node_mask]
                labels = val_x0[val_node_mask]

            plot_tsne(
                embeddings=embeddings,
                labels=labels,
                output_dir=FIGURES_DIR,
                epoch=epoch,
                global_step=global_step,
                wandb_mode=wandb_mode,
            )

            sampled, chain = diffusion.sample(
                model=denoiser,
                batch_size=validation_sample_count,
                num_nodes=val_e0.size(1),
                keep_chain=True,
                node_mask=val_node_mask[:validation_sample_count],
                device=device,
            )

            log_discrete_samples(
                samples=sampled,
                real=[val_x0, val_e0],
                node_mask=val_node_mask,
                epoch=epoch,
                global_step=global_step,
                wandb_mode=wandb_mode,
                device=device,
                figure_path=FIGURES_DIR / f"discrete_validation_output_{epoch}.png",
                num_graphs=validation_sample_count,
            )

            if visualize_gif_every > 0 and epoch % visualize_gif_every == 0:
                visualize_chain(
                    chain=chain,
                    node_mask=val_node_mask[:validation_sample_count],
                    gif_path=FIGURES_DIR / "sampling_chain.gif",
                    duration=60,
                    wandb_mode=wandb_mode,
                    global_step=global_step,
                )


    logger.success("Discrete diffusion training complete.")

    torch.save(denoiser.state_dict(), MODELS_DIR / model_prefix)
    wandb.save(str(MODELS_DIR / model_prefix))
    logger.success(f"Saved model: {str(MODELS_DIR / model_prefix)}")
    wandb.finish()


if __name__ == "__main__":
    app()