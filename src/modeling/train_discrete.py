from pathlib import Path

from loguru import logger
import numpy as np
import random
from tqdm import tqdm
import typer

import torch
import torch.nn.functional as F
import wandb

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.dataset_discrete import (
    get_data,
    construct_dataloader,
    to_dense,
    DATASET,
)
from src.modeling.model_discrete import DiscreteDiffusion, TransformerDenoiser, sample_timesteps
from src.plots import make_sample_figure
from src.modeling.predict import eval_torch_batch
from src.modeling.utils import masked_upper_edge_cross_entropy, masked_node_cross_entropy

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
    lr: float = 1e-4,
    dropout: float = 0.1,
    x_loss_scale: float = 4.0,
    wandb_project: str = "graph-diffusion",
    wandb_entity: str | None = None,
    wandb_run_name: str = "local_discrete_run",
    wandb_mode: str = "online",
    wandb_log_interval: int = 10,
    seed: int = 42,
    val_seed: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = get_data(PROCESSED_DATA_DIR / DATASET)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=lr * 0.05,
    )

    wandb.watch(denoiser, log="gradients", log_freq=max(1, wandb_log_interval * 10))
    global_step = 0
    best_val_loss = 1e6

    epoch_bar = tqdm(
        range(1, max_epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        run_val_stats = False
        denoiser.train()
        train_loss_sum = 0.0
        train_x_loss_sum = 0.0
        train_e_loss_sum = 0.0
        num_train_graphs = 0

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
                num_steps=diffusion.T,
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

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss encountered at epoch={epoch}, "
                    f"step={step}: loss={loss.item()}"
                )

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                denoiser.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            current_batch_size = e0.size(0)

            train_loss_sum += loss.item() * current_batch_size
            train_x_loss_sum += loss_x.item() * current_batch_size
            train_e_loss_sum += loss_e.item() * current_batch_size
            num_train_graphs += current_batch_size

            running_loss = train_loss_sum / max(num_train_graphs, 1)
            running_x_loss = train_x_loss_sum / max(num_train_graphs, 1)
            running_e_loss = train_e_loss_sum / max(num_train_graphs, 1)

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{running_loss:.4f}",
            )

            global_step += 1
            if global_step % wandb_log_interval == 0:
                log_dict = {
                    "train/running_loss": running_loss,
                    "train/edge_loss": running_e_loss,
                    "train/x_loss": running_x_loss,
                    "train/weighted_x_loss": x_loss_scale * running_x_loss,
                    "train/epoch": epoch,
                }
                wandb.log(log_dict, step=global_step)

        avg_loss = train_loss_sum / max(num_train_graphs, 1)
        avg_train_x_loss = train_x_loss_sum / max(num_train_graphs, 1)
        avg_train_e_loss = train_e_loss_sum / max(num_train_graphs, 1)

        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        logger.info(
            f"Epoch {epoch:03d}/{max_epochs:03d} | "
            f"loss={avg_loss:.4f} | "
            f"x_loss={avg_train_x_loss:.4f} | "
            f"weighted_x_loss={x_loss_scale * avg_train_x_loss:.4f} | "
            f"e_loss={avg_train_e_loss:.4f}"
        )

        wandb.log(
            {
                "train/epoch_loss": avg_loss,
                "train/x_loss_epoch": avg_train_x_loss,
                "train/weighted_x_loss_epoch": x_loss_scale * avg_train_x_loss,
                "train/edge_loss_epoch": avg_train_e_loss,
                "train/epoch": epoch,
            },
            step=global_step,
        )

        denoiser.eval()

        val_loss_sum = 0.0
        val_x_loss_sum = 0.0
        val_e_loss_sum = 0.0
        num_val_graphs = 0

        val_batch_bar = tqdm(
            val_loader,
            desc=f"Validation {epoch:03d}/{max_epochs:03d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        validation_cuda_devices = (
            [torch.cuda.current_device()] if device == "cuda" else []
        )
        with torch.random.fork_rng(devices=validation_cuda_devices):
            torch.manual_seed(val_seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(val_seed)
            with torch.no_grad():
                for step, batch in enumerate(val_batch_bar, start=1):
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
                        batch_size=e0.size(0),
                        num_steps=diffusion.T,
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

                    current_batch_size = e0.size(0)

                    val_loss_sum += loss.item() * current_batch_size
                    val_x_loss_sum += loss_x.item() * current_batch_size
                    val_e_loss_sum += loss_e.item() * current_batch_size
                    num_val_graphs += current_batch_size

                    running_val_loss = val_loss_sum / num_val_graphs

                    val_batch_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg=f"{running_val_loss:.4f}",
                    )

        avg_val_loss = val_loss_sum / max(num_val_graphs, 1)
        avg_val_x_loss = val_x_loss_sum / max(num_val_graphs, 1)
        avg_val_e_loss = val_e_loss_sum / max(num_val_graphs, 1)

        epoch_bar.set_postfix(
            train_loss=f"{avg_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
        )

        logger.info(
            f"Epoch {epoch:03d}/{max_epochs:03d} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_x_loss={avg_val_x_loss:.4f} | "
            f"val_e_loss={avg_val_e_loss:.4f}"
        )

        wandb.log(
            {
                "validation/epoch_loss": avg_val_loss,
                "validation/x_loss": avg_val_x_loss,
                "validation/weighted_x_loss": x_loss_scale * avg_val_x_loss,
                "validation/edge_loss": avg_val_e_loss,
                "validation/epoch": epoch,
            },
            step=global_step,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            run_val_stats = True

            logger.info(
                f"New best validation loss at epoch {epoch:03d}: "
                f"{best_val_loss:.4f}. Saving checkpoint."
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": denoiser.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "avg_val_loss": avg_val_loss,
                "config": {
                    "max_nodes": max_nodes,
                    "diffusion_steps": diffusion_steps,
                    "hidden_dimension": hidden_dimension,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "time_emb_dim": time_emb_dim,
                    "dropout": dropout,
                    "x_loss_scale": x_loss_scale,
                },
            }

            torch.save(checkpoint, MODELS_DIR / model_prefix)

        scheduler.step()
        wandb.log(
            {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=global_step,
        )

        if run_val_stats:
            batch_bar = tqdm(
                val_loader,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            
            all_real = []
            all_generated = []
            all_node_masks = []

            validation_cuda_devices = (
                        [torch.cuda.current_device()] if device == "cuda" else []
            )
            with torch.random.fork_rng(devices=validation_cuda_devices):
                torch.manual_seed(val_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(val_seed)

            with torch.no_grad():
                denoiser.eval()
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
                        real_x = x0.to(device).long()
                        real_e = e0.to(device).long()
                        node_mask = node_mask.to(device)
            
                        sampled, _ = diffusion.sample(
                                        model=denoiser,
                                        batch_size=real_e.size(0),
                                        num_nodes=real_e.size(1),
                                        keep_chain=False,
                                        node_mask=node_mask,
                                        device=device,
                                    )

                        sampled_x = sampled["X"].to(device).float()
                        sampled_e = sampled["E"].to(device).float()
            
                        real_adj = (real_e > 0).float()
                        sampled_adj = (sampled_e > 0).float()
                        node_mask = (node_mask > 0).float()
            
                        all_real.append(real_adj.detach().cpu())
                        all_generated.append(sampled_adj.detach().cpu())
                        all_node_masks.append(node_mask.detach().cpu())
            
                all_real = torch.cat(all_real, dim=0)
                all_generated = torch.cat(all_generated, dim=0)
                all_node_masks = torch.cat(all_node_masks, dim=0)
                metrics = eval_torch_batch(
                                all_real,
                                all_generated,
                                node_mask=all_node_masks,
                                methods=[
                                    "degree",
                                    "cluster",
                                    "spectral",
                                ],
                            )
                
                graph_log = {
                    f"validation_graphs/{name}_mmd": value
                    for name, value in metrics.items()
                }
                graph_log["validation_graphs/epoch"] = epoch
                wandb.log(graph_log, step=global_step)

                fig = make_sample_figure(
                        real_x=real_x,
                        real_e=real_e,
                        sampled_x=sampled_x,
                        sampled_e=sampled_e,
                        node_mask=node_mask,
                        num_graphs=6,
                    )

                image = wandb.Image(
                    fig,
                    caption=f"Epoch {epoch}",
                )
                
                wandb.log({"validation_graphs/sampled": image})

    last_checkpoint = {
        "epoch": epoch,
        "model_state_dict": denoiser.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "max_nodes": max_nodes,
            "min_nodes": min_nodes,
            "num_hops": num_hops,
            "batch_size": batch_size,
            "diffusion_steps": diffusion_steps,
            "hidden_dimension": hidden_dimension,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "time_emb_dim": time_emb_dim,
            "dropout": dropout,
            "x_loss_scale": x_loss_scale,
            "lr": lr,
            "seed": seed,
            "val_seed": val_seed,
        },
    }

    torch.save(
        last_checkpoint,
        MODELS_DIR / f"last_{model_prefix}",
    )

    logger.success("Discrete diffusion training complete.")
    wandb.finish()


if __name__ == "__main__":
    app()