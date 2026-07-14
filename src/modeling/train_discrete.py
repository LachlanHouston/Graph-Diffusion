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
from src.modeling.utils import masked_upper_edge_cross_entropy, masked_node_cross_entropy
from src.plots import visualize_chain, log_discrete_samples, plot_tsne

app = typer.Typer()


def masked_multiclass_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    num_classes: int,
    prefix: str,
):
    pred = pred[node_mask].detach()
    target = target[node_mask].detach()

    if target.numel() == 0:
        return {
            f"{prefix}/accuracy": 0.0,
            f"{prefix}/macro_precision": 0.0,
            f"{prefix}/macro_recall": 0.0,
            f"{prefix}/macro_f1": 0.0,
        }

    eps = 1e-8
    accuracy = (pred == target).float().mean().item()

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for class_idx in range(num_classes):
        pred_is_class = pred == class_idx
        target_is_class = target == class_idx

        true_positive = (pred_is_class & target_is_class).float().sum()
        false_positive = (pred_is_class & ~target_is_class).float().sum()
        false_negative = (~pred_is_class & target_is_class).float().sum()
        support = target_is_class.float().sum()

        precision = true_positive / (true_positive + false_positive + eps)
        recall = true_positive / (true_positive + false_negative + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class_metrics[f"{prefix}/class_{class_idx}_precision"] = precision.item()
        per_class_metrics[f"{prefix}/class_{class_idx}_recall"] = recall.item()
        per_class_metrics[f"{prefix}/class_{class_idx}_f1"] = f1.item()
        per_class_metrics[f"{prefix}/class_{class_idx}_support"] = support.item()

    macro_precision = torch.stack(precisions).mean().item()
    macro_recall = torch.stack(recalls).mean().item()
    macro_f1 = torch.stack(f1s).mean().item()

    metrics = {
        f"{prefix}/accuracy": accuracy,
        f"{prefix}/macro_precision": macro_precision,
        f"{prefix}/macro_recall": macro_recall,
        f"{prefix}/macro_f1": macro_f1,
    }
    metrics.update(per_class_metrics)
    return metrics


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / DATASET,
    model_path: Path = MODELS_DIR / "model_discrete.pt",
    max_epochs: int = 5,
    batch_size: int = 32,
    max_nodes: int = 16,
    num_samples: int = 10_000,
    num_hops: int = 3,
    min_nodes: int = 4,
    lr: float = 0.00002,
    dropout: float = 0.1,
    x_loss_scale: float = 4.0,
    wandb_project: str = "graph-diffusion",
    wandb_entity: str | None = None,
    wandb_run_name: str = "local_discrete_run",
    wandb_mode: str = "online",
    wandb_log_interval: int = 10,
    sample_every: int = 1,
    sample_graphs: int = 6,
    visualize_gif_every: int = 10,
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
        hidden_dim=256,
        time_emb_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=dropout,
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

            f0, x0, e0, node_mask = to_dense(
                x=batch.x,
                y=batch.y,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch=batch.batch,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
            )

            f0 = f0.to(device).long()
            x0 = x0.to(device).long()
            e0 = e0.to(device).long()
            node_mask = node_mask.to(device)

            training_node_mask = node_mask if use_node_mask else None
            latest_f0 = f0.detach()
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
                node_features=f0,
                adj_noisy=noised["E_t"],
                t=t,
                node_mask=node_mask,
            )

            x_pred = pred["X"].argmax(dim=-1)
            valid_mask = node_mask if training_node_mask is None else training_node_mask
            node_metrics = masked_multiclass_metrics(
                pred=x_pred,
                target=x0,
                node_mask=valid_mask,
                num_classes=x_classes,
                prefix="node_features/",
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
                    "train/timestep_mean": t.float().mean().item(),
                    "train/edge_density": e0.float().mean().item(),
                    **node_metrics,
                    "train/epoch": epoch,
                }
                wandb.log(log_dict, step=global_step)

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
            with torch.no_grad():
                B = x0.size(0)

                t = torch.full(
                    (B,),
                    fill_value=diffusion.num_steps // 2,
                    dtype=torch.long,
                    device=x0.device,
                )

                noised = diffusion.q_sample(
                    x0=x0,
                    e0=e0,
                    t=t,
                    node_mask=node_mask,
                )

                h = denoiser.encode_nodes(
                    x=noised["X_t"],
                    node_features=latest_f0,
                    adj_noisy=noised["E_t"],
                    t=t,
                    node_mask=node_mask,
                )

                embeddings = h[node_mask]
                labels = x0[node_mask]

            plot_tsne(
                embeddings=embeddings,
                labels=labels,
                n_classes=diffusion.x_classes,
                output_dir=FIGURES_DIR,
                epoch=epoch,
                global_step=global_step,
                wandb_mode=wandb_mode,
            )

            sampled, chain = diffusion.sample(
                model=denoiser,
                node_features=latest_f0[:sample_graphs],
                batch_size=sample_graphs,
                num_nodes=latest_e0.size(1),
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
                    duration=60,
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