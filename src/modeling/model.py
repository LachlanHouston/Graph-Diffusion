from pathlib import Path

from loguru import logger
import typer

import math
import torch
from torch import nn
from torch import Tensor
import torch.distributions as td

app = typer.Typer()

def sample_timesteps(batch_size, num_steps, device):
    return torch.randint(
        low=0,
        high=num_steps,
        size=(batch_size,),
        device=device,
    )

def linear_beta_schedule(num_steps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_steps)

def cosine_beta_schedule(num_steps: int, s: float = 0.008, max_beta: float = 0.999):
    """
    Cosine beta schedule from Nichol & Dhariwal.

    Returns:
        betas: [num_steps]
    """
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps) / num_steps

    alpha_bars = torch.cos(((t + s) / (1.0 + s)) * torch.pi / 2.0) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]

    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = betas.clamp(min=1e-8, max=max_beta)

    return betas

def masked_mean_pool_x(x_dense, node_mask):
    mask = node_mask.unsqueeze(-1).float()
    x_sum = (x_dense * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return x_sum / denom

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: [B]
        returns: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=device) * -emb_scale
        )

        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb

class GaussianDiffusion(nn.Module):
    def __init__(self, num_steps: int = 1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.num_steps = num_steps

        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        # Avoid log(0) at t = 0.
        posterior_log_variance = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        )

        posterior_mean_coef1 = (
            betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        )

        posterior_mean_coef2 = (
            (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", posterior_log_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None):
        """
        x0: [B, N, N]
        t:  [B]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self._extract(self.alpha_bars, t, x0.shape)

        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

        return xt, noise

    def predict_x0_from_noise(self, xt: Tensor, t: Tensor, noise_pred: Tensor):
        alpha_bar_t = self._extract(self.alpha_bars, t, xt.shape)

        x0_pred = (
            xt - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
        ) / torch.sqrt(alpha_bar_t)

        return x0_pred

    def p_mean_variance(
        self,
        model: nn.Module,
        x: Tensor,
        xt: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ):
        noise_pred = model(x, xt, t, node_mask)

        x0_pred = self.predict_x0_from_noise(xt, t, noise_pred)
        x0_pred = x0_pred.clamp(0.0, 1.0)

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            x0_pred = x0_pred * pair_mask.float()

        eye = torch.eye(x0_pred.size(1), device=x0_pred.device).unsqueeze(0)
        x0_pred = x0_pred * (1.0 - eye)
        x0_pred = 0.5 * (x0_pred + x0_pred.transpose(1, 2))

        model_mean = (
            self._extract(self.posterior_mean_coef1, t, xt.shape) * x0_pred
            + self._extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )

        model_log_variance = self._extract(
            self.posterior_log_variance,
            t,
            xt.shape,
        )

        return model_mean, model_log_variance

    @torch.no_grad()
    def p_sample_step(
        self,
        model: nn.Module,
        x: Tensor,
        xt: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ):
        model_mean, model_log_variance = self.p_mean_variance(
            model=model,
            x=x,
            xt=xt,
            t=t,
            node_mask=node_mask,
        )

        noise = torch.randn_like(xt)
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.transpose(1, 2)

        nonzero_mask = (t != 0).float().view(-1, 1, 1)

        xt_prev = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            xt_prev = xt_prev * pair_mask.float()

        eye = torch.eye(xt_prev.size(1), device=xt_prev.device).unsqueeze(0)
        xt_prev = xt_prev * (1.0 - eye)

        xt_prev = 0.5 * (xt_prev + xt_prev.transpose(1, 2))

        return xt_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x: Tensor,
        adj_shape: tuple[int, int, int] | None = None,
        node_mask: Tensor | None = None,
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = next(model.parameters()).device

        x = x.to(device)

        if adj_shape is None:
            if x.dim() != 3:
                raise ValueError(
                    "Expected x with shape [B, N, F] when adj_shape is not provided."
                )
            adj_shape = (x.shape[0], x.shape[1], x.shape[1])

        xt = torch.randn(adj_shape, device=device)
        xt = torch.triu(xt, diagonal=1)
        xt = xt + xt.transpose(1, 2)

        if node_mask is not None:
            node_mask = node_mask.to(device)
            pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            xt = xt * pair_mask.float()

        eye = torch.eye(xt.size(1), device=device).unsqueeze(0)
        xt = xt * (1.0 - eye)

        for i in reversed(range(self.num_steps)):
            t = torch.full(
                size=(adj_shape[0],),
                fill_value=i,
                device=device,
                dtype=torch.long,
            )

            xt = self.p_sample_step(
                model=model,
                x=x,
                xt=xt,
                t=t,
                node_mask=node_mask,
            )

        return xt

    def _extract(self, arr: Tensor, timesteps: Tensor, broadcast_shape: torch.Size):
        res = arr[timesteps].float()

        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]

        return res.expand(broadcast_shape)

    
class Denoiser(nn.Module):
    """
    Simple MLP denoiser for dense adjacency matrices.

    Input:
        adj_noisy: [B, N, N]
        t:         [B]

    Output:
        pred_noise: [B, N, N]
    """

    def __init__(
        self,
        max_nodes: int = 64,
        encoder_dims: list[int] | None = None,
        latent_dim: int = 512,
        decoder_dims: list[int] | None = None,
        feature_dim: int = 512,
        time_emb_dim: int = 16,
        dropout: float = 0.0,
        force_symmetric_output: bool = True,
    ):
        super().__init__()

        if encoder_dims is None:
            encoder_dims = [1024]

        if decoder_dims is None:
            decoder_dims = [1024]

        self.max_nodes = max_nodes
        self.adj_dim = max_nodes * max_nodes
        self.time_emb_dim = time_emb_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.force_symmetric_output = force_symmetric_output

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        encoder_input_dim = self.adj_dim + latent_dim + time_emb_dim
        decoder_output_dim = self.adj_dim

        self.encoder = self._build_mlp(
            dims=[encoder_input_dim, *encoder_dims, latent_dim],
            dropout=dropout,
            activate_last=False,
        )

        self.decoder = self._build_mlp(
            dims=[latent_dim, *decoder_dims, decoder_output_dim],
            dropout=dropout,
            activate_last=False,
        )

        self.x_encoder = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    @staticmethod
    def _build_mlp(
        dims: list[int],
        dropout: float = 0.0,
        activate_last: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []

        for layer_idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = layer_idx == len(dims) - 2

            layers.append(nn.Linear(d_in, d_out))

            if activate_last or not is_last:
                layers.append(nn.SiLU())

                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def encode(self, x, adj_noisy: Tensor, t: Tensor, node_mask: Tensor | None = None) -> Tensor:
        """
        Encode noisy adjacency and timestep into a latent representation.

        adj_noisy: [B, N, N]
        t:         [B]
        node_mask: [B, N], optional
        returns:   [B, latent_dim]
        """
        B, N, _ = adj_noisy.shape

        if N != self.max_nodes:
            raise ValueError(
                f"Expected adj_noisy with N={self.max_nodes}, "
                f"but got N={N}."
            )

        if node_mask is not None:
            adj_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            adj_noisy = adj_noisy * adj_mask.float()

        adj_flat = adj_noisy.reshape(B, -1)
        t_emb = self.time_embedding(t)
        x_graph = masked_mean_pool_x(x, node_mask)
        x_emb = self.x_encoder(x_graph)

        h = torch.cat([adj_flat, x_emb, t_emb], dim=-1)
        z = self.encoder(h)

        return z

    def decode(self, z: Tensor, node_mask: Tensor | None = None, sample: bool = False) -> Tensor:
        """
        Decode latent representation into predicted adjacency noise.

        z:         [B, latent_dim]
        node_mask: [B, N], optional
        returns:   [B, N, N]
        """
        B = z.shape[0]

        pred = self.decoder(z)
        pred = pred.reshape(B, self.max_nodes, self.max_nodes)

        if self.force_symmetric_output:
            pred = 0.5 * (pred + pred.transpose(1, 2))

        if node_mask is not None:
            adj_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            pred = pred * adj_mask.float()

        return pred

    def forward(
        self,
        x: Tensor,
        adj_noisy: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """
        adj_noisy: [B, N, N]
        t:         [B]
        node_mask: [B, N], optional
        """
        z = self.encode(x, adj_noisy, t, node_mask)
        pred = self.decode(z, node_mask)

        return pred

@app.command()
def main():
    # ---- REPLACE THIS WITH YOUR OWN CODE ---- 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 32
    N = 64
    F = 1433

    adj = torch.randint(0, 2, (B, N, N), dtype=torch.float, device=device)

    # Make symmetric and remove self-loops.
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.transpose(1, 2)

    x_feat = torch.randn((B, N, F), dtype=torch.float, device=device)
    node_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    diffusion = GaussianDiffusion().to(device)

    t = sample_timesteps(
        batch_size=adj.size(0),
        num_steps=diffusion.num_steps,
        device=device,
    )

    adj_noised, noise = diffusion.q_sample(x0=adj, t=t)

    model = Denoiser(
        max_nodes=N,
        encoder_dims=[1024],
        latent_dim=512,
        decoder_dims=[1024],
        feature_dim=F,
        time_emb_dim=16,
        dropout=0.1,
    ).to(device)

    pred_noise = model(x_feat, adj_noised, t, node_mask)

    logger.debug(f"Adjacency Matrix: \n {adj[0, :3, :3]}")
    logger.debug(f"Noised Adjacency Matrix: \n {adj_noised[0, :3, :3]}")
    logger.debug(f"Noise: \n {noise[0, :3, :3]}")
    logger.debug(f"Predicted Noise: \n {pred_noise.detach()[0, :3, :3]}")

    sampled = diffusion.sample(
        model=model,
        x=x_feat,
        adj_shape=(B, N, N),
        node_mask=node_mask,
        device=device,
    )

    sampled = 0.5 * (sampled + sampled.transpose(1, 2))

    logger.debug(f"Sampled Adjacency Matrix: \n {sampled.detach()[0, :3, :3]}")

    logger.success("Model construction complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
