from pathlib import Path

from loguru import logger
import typer

import math
import torch
from torch import nn
from torch import Tensor

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
        model_out = model(x, xt, t, node_mask)
        noise_pred = model_out["E"] if isinstance(model_out, dict) else model_out

        x0_pred = self.predict_x0_from_noise(xt, t, noise_pred)

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
    

class DenseGraphAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(1, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: Tensor,
        adj_noisy: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        B, N, H = h.shape

        q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        edge_bias = self.edge_bias(adj_noisy.unsqueeze(-1)).permute(0, 3, 1, 2)
        attn_scores = attn_scores + edge_bias

        if node_mask is not None:
            key_mask = node_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(~key_mask, -1e9)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        h_attn = torch.matmul(attn, v)
        h_attn = h_attn.transpose(1, 2).contiguous().view(B, N, H)
        h_attn = self.out_proj(h_attn)

        h = self.norm_attn(h + self.dropout(h_attn))
        h = self.norm_ffn(h + self.dropout(self.ffn(h)))

        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).float()

        return h


class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        max_nodes: int = 64,
        feature_dim: int = 1433,
        hidden_dim: int = 128,
        time_emb_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        force_symmetric_output: bool = True,
        x_out_dim: int | None = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.force_symmetric_output = force_symmetric_output
        self.x_out_dim = feature_dim if x_out_dim is None else x_out_dim

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        self.node_input_proj = nn.Sequential(
            nn.Linear(feature_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.attn_layers = nn.ModuleList(
            [
                DenseGraphAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        edge_input_dim = 3 * hidden_dim + time_emb_dim + 1
        self.out_E = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.out_X = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.x_out_dim),
        )

    def encode_nodes(
        self,
        x: Tensor,
        adj_noisy: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        B, N, _ = x.shape

        t_emb = self.time_embedding(t)
        t_node = t_emb[:, None, :].expand(B, N, self.time_emb_dim)

        h = torch.cat([x, t_node], dim=-1)
        h = self.node_input_proj(h)

        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).float()

        for attn_layer in self.attn_layers:
            h = attn_layer(
                h=h,
                adj_noisy=adj_noisy,
                node_mask=node_mask,
            )

        return h

    def decode_E(
        self,
        h: Tensor,
        adj_noisy: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        B, N, H = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)
        h_pair = h_i * h_j

        t_emb = self.time_embedding(t)
        t_pair = t_emb[:, None, None, :].expand(B, N, N, self.time_emb_dim)
        adj_pair = adj_noisy.unsqueeze(-1)

        edge_input = torch.cat([h_i, h_j, h_pair, t_pair, adj_pair], dim=-1)
        out_E = self.out_E(edge_input).squeeze(-1)

        if self.force_symmetric_output:
            out_E = 0.5 * (out_E + out_E.transpose(1, 2))

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            out_E = out_E * pair_mask.float()

        eye = torch.eye(N, device=out_E.device).unsqueeze(0)
        out_E = out_E * (1.0 - eye)

        return out_E

    def decode_X(
        self,
        h: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        B, N, _ = h.shape

        t_emb = self.time_embedding(t)
        t_node = t_emb[:, None, :].expand(B, N, self.time_emb_dim)

        x_input = torch.cat([h, t_node], dim=-1)
        out_X = self.out_X(x_input)

        if node_mask is not None:
            out_X = out_X * node_mask.unsqueeze(-1).float()

        return out_X

    def forward(
        self,
        x: Tensor,
        adj_noisy: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        h = self.encode_nodes(
            x=x,
            adj_noisy=adj_noisy,
            t=t,
            node_mask=node_mask,
        )

        out_E = self.decode_E(
            h=h,
            adj_noisy=adj_noisy,
            t=t,
            node_mask=node_mask,
        )

        out_X = self.decode_X(
            h=h,
            t=t,
            node_mask=node_mask,
        )

        return {
            "X": out_X,
            "E": out_E,
        }

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

    model = TransformerDenoiser(
        max_nodes=64,
        feature_dim=1433,
        hidden_dim=128,
        time_emb_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
    ).to(device)

    pred = model(x_feat, adj_noised, t)
    pred_noise = pred["E"]
    x0_pred = diffusion.predict_x0_from_noise(adj_noised, t, pred_noise)

    logger.debug(f"Adjacency Matrix: \n {adj[0, :3, :3]}")
    logger.debug(f"Noised Adjacency Matrix: \n {adj_noised[0, :3, :3]}")
    logger.debug(f"Noise: \n {noise[0, :3, :3]}")
    logger.debug(f"Predicted Noise: \n {pred_noise.detach()[0, :3, :3]}")
    logger.debug(f"Predicted x0: \n {x0_pred.detach()[0, :3, :3]}")

    sampled = diffusion.sample(
        model=model,
        x=x_feat[:1],
        adj_shape=(1, N, N),
        node_mask=node_mask[:1],
        device=device,
    )

    sampled = 0.5 * (sampled + sampled.transpose(1, 2))

    logger.debug(f"Sampled Adjacency Matrix: \n {sampled.detach()[0, :3, :3]}")
    logger.success("Model construction complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
