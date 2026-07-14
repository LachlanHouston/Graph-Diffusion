from pathlib import Path

from loguru import logger
import typer

import math
import torch
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
import networkx as nx

from src.modeling.utils import graph_from_adjacency

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

class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        x_classes: int,
        e_classes: int,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.x_classes = x_classes
        self.e_classes = e_classes
        self.num_steps = num_steps

        betas = linear_beta_schedule(num_steps=self.num_steps, beta_start=self.beta_start, beta_end=self.beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer(
            "u_x",
            torch.full((x_classes,), 1.0 / x_classes),
        )
        self.register_buffer(
            "u_e",
            torch.full((e_classes,), 1.0 / e_classes),
        )

        self.register_buffer("eye_x", torch.eye(x_classes).unsqueeze(0))
        self.register_buffer("eye_e", torch.eye(e_classes).unsqueeze(0))

    def _normalize_node_mask(
        self,
        node_mask: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        if node_mask is None:
            return None
        return node_mask.to(device=device, dtype=torch.bool)

    def sample_symmetric_edge_classes(self, probs: Tensor) -> Tensor:
        """
        Sample one categorical value per undirected node pair and mirror it.

        probs: [B, N, N, K]
        returns: [B, N, N]
        """
        B, N, _, K = probs.shape

        probs = 0.5 * (probs + probs.transpose(1, 2))

        upper_i, upper_j = torch.triu_indices(
            N,
            N,
            offset=1,
            device=probs.device,
        )
        upper_probs = probs[:, upper_i, upper_j, :]  # [B, P, K]
        upper_samples = self.sample_categorical(upper_probs)  # [B, P]

        e = torch.zeros(B, N, N, dtype=torch.long, device=probs.device)
        e[:, upper_i, upper_j] = upper_samples
        e[:, upper_j, upper_i] = upper_samples
        return e

    def get_Qt(self, t: Tensor):
        beta_t = self.betas[t].view(-1, 1, 1)

        q_x = (1.0 - beta_t) * self.eye_x + beta_t * self.u_x
        q_e = (1.0 - beta_t) * self.eye_e + beta_t * self.u_e

        return {
            "X": q_x,
            "E": q_e,
        }

    def get_Qt_bar(self, t: Tensor):
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)

        q_x = alpha_bar_t * self.eye_x + (1.0 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * self.eye_e + (1.0 - alpha_bar_t) * self.u_e

        return {
            "X": q_x,
            "E": q_e,
        }

    def get_Qt_bar_prev(self, t: Tensor):
        alpha_bar_prev_t = torch.ones_like(self.alpha_bars[t])
        nonzero_t = t > 0
        alpha_bar_prev_t[nonzero_t] = self.alpha_bars[t[nonzero_t] - 1]
        alpha_bar_prev_t = alpha_bar_prev_t.view(-1, 1, 1)

        q_x = alpha_bar_prev_t * self.eye_x + (1.0 - alpha_bar_prev_t) * self.u_x
        q_e = alpha_bar_prev_t * self.eye_e + (1.0 - alpha_bar_prev_t) * self.u_e

        return {
            "X": q_x,
            "E": q_e,
        }

    def posterior_node_probs(self, pred_x0_probs: Tensor, x_t: Tensor, t: Tensor):
        """
        Approximate p_theta(x_{t-1} | x_t) by marginalising over predicted x_0.

        For each possible clean class x_0 and previous class x_{t-1}, use:

            q(x_{t-1} | x_t, x_0)
            ∝ q(x_t | x_{t-1}) q(x_{t-1} | x_0) / q(x_t | x_0)

        then weight by p_theta(x_0 | x_t).
        """
        B, N, K = pred_x0_probs.shape

        q_t = self.get_Qt(t)["X"]                       # [B, K, K], from x_{t-1} to x_t
        q_bar_t = self.get_Qt_bar(t)["X"]               # [B, K, K], from x_0 to x_t
        q_bar_prev = self.get_Qt_bar_prev(t)["X"]       # [B, K, K], from x_0 to x_{t-1}

        # q(x_t=current | x_{t-1}=k) for all candidate previous classes k.
        q_t_given_prev = q_t[:, None, :, :].expand(B, N, K, K)
        current_x_for_prev = x_t[:, :, None, None].expand(B, N, K, 1)
        q_current_given_prev = q_t_given_prev.gather(
            dim=-1,
            index=current_x_for_prev,
        ).squeeze(-1)                                    # [B, N, K]

        # q(x_t=current | x_0=c) for all candidate clean classes c.
        q_bar_t_expanded = q_bar_t[:, None, :, :].expand(B, N, K, K)
        current_x_for_x0 = x_t[:, :, None, None].expand(B, N, K, 1)
        q_current_given_x0 = q_bar_t_expanded.gather(
            dim=-1,
            index=current_x_for_x0,
        ).squeeze(-1).clamp_min(1e-12)                   # [B, N, K]

        # q(x_{t-1}=k | x_0=c) for all clean classes c and previous classes k.
        q_prev_given_x0 = q_bar_prev[:, None, :, :].expand(B, N, K, K)

        # Sum over possible clean classes c:
        # p_theta(c | x_t) * q(x_{t-1}=k | c) * q(x_t | k) / q(x_t | c)
        weights_x0 = pred_x0_probs / q_current_given_x0
        posterior = torch.einsum(
            "bnc,bnck,bnk->bnk",
            weights_x0,
            q_prev_given_x0,
            q_current_given_prev,
        )

        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return posterior

    def posterior_edge_probs(self, pred_e0_probs: Tensor, e_t: Tensor, t: Tensor):
        """
        Approximate p_theta(e_{t-1} | e_t) by marginalising over predicted e_0.

        This is the edge analogue of posterior_node_probs.
        """
        B, N, _, K = pred_e0_probs.shape

        q_t = self.get_Qt(t)["E"]                       # [B, K, K], from e_{t-1} to e_t
        q_bar_t = self.get_Qt_bar(t)["E"]               # [B, K, K], from e_0 to e_t
        q_bar_prev = self.get_Qt_bar_prev(t)["E"]       # [B, K, K], from e_0 to e_{t-1}

        # q(e_t=current | e_{t-1}=k) for all candidate previous edge classes k.
        q_t_given_prev = q_t[:, None, None, :, :].expand(B, N, N, K, K)
        current_e_for_prev = e_t[:, :, :, None, None].expand(B, N, N, K, 1)
        q_current_given_prev = q_t_given_prev.gather(
            dim=-1,
            index=current_e_for_prev,
        ).squeeze(-1)                                    # [B, N, N, K]

        # q(e_t=current | e_0=c) for all candidate clean edge classes c.
        q_bar_t_expanded = q_bar_t[:, None, None, :, :].expand(B, N, N, K, K)
        current_e_for_e0 = e_t[:, :, :, None, None].expand(B, N, N, K, 1)
        q_current_given_e0 = q_bar_t_expanded.gather(
            dim=-1,
            index=current_e_for_e0,
        ).squeeze(-1).clamp_min(1e-12)                   # [B, N, N, K]

        # q(e_{t-1}=k | e_0=c) for all clean classes c and previous classes k.
        q_prev_given_e0 = q_bar_prev[:, None, None, :, :].expand(B, N, N, K, K)

        # Sum over possible clean classes c:
        # p_theta(c | e_t) * q(e_{t-1}=k | c) * q(e_t | k) / q(e_t | c)
        weights_e0 = pred_e0_probs / q_current_given_e0
        posterior = torch.einsum(
            "bnmc,bnmck,bnmk->bnmk",
            weights_e0,
            q_prev_given_e0,
            q_current_given_prev,
        )

        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return posterior

    def p_sample_step(
        self,
        model: nn.Module,
        x_t: Tensor,
        node_features: Tensor,
        e_t: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ):
        node_mask = self._normalize_node_mask(node_mask, x_t.device)
        node_features = node_features.to(device=x_t.device, dtype=torch.float)

        logits = model(
            x=x_t,
            node_features=node_features,
            adj_noisy=e_t,
            t=t,
            node_mask=node_mask,
        )

        edge_logits = logits["E"]
        edge_logits = 0.5 * (
            edge_logits + edge_logits.transpose(1, 2)
        )

        pred_x0_probs = torch.softmax(logits["X"], dim=-1)
        pred_e0_probs = torch.softmax(edge_logits, dim=-1)

        x_prev_probs = self.posterior_node_probs(
            pred_x0_probs=pred_x0_probs,
            x_t=x_t,
            t=t,
        )
        e_prev_probs = self.posterior_edge_probs(
            pred_e0_probs=pred_e0_probs,
            e_t=e_t,
            t=t,
        )

        zero_mask_x = (t == 0).view(-1, 1, 1)
        zero_mask_e = (t == 0).view(-1, 1, 1, 1)

        x_prev_probs = torch.where(
            zero_mask_x,
            pred_x0_probs,
            x_prev_probs,
        )
        e_prev_probs = torch.where(
            zero_mask_e,
            pred_e0_probs,
            e_prev_probs,
        )

        x_prev = self.sample_categorical(x_prev_probs)
        e_prev = self.sample_symmetric_edge_classes(e_prev_probs)

        x_prev = self.clean_node_classes(x_prev, node_mask)
        e_prev = self.clean_edge_classes(e_prev, node_mask)

        return x_prev, e_prev

    def sample_categorical(self, probs: Tensor):
        original_shape = probs.shape[:-1]
        num_classes = probs.shape[-1]

        probs = probs.reshape(-1, num_classes)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return samples.reshape(original_shape)

    def clean_node_classes(self, x: Tensor, node_mask: Tensor | None = None):
        node_mask = self._normalize_node_mask(node_mask, x.device)
        if node_mask is not None:
            x = x.masked_fill(~node_mask, 0)
        return x.long()

    def clean_edge_classes(self, e: Tensor, node_mask: Tensor | None = None):
        B, N, _ = e.shape
        node_mask = self._normalize_node_mask(node_mask, e.device)

        upper = torch.triu(e.long(), diagonal=1)
        e = upper + upper.transpose(1, 2)

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            e = e.masked_fill(~pair_mask, 0)

        eye = torch.eye(N, dtype=torch.bool, device=e.device).unsqueeze(0)
        e = e.masked_fill(eye, 0)

        return e.long()

    def q_sample(
        self,
        x0: Tensor,
        e0: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ):
        node_mask = self._normalize_node_mask(node_mask, torch.device(x0.device))

        x0 = self.clean_node_classes(x0, node_mask)
        e0 = self.clean_edge_classes(e0, node_mask)

        q_bar = self.get_Qt_bar(t)

        x0_onehot = torch.nn.functional.one_hot(
            x0,
            num_classes=self.x_classes,
        ).float()

        e0_onehot = torch.nn.functional.one_hot(
            e0,
            num_classes=self.e_classes,
        ).float()

        prob_X = torch.einsum("bnc,bcd->bnd", x0_onehot, q_bar["X"])
        prob_E = torch.einsum("bnmc,bcd->bnmd", e0_onehot, q_bar["E"])

        x_t = self.sample_categorical(prob_X)
        e_t = self.sample_symmetric_edge_classes(prob_E)

        x_t = self.clean_node_classes(x_t, node_mask)
        e_t = self.clean_edge_classes(e_t, node_mask)

        return {
            "X_t": x_t,
            "E_t": e_t,
            "prob_X": prob_X,
            "prob_E": prob_E,
        }

    def sample_prior(
        self,
        batch_size: int,
        num_nodes: int,
        node_mask: Tensor | None = None,
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = self.betas.device

        x_probs = self.u_x.to(device).view(
            1, 1, self.x_classes
        ).expand(
            batch_size, num_nodes, self.x_classes
        )

        e_probs = self.u_e.to(device).view(
            1, 1, 1, self.e_classes
        ).expand(
            batch_size, num_nodes, num_nodes, self.e_classes
        )

        node_mask = self._normalize_node_mask(node_mask, torch.device(device))

        x_t = self.sample_categorical(x_probs)
        e_t = self.sample_symmetric_edge_classes(e_probs)

        x_t = self.clean_node_classes(x_t, node_mask)
        e_t = self.clean_edge_classes(e_t, node_mask)

        return {
            "X": x_t,
            "E": e_t,
        }

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        node_features: Tensor,
        batch_size: int,
        num_nodes: int,
        keep_chain: bool = False,
        n_chains: int = 1,
        node_mask: Tensor | None = None,
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = next(model.parameters()).device

        device = torch.device(device)
        node_mask = self._normalize_node_mask(node_mask, device)
        node_features = node_features.to(device=device, dtype=torch.float)

        expected_shape = (batch_size, num_nodes)
        if node_features.shape[:2] != expected_shape:
            raise ValueError(
                "node_features must have shape "
                f"[batch_size, num_nodes, feature_dim], got {tuple(node_features.shape)} "
                f"for batch_size={batch_size} and num_nodes={num_nodes}."
            )

        if node_mask is not None and node_mask.shape != expected_shape:
            raise ValueError(
                f"node_mask must have shape {expected_shape}, got {tuple(node_mask.shape)}."
            )

        if node_mask is not None:
            node_features = node_features * node_mask.unsqueeze(-1).to(node_features.dtype)

        prior = self.sample_prior(
            batch_size=batch_size,
            num_nodes=num_nodes,
            node_mask=node_mask,
            device=device,
        )

        x_t = prior["X"]
        e_t = prior["E"]

        x_chain = None
        e_chain = None
        if keep_chain:
            x_chain = torch.zeros(
                self.num_steps,
                min(n_chains, batch_size),
                x_t.size(1),
                dtype=torch.long,
                device=device,
            )
            e_chain = torch.zeros(
                self.num_steps,
                min(n_chains, batch_size),
                e_t.size(1),
                e_t.size(2),
                dtype=torch.long,
                device=device,
            )

        for i in reversed(range(self.num_steps)):
            t = torch.full(
                size=(batch_size,),
                fill_value=i,
                device=device,
                dtype=torch.long,
            )

            x_t, e_t = self.p_sample_step(
                model=model,
                x_t=x_t,
                node_features=node_features,
                e_t=e_t,
                t=t,
                node_mask=node_mask,
            )

            if keep_chain:
                x_chain[self.num_steps - i - 1] = x_t[: x_chain.size(1)]
                e_chain[self.num_steps - i - 1] = e_t[: e_chain.size(1)]

        if keep_chain:
            return {
                "X": x_t,
                "E": e_t,
            }, {
                "X_chain": x_chain,
                "E_chain": e_chain,
            }

        return {
            "X": x_t,
            "E": e_t,
        }, None

class DenseGraphAttentionBlock(nn.Module):
    def __init__(
        self,
        e_classes: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.e_classes = e_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Embedding(e_classes, num_heads)

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

        if node_mask is not None:
            node_mask = node_mask.to(device=h.device, dtype=torch.bool)

        q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        edge_bias = self.edge_bias(adj_noisy.long()).permute(0, 3, 1, 2)
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
        feature_dim: int = 3703,
        x_classes: int = 6,
        e_classes: int = 2,
        hidden_dim: int = 128,
        time_emb_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        force_symmetric_output: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.x_classes = x_classes
        self.e_classes = e_classes
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.force_symmetric_output = force_symmetric_output

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        self.node_embedding = nn.Embedding(x_classes, hidden_dim)

        self.node_input_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        edge_input_dim = 3 * hidden_dim + time_emb_dim + e_classes

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
            if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.attn_layers = nn.ModuleList(
            [
                DenseGraphAttentionBlock(
                    e_classes=e_classes,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_E = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, e_classes),
        )

        self.out_X = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, self.x_classes),
        )

    def encode_nodes(
        self,
        x: Tensor,
        node_features: Tensor,
        adj_noisy: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        B, N = x.shape

        if node_mask is not None:
            node_mask = node_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

        node_features = node_features.to(device=x.device, dtype=torch.float)

        if node_features.shape[:2] != (B, N):
            raise ValueError(
                "node_features must have shape [B, N, feature_dim], "
                f"got {tuple(node_features.shape)} for x shape {tuple(x.shape)}."
            )

        if node_features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected node feature dimension {self.feature_dim}, "
                f"got {node_features.size(-1)}."
            )

        if node_mask is not None:
            node_features = node_features * node_mask.unsqueeze(-1).to(node_features.dtype)

        class_embed = self.node_embedding(x)

        feature_embed = self.feature_encoder(
            node_features.float()
        )

        t_emb = self.time_embedding(t)
        t_node = t_emb[:, None, :].expand(
            B,
            N,
            self.time_emb_dim,
        )

        h = torch.cat(
            [
                class_embed,
                feature_embed,
                t_node,
            ],
            dim=-1,
        )
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

        if node_mask is not None:
            node_mask = node_mask.to(device=h.device, dtype=torch.bool)

        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)
        h_pair = h_i * h_j

        t_emb = self.time_embedding(t)
        t_pair = t_emb[:, None, None, :].expand(B, N, N, self.time_emb_dim)
        adj_pair = torch.nn.functional.one_hot(
            adj_noisy.long(),
            num_classes=self.e_classes,
        ).float()

        edge_input = torch.cat([h_i, h_j, h_pair, t_pair, adj_pair], dim=-1)
        out_E = self.out_E(edge_input)

        if self.force_symmetric_output:
            out_E = 0.5 * (out_E + out_E.transpose(1, 2))

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            out_E = out_E * pair_mask.unsqueeze(-1).float()

        eye = torch.eye(N, dtype=torch.bool, device=out_E.device).unsqueeze(0).unsqueeze(-1)
        out_E = out_E.masked_fill(eye, 0.0)

        return out_E

    def decode_X(
        self,
        h: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        B, N, _ = h.shape

        if node_mask is not None:
            node_mask = node_mask.to(device=h.device, dtype=torch.bool)

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
        node_features: Tensor,
        adj_noisy: Tensor,
        t: Tensor,
        node_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        h = self.encode_nodes(
            x=x,
            node_features=node_features,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 4
    N = 8
    F = 3703
    x_classes = 6
    e_classes = 2

    f = torch.randint(
        0,
        2,
        (B, N, F),
        dtype=torch.float,
        device=device,
    )
    x = torch.randint(0, x_classes, (B, N), dtype=torch.long, device=device)
    e = torch.randint(0, e_classes, (B, N, N), dtype=torch.long, device=device)
    e = torch.triu(e, diagonal=1)
    e = e + e.transpose(1, 2)

    node_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    diffusion = DiscreteDiffusion(
        x_classes=x_classes,
        e_classes=e_classes,
        num_steps=1000,
    ).to(device)

    t = sample_timesteps(
        batch_size=B,
        num_steps=diffusion.num_steps,
        device=device,
    )

    noised = diffusion.q_sample(
        x0=x,
        e0=e,
        t=t,
        node_mask=node_mask,
    )

    prior = diffusion.sample_prior(
        batch_size=B,
        num_nodes=N,
        node_mask=node_mask,
        device=device,
    )

    model = TransformerDenoiser(
        max_nodes=N,
        feature_dim=F,
        x_classes=x_classes,
        e_classes=e_classes,
        hidden_dim=128,
        time_emb_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    ).to(device)

    logits = model(
        x=noised["X_t"],
        node_features=f,
        adj_noisy=noised["E_t"],
        t=t,
        node_mask=node_mask,
    )

    sampled, chain = diffusion.sample(
        model=model,
        node_features=f,
        batch_size=B,
        num_nodes=N,
        keep_chain=True,
        node_mask=node_mask,
        device=device,
    )

    logger.debug(f"Original X: \n{x[0]}")
    logger.debug(f"Noised X_t: \n{noised['X_t'][0]}")
    logger.debug(f"Original E: \n{e[0, :4, :4]}")
    logger.debug(f"Noised E_t: \n{noised['E_t'][0, :4, :4]}")
    logger.debug(f"Prior X_T: \n{prior['X'][0]}")
    logger.debug(f"Prior E_T: \n{prior['E'][0, :4, :4]}")
    logger.debug(f"Logits X shape: {logits['X'].shape}")
    logger.debug(f"Logits E shape: {logits['E'].shape}")
    logger.debug(f"Sampled X shape: {sampled['X'].shape}")
    logger.debug(f"Sampled E shape: {sampled['E'].shape}")
    logger.debug(f"Sampled X: \n{sampled['X'][0]}")
    logger.debug(f"Sampled E: \n{sampled['E'][0, :4, :4]}")
    logger.success("Discrete diffusion construction complete.")

if __name__ == "__main__":
    app()
