import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from layers.components.attention import CRATEAttention
from layers.components.feedforward import CRATEFeedForward
from layers.components.norm import EnergyLayerNorm, PreNorm
from layers.et import ETBlock
from layers.pos_embed import SinusoidalPosEmb, get_2d_sincos_pos_embed
from utils.metric_utils import (
    compute_average_angle,
    compute_effective_rank,
    compute_rank,
)


class ModelConfig:
    """Base configuration shared by all Sudoku model variants.

    Attributes:
        vocab_size: Size of the token vocabulary (digits 0-9, so 10).
        block_size: Sequence length (81 cells for a 9x9 Sudoku grid).
        embd_pdrop: Embedding dropout probability.
        resid_pdrop: Residual connection dropout probability.
        attn_pdrop:  Attention weight dropout probability.
    """

    embd_pdrop = 0.1
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    def __init__(self, vocab_size: int, block_size: int, **kwargs) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = None
        for k, v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.config = config
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        if config.model_type == "hyper-set":
            self.attn_drop = nn.Dropout(config.attn_pdrop)
            self.resid_drop = nn.Dropout(config.resid_pdrop)
            self.dim_head = config.n_embd // config.n_head
            self.n_head = config.n_head
            self.qkv = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.scale = self.dim_head**-0.5
            self.ln1 = nn.RMSNorm(self.dim_head)

        elif config.model_type == "transformer":
            self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.attn_drop = nn.Dropout(config.attn_pdrop)
            self.resid_drop = nn.Dropout(config.resid_pdrop)
            # output projection
            self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.n_head = config.n_head

        elif config.model_type == "transformer_shared":
            self.qkv = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.attn_drop = nn.Dropout(config.attn_pdrop)
            self.resid_drop = nn.Dropout(config.resid_pdrop)
            # output projection
            self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.n_head = config.n_head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.model_type == "hyper-set":
            B, T, C = x.size()
            w = self.qkv(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            w = self.ln1(w)
            att = (w @ w.transpose(-1, -2)) * self.scale
            att_to_check = att.clone()
            att = F.softmax(att, dim=-1) + F.softmax(att, dim=-2)
            att = self.attn_drop(att)
            y = att @ w
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = F.linear(y, self.qkv.weight.t())
            y = self.resid_drop(y)
            return y, att_to_check

        elif self.config.model_type == "transformer":
            if isinstance(x, tuple):
                x = x[0]
            B, T, C = x.size()

            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att_to_check = att.clone()
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            # re-assemble all head outputs side by side
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            # output projection
            y = self.resid_drop(self.proj(y))
            return y, att_to_check

        elif self.config.model_type == "transformer_shared":
            if isinstance(x, tuple):
                x = x[0]
            B, T, C = x.size()

            qkv = self.qkv(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            att = (qkv @ qkv.transpose(-2, -1)) * (1.0 / math.sqrt(qkv.size(-1)))
            att_to_check = att.clone()
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ qkv  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            # re-assemble all head outputs side by side
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            # output projection
            y = self.resid_drop(self.proj(y))
            return y, att_to_check


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.model_type == "hyper-set":
            self.mlp = nn.Linear(config.n_embd, config.multiplier * config.n_embd, bias=False)
            self.attn = SelfAttention(config)
            self.ln2 = nn.RMSNorm(config.n_embd * config.multiplier)
            self.mlp_drop = nn.Dropout(config.resid_pdrop)

            if config.adaptive_mode == "gd_momentum_with_learnable_step_size":
                self.momentum = 0.9
            elif config.adaptive_mode == "rmsprop_with_fix_step_size":
                self.eta = 0.8
            elif config.adaptive_mode == "rmsprop_momentum_with_fix_step_size":
                self.eta = 0.8
                self.momentum = 0.9
            elif config.adaptive_mode == "adam_with_fix_step_size":
                self.eta1 = 0.9
                self.eta2 = 0.96
        elif config.model_type == "transformer" or config.model_type == "transformer_shared":
            self.ln1 = nn.RMSNorm(config.n_embd)
            self.ln2 = nn.RMSNorm(config.n_embd)
            self.attn = SelfAttention(config)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.multiplier * config.n_embd, bias=False),
                nn.ReLU(),
                nn.Linear(config.multiplier * config.n_embd, config.n_embd, bias=False),
                nn.Dropout(config.resid_pdrop),
            )

        elif config.model_type == "crate":
            self.attn = PreNorm(
                config.n_embd,
                CRATEAttention(
                    config.n_embd,
                    heads=config.n_head,
                    dim_head=config.n_embd // config.n_head,
                    dropout=0.0,
                ),
            )
            self.ff = PreNorm(
                config.n_embd,
                CRATEFeedForward(config.n_embd, config.n_embd, dropout=0.0, step_size=0.1),
            )
        elif config.model_type == "et":
            self.norm = EnergyLayerNorm(config.n_embd)
            self.block = ETBlock(
                config.n_embd,
                config.n_embd // config.n_head,
                config.n_head,
                4.0,
                None,
                False,
                False,
            )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        recur: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.model_type == "hyper-set":
            if time_emb is None:
                raise ValueError("time_emb must be provided for hyper-set mode")

            alpha, beta = time_emb.chunk(2, dim=-1)

            if self.config.adaptive_mode == "gd_with_learnable_step_size":
                att, att_to_check = self.attn(x)
                x = x - alpha * att
                x = x + beta * self.mlp_drop(F.linear(F.relu(self.ln2(self.mlp(x))), self.mlp.weight.t()))

            elif self.config.adaptive_mode == "gd_momentum_with_learnable_step_size":
                if recur == 1:
                    self.b1 = torch.zeros_like(x, device=x.device)
                    self.b2 = torch.zeros_like(x, device=x.device)
                att, att_to_check = self.attn(x)
                self.b1 = self.momentum * self.b1 + att
                x = x - alpha * self.b1

                self.b2 = self.momentum * self.b2 + F.linear(F.relu(self.ln2(self.mlp(x))), self.mlp.weight.t())
                x = x + beta * self.b2

            elif self.config.adaptive_mode == "rmsprop_with_fix_step_size":
                if recur == 1:
                    self.v1 = torch.zeros_like(x, device=x.device)
                    self.v2 = torch.zeros_like(x, device=x.device)

                att, att_to_check = self.attn(x)
                self.v1 = self.eta * self.v1 + (1 - self.eta) * (att**2)
                x = x - alpha * att / (self.v1.sqrt() + 1e-6)

                ff_out = F.linear(F.relu(self.ln2(self.mlp(x))), self.mlp.weight.t())
                self.v2 = self.eta * self.v2 + (1 - self.eta) * (ff_out**2)
                x = x + beta * ff_out / (self.v2.sqrt() + 1e-6)

            elif self.config.adaptive_mode == "rmsprop_momentum_with_fix_step_size":
                if recur == 1:
                    self.v1 = torch.zeros_like(x, device=x.device)
                    self.v2 = torch.zeros_like(x, device=x.device)
                    self.b1 = torch.zeros_like(x, device=x.device)
                    self.b2 = torch.zeros_like(x, device=x.device)

                att, att_to_check = self.attn(x)
                self.v1 = self.eta * self.v1 + (1 - self.eta) * (att**2)
                self.b1 = self.momentum * self.b1 + att / (self.v1.sqrt() + 1e-6)
                x = x - alpha * self.b1

                ff_out = F.linear(F.relu(self.ln2(self.mlp(x))), self.mlp.weight.t())
                self.v2 = self.eta * self.v2 + (1 - self.eta) * (ff_out**2)
                self.b2 = self.momentum * self.b2 + ff_out / (self.v2.sqrt() + 1e-6)
                x = x + beta * self.b2
                # x = self.mlp_drop(x)

            elif self.config.adaptive_mode == "adam_with_fix_step_size":
                if recur == 1:
                    self.v1 = torch.zeros_like(x, device=x.device)
                    self.v2 = torch.zeros_like(x, device=x.device)
                    self.m1 = torch.zeros_like(x, device=x.device)
                    self.m2 = torch.zeros_like(x, device=x.device)

                att, att_to_check = self.attn(x)
                self.m1 = self.eta1 * self.m1 + (1 - self.eta1) * att
                self.v1 = self.eta2 * self.v1 + (1 - self.eta2) * (att**2)
                m1_tilde = self.m1 / (1 - self.eta1**recur)
                v1_tilde = self.v1 / (1 - self.eta2**recur)
                x = x - alpha * m1_tilde / (v1_tilde.sqrt() + 1e-6)

                ff_out = F.linear(F.relu(self.ln2(self.mlp(x))), self.mlp.weight.t())
                self.m2 = self.eta1 * self.m2 + (1 - self.eta1) * ff_out
                self.v2 = self.eta2 * self.v2 + (1 - self.eta2) * (ff_out**2)
                m2_tilde = self.m2 / (1 - self.eta1**recur)
                v2_tilde = self.v2 / (1 - self.eta2**recur)
                x = x + beta * m2_tilde / (v2_tilde.sqrt() + 1e-6)

            return x, att_to_check

        elif self.config.model_type == "transformer" or self.config.model_type == "transformer_shared":
            if isinstance(x, tuple):
                x = x[0]
            att, att_to_check = self.attn(self.ln1(x))
            x = x + att
            x = x + self.mlp(self.ln2(x))
            return x, att_to_check

        elif self.config.model_type == "crate":
            grad_x = self.attn(x) + x
            x = self.ff(grad_x)
            return x, torch.randn(x.shape[0], self.config.n_head, x.shape[1], x.shape[1], device=x.device)

        elif self.config.model_type == "et":
            g = self.norm(x)
            dEdg, _ = torch.func.grad_and_value(self.block)(g)
            x = x - 1 * dEdg
            return x, torch.randn(x.shape[0], self.config.n_head, x.shape[1], x.shape[1], device=x.device)

    def energy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E_attn = self.attn_energy(x)
        E_ff = self.ff_energy(x)

        return E_attn + E_ff, E_attn, E_ff

    def attn_energy(self, x: torch.Tensor, attention_mode: str = "bisoftmax") -> torch.Tensor:
        B = x.shape[0]
        x_weight = self.attn.qkv(x)
        w = rearrange(x_weight, "b n (h d) -> b h n d", h=self.attn.n_head)
        w = F.rms_norm(w, (w.shape[-1],))
        attn = torch.matmul(w, w.transpose(-1, -2)) * self.attn.scale

        if attention_mode == "bisoftmax":
            E_attn = attn.logsumexp(dim=-1).view(B, -1).sum(dim=1, keepdim=True) / self.attn.scale
        elif attention_mode == "relu":
            E_attn = (0.25 * (F.relu(attn)) ** 2).view(B, -1).sum(dim=1, keepdim=True)  # .mean(dim=0)
        else:
            raise NotImplementedError

        return E_attn

    def ff_energy(self, x: torch.Tensor, feedforward_mode: str = "relu") -> torch.Tensor:
        B = x.shape[0]
        ff = self.mlp(x)
        ff = F.rms_norm(ff, (ff.shape[-1],))
        if feedforward_mode == "relu":
            E_ff = -0.5 * (F.relu(ff)).view(B, -1).norm(dim=1, keepdim=True) ** 2  # .mean(dim=0)
        elif feedforward_mode == "softmax":
            E_ff = -ff.logsumexp(dim=-1).view(B, -1).sum(dim=1, keepdim=True)  # .mean(dim=0)
        else:
            raise NotImplementedError

        return E_ff

    def stats(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ):
        B, T, C = x.size()
        if self.config.model_type == "hyper-set":
            if time_emb is None:
                raise ValueError("time_emb must be provided for hyper-set mode")
            alpha, beta = time_emb.chunk(2, dim=-1)
            if self.config.adaptive_mode == "gd_with_learnable_step_size":
                att, att_to_check = self.attn(x)
                x = x - alpha * att
                x = x + beta * F.linear(F.relu(self.ln2(self.mlp(x))), self.mlp.weight.t())
                x = self.mlp_drop(x)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        w = self.attn.qkv(x).view(B, T, self.attn.n_head, C // self.attn.n_head).transpose(1, 2)
        w = F.rms_norm(w, (w.shape[-1],))

        w_effective_rank = compute_effective_rank(w)
        assert w_effective_rank.shape == (B, self.attn.n_head, 1)

        w_average_angle = compute_average_angle(w)
        assert w_average_angle.shape == (B, self.attn.n_head, 1)

        w_rank = compute_rank(w)
        assert w_rank.shape == (B, self.attn.n_head, 1)

        energy, energy_attn, energy_ff = self.energy(x)
        assert energy.shape == energy_attn.shape == energy_ff.shape == (B, 1)

        return x, att_to_check, w_effective_rank, w_average_angle, w_rank, energy, energy_attn, energy_ff


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        if config.tok_emb:
            self.tok_emb = config.tok_emb(config=config)
        else:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        if config.pos_emb == "learnable":
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
            torch.nn.init.normal_(self.pos_emb, 0.0, 0.02)
        elif config.pos_emb == "sincos":
            self.pos_emb = (
                torch.from_numpy(get_2d_sincos_pos_embed(config.n_embd, 9, cls_token=False, extra_tokens=0))
                .float()
                .unsqueeze(0)
            )
        else:
            raise NotImplementedError

        assert self.pos_emb.shape == (
            1,
            config.block_size,
            config.n_embd,
        ), self.pos_emb.shape
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer block
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_classes, bias=False)

        self.n_recur = config.n_recur
        self.n_layer = config.n_layer

        self.block_size = config.block_size

        self.apply(self._init_weights)

        # learning adaptive step sizes
        if config.model_type == "hyper-set":
            sinu_pos_emb = SinusoidalPosEmb(config.time_emb)

            self.time_mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        sinu_pos_emb,
                        nn.Linear(config.time_emb, config.n_embd),
                        nn.GELU(),
                        nn.Linear(config.n_embd, config.n_embd),
                    ),
                    nn.Sequential(nn.GELU(), nn.Linear(config.n_embd, 2 * config.n_embd)),
                ]
            )
            nn.init.constant_(self.time_mlp[0][1].weight, 0)
            nn.init.constant_(self.time_mlp[0][1].bias, 0)
            nn.init.constant_(self.time_mlp[0][3].weight, 0)
            nn.init.constant_(self.time_mlp[0][3].bias, 0)

            nn.init.constant_(self.time_mlp[1][1].weight, 0)
            nn.init.constant_(self.time_mlp[1][1].bias, 0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.RMSNorm) and module.weight is not None:
            module.weight.data.fill_(1.0)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        idx_ulb: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        """
        Returns:
            loss as a scalar
            logits in the final prediction; (batch_size, 81, 9)
            attention for the first sample in a batch;
            (n_layer * n_recur, num_heads, 81, 81)
        """
        b, t = idx.shape[0], idx.shape[1]
        assert t == self.block_size
        # Each index maps to a learnable token vector.
        token_embeddings = self.tok_emb(idx)
        # Each position maps to a positional vector.
        position_embeddings = self.pos_emb[:, :t, :].to(token_embeddings.device)
        x = self.drop(token_embeddings + position_embeddings)

        if return_stats:
            energy_traj: List[torch.Tensor] = []
            energy_attn_traj: List[torch.Tensor] = []
            energy_ff_traj: List[torch.Tensor] = []
            effective_rank_traj: List[torch.Tensor] = []
            average_angle_traj: List[torch.Tensor] = []
            rank_traj: List[torch.Tensor] = []

        if self.config.model_type == "hyper-set":
            time = torch.arange(1, self.n_recur + 1, device=idx.device)
            time_emb = self.time_mlp[0](time)

        x_inp = x

        for block in self.blocks:
            for recur in range(self.n_recur):
                if self.config.model_type == "hyper-set":
                    c = time_emb[recur].expand_as(x)
                    time_emb_recur = self.time_mlp[1](c + x if not self.config.input_cond else c + x_inp)
                    assert time_emb_recur.shape == (
                        x.shape[0],
                        self.block_size,
                        2 * self.config.n_embd,
                    ), time_emb_recur.shape
                else:
                    time_emb_recur = None
                if return_stats:
                    x, attn, effective_rank, average_angle, rank, energy, energy_attn, energy_ff = block.stats(
                        x, time_emb_recur
                    )

                    energy_traj.append(energy)
                    energy_attn_traj.append(energy_attn)
                    energy_ff_traj.append(energy_ff)
                    effective_rank_traj.append(effective_rank)
                    average_angle_traj.append(average_angle)
                    rank_traj.append(rank)
                else:
                    x, attn = block(x, time_emb_recur, recur + 1)

        logit = self.head(self.ln_f(x))

        if targets is not None:
            loss = F.cross_entropy(logit.reshape(-1, logit.size(-1)), targets.view(-1))
        else:
            loss = 0

        if return_stats:
            return (
                logit,
                loss,
                torch.cat(energy_traj, dim=1).detach().cpu(),
                torch.cat(energy_attn_traj, dim=1).detach().cpu(),
                torch.cat(energy_ff_traj, dim=1).detach().cpu(),
                torch.cat(effective_rank_traj, dim=2).detach().cpu(),
                torch.cat(average_angle_traj, dim=2).detach().cpu(),
                torch.cat(rank_traj, dim=2).detach().cpu(),
            )
        return logit, loss
