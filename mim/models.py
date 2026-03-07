import math
from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from layers.activations import d_phi, phi
from layers.pos_embed import SinusoidalPosEmb, get_2d_sincos_pos_embed


class HyperSETBlock(nn.Module):
    """Self-contained HyperSET recurrent block for the MIM (masked image modelling) task.

    Unlike :class:`layers.hyperset.HyperSETLayer` (the IC single-step block),
    this module owns its own sinusoidal recurrence number embedder and time MLP, and
    runs the full recurrent loop internally via ``forward(..., n_recur=N)``.

    Args:
        dim:           Token embedding dimension.
        heads:         Number of attention heads.
        dim_head:      Per-head dimension (must satisfy ``dim_head * heads == dim``).
        lmbd:          Regularisation weight.
        multiplier:    Expansion ratio for the feedforward projection.
        phi_func:      If set to a string activation name, use that activation
                       instead of bisoftmax attention or relu feedforward.
        adaptive_mode: Optimiser variant for the update step.  Required.
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        lmbd: float = 1.0,
        multiplier: int = 4,
        phi_func: Union[str, None] = None,
        adaptive_mode: str = "gd_with_learnable_step_size",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        assert inner_dim == dim

        self.heads = heads
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.phi_func = phi_func
        self.adaptive_mode = adaptive_mode
        self.input_cond = True
        assert self.adaptive_mode is not None, "adaptive_mode must be specified for HyperSETBlock"
        self.lmbd = lmbd

        self.qkv = nn.Linear(self.dim, self.dim, bias=False)
        self.ln1 = nn.RMSNorm(self.dim_head)

        self.mlp = nn.Linear(self.dim, multiplier * self.dim, bias=False)
        self.ln2 = nn.RMSNorm(self.dim * multiplier)

        sinu_pos_emb = SinusoidalPosEmb(self.dim)

        self.time_mlp = nn.ModuleList(
            [
                sinu_pos_emb,
                nn.Sequential(nn.RMSNorm(self.dim), nn.GELU(), nn.Linear(self.dim, 2 * self.dim)),
            ]
        )

        nn.init.constant_(self.time_mlp[1][2].weight, 0)
        nn.init.constant_(self.time_mlp[1][2].bias, 0)

        if self.phi_func is not None:
            assert type(self.phi_func) == str
            self.act_func = partial(phi, mode=self.phi_func)
            self.d_act_func = partial(d_phi, mode=self.phi_func)

        # This is a experimental feature.
        # Other modes perform worse than "gd_with_learnable_step_size" in our preliminary tests, but may be worth exploring further.
        if self.adaptive_mode == "gd_momentum_with_learnable_step_size":
            self.momentum = 0.9
        elif self.adaptive_mode == "rmsprop_with_fix_step_size":
            self.eta = 0.8
            self.alpha = 0.5
            self.beta = 0.1
        elif self.adaptive_mode == "rmsprop_momentum_with_fix_step_size":
            self.eta = 0.8
            self.momentum = 0.9
            self.alpha = 0.5
            self.beta = 0.1
        elif self.adaptive_mode == "adam_with_fix_step_size":
            self.eta1 = 0.9
            self.eta2 = 0.96
            self.alpha = 0.5
            self.beta = 0.1
        else:
            raise NotImplementedError

    def forward(
        self,
        x,
        attention_mode: str = "bisoftmax",
        feedforward_mode: str = "relu",
        n_recur: int = 12,
    ):
        n_recur_tensor = torch.arange(1, n_recur + 1, device=x.device)  # L 1
        n_recur_embedding = self.time_mlp[0](n_recur_tensor)  # L d

        x_inp = x
        for it in range(1, n_recur + 1):
            c = n_recur_embedding[it - 1].expand_as(x)
            time_emb_recur = self.time_mlp[1](c + x if not self.input_cond else c + x_inp)
            assert time_emb_recur.shape == (
                x.shape[0],
                256,
                2 * self.dim,
            ), time_emb_recur.shape

            x = self.update_x(
                x=x,
                attention_mode=attention_mode,
                feedforward_mode=feedforward_mode,
                time_emb_recur=time_emb_recur,
                it=it,
            )
        x_out = x
        return x_out

    def attention(self, x, attention_mode):
        B, T, C = x.size()
        w = self.qkv(x).view(B, T, self.heads, C // self.heads).transpose(1, 2)
        w = self.ln1(w)

        if self.phi_func is None:
            attn = (torch.matmul(w, w.transpose(-1, -2))) * self.scale
            att_to_check = attn.clone()
            if attention_mode == "relu":
                attn = F.relu(attn)
            elif attention_mode == "bisoftmax":
                attn = F.softmax(attn, dim=-1) + F.softmax(attn, dim=-2)
            elif attention_mode == "sigma":
                attn = F.sigmoid(attn) * (1 - F.sigmoid(attn))
            else:
                raise NotImplementedError

            out = torch.matmul(attn, w)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = F.linear(out, self.qkv.weight.t())
        else:
            # lienar attention
            phi_w = self.act_func(w)
            d_phi_w = self.d_act_func(w)

            attn = torch.matmul(phi_w.transpose(-1, -2), phi_w) * (T**-0.5)
            att_to_check = attn.clone()
            out = torch.matmul(phi_w, attn)
            out = out * d_phi_w
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = F.linear(out, self.qkv.weight.t())

        return out, att_to_check

    def ff(self, x, feedforward_mode):
        ff = self.ln2(self.mlp(x))
        if self.phi_func is None:
            if feedforward_mode == "relu":
                ff_hidden = F.relu(ff)
            elif feedforward_mode == "softmax":
                ff_hidden = F.softmax(ff, dim=-1)
            else:
                raise NotImplementedError
            out = F.linear(ff_hidden, self.mlp.weight.t())
        else:
            # gated feedforward
            phi_ff = self.act_func(ff)
            d_phi_ff = self.d_act_func(ff)
            ff_hidden = phi_ff.sum(dim=-1, keepdim=True) * d_phi_ff
            out = F.linear(ff_hidden, self.mlp.weight.t())

        return out

    def update_x(self, x, attention_mode, feedforward_mode, it, time_emb_recur):
        alpha, beta = time_emb_recur.chunk(2, dim=-1)

        if self.adaptive_mode == "gd_with_learnable_step_size":
            attn_out, _ = self.attention(x, attention_mode)
            x = x - alpha * attn_out
            ff_out = self.ff(x, feedforward_mode)
            x_out = x + beta * ff_out

        elif self.adaptive_mode == "gd_momentum_with_learnable_step_size":
            if it == 1:
                self.b1 = torch.zeros_like(x, device=x.device)
                self.b2 = torch.zeros_like(x, device=x.device)
            attn_out, _ = self.attention(x, attention_mode)
            self.b1 = self.momentum * self.b1 + attn_out
            x = x - alpha * self.b1
            ff_out = self.ff(x, feedforward_mode)
            self.b2 = self.momentum * self.b2 + ff_out
            x_out = x + beta * self.b2
            # x_out = self.post_norm(x_out)

        elif self.adaptive_mode == "rmsprop_with_fix_step_size":
            if it == 1:
                self.v1 = torch.zeros_like(x, device=x.device)
                self.v2 = torch.zeros_like(x, device=x.device)

            attn_out, _ = self.attention(x, attention_mode)
            self.v1 = self.eta * self.v1 + (1 - self.eta) * (attn_out**2)
            x = x - self.alpha * attn_out / (self.v1.sqrt() + 1e-6)

            ff_out = self.ff(x, feedforward_mode)
            self.v2 = self.eta * self.v2 + (1 - self.eta) * (ff_out**2)
            x_out = x + self.beta * ff_out / (self.v2.sqrt() + 1e-6)
            # x_out = self.post_norm(x_out)

        elif self.adaptive_mode == "rmsprop_momentum_with_fix_step_size":
            if it == 1:
                self.v1 = torch.zeros_like(x, device=x.device)
                self.v2 = torch.zeros_like(x, device=x.device)
                self.b1 = torch.zeros_like(x, device=x.device)
                self.b2 = torch.zeros_like(x, device=x.device)

            attn_out, _ = self.attention(x, attention_mode)
            self.v1 = self.eta * self.v1 + (1 - self.eta) * (attn_out**2)
            self.b1 = self.momentum * self.b1 + attn_out / (self.v1.sqrt() + 1e-6)
            x = x - self.alpha * self.b1

            ff_out = self.ff(x, feedforward_mode)
            self.v2 = self.eta * self.v2 + (1 - self.eta) * (ff_out**2)
            self.b2 = self.momentum * self.b2 + ff_out / (self.v2.sqrt() + 1e-6)
            x_out = x + self.beta * self.b2
            # x_out = self.post_norm(x_out)

        elif self.adaptive_mode == "adam_with_fix_step_size":
            if it == 1:
                self.v1 = torch.zeros_like(x, device=x.device)
                self.v2 = torch.zeros_like(x, device=x.device)
                self.m1 = torch.zeros_like(x, device=x.device)
                self.m2 = torch.zeros_like(x, device=x.device)

            attn_out, _ = self.attention(x, attention_mode)
            self.m1 = self.eta1 * self.m1 + (1 - self.eta1) * attn_out
            self.v1 = self.eta2 * self.v1 + (1 - self.eta2) * (attn_out**2)
            m1_tilde = self.m1 / (1 - self.eta1**it)
            v1_tilde = self.v1 / (1 - self.eta2**it)
            x = x - self.alpha * m1_tilde / (v1_tilde.sqrt() + 1e-6)

            ff_out = self.ff(x, feedforward_mode)
            self.m2 = self.eta1 * self.m2 + (1 - self.eta1) * ff_out
            self.v2 = self.eta2 * self.v2 + (1 - self.eta2) * (ff_out**2)
            m2_tilde = self.m2 / (1 - self.eta1**it)
            v2_tilde = self.v2 / (1 - self.eta2**it)
            x_out = x + self.beta * m2_tilde / (v2_tilde.sqrt() + 1e-6)
            # x_out = self.post_norm(x_out)

        return x_out

    @torch.no_grad()
    def energy(self, x, attention_mode, feedforward_mode):
        B, T, C = x.size()
        w = self.qkv(x).view(B, T, self.heads, C // self.heads).transpose(1, 2)
        w = F.rms_norm(w, (w.shape[-1],))
        if self.phi_func is None:
            attn = torch.matmul(w, w.transpose(-1, -2)) * self.scale
            if attention_mode == "relu":
                E_attn = (0.25 * (F.relu(attn)) ** 2).view(B, -1).sum(dim=1, keepdim=True) / self.scale
            elif attention_mode == "bisoftmax":
                E_attn = attn.logsumexp(dim=-1).view(B, -1).sum(dim=1, keepdim=True) / self.scale
            elif attention_mode == "sigma":
                E_attn = (0.5 * (F.sigmoid(attn))).view(B, -1).sum(dim=1, keepdim=True) / self.scale
            else:
                raise NotImplementedError
        else:
            phi_w = self.act_func(w)
            attn = torch.matmul(phi_w, phi_w.transpose(-1, -2)) * self.scale
            E_attn = 0.25 * (attn**2).view(B, -1).sum(dim=1).mean(0) / self.scale

        ff = self.mlp(x)
        ff = F.rms_norm(ff, (ff.shape[-1],))
        if self.phi_func is None:
            if feedforward_mode == "relu":
                E_ff = -(0.5 * (F.relu(ff)).view(B, -1).norm(dim=1, keepdim=True) ** 2)
            elif feedforward_mode == "softmax":
                E_ff = -ff.logsumexp(dim=-1).view(B, -1).sum(dim=1, keepdim=True)
            else:
                raise NotImplementedError
        else:
            phi_ff = self.act_func(ff)
            E_ff = -0.5 * (phi_ff.norm(dim=-1) ** 2).view(B, -1).sum(dim=1, keepdim=True)

        return E_attn + E_ff, E_attn, E_ff


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """PreNorm module to apply normalization before a given function
        :param:
            dim  -> int: Dimension of the input
            fn   -> nn.Module: The function to apply after normalization
        """
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass through the PreNorm module
        :param:
            x        -> torch.Tensor: Input tensor
            **kwargs -> _ : Additional keyword arguments for the function
        :return
            torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        """Initialize the Multi-Layer Perceptron (MLP).
        :param:
            dim        -> int : Dimension of the input
            dim        -> int : Dimension of the hidden layer
            dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x):
        """Forward pass through the MLP module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """Initialize the Attention module.
        :param:
            embed_dim     -> int : Dimension of the embedding
            num_heads     -> int : Number of heads
            dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.n_head = num_heads

    def forward(self, x):
        """Forward pass through the Attention module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            attention_value  -> torch.Tensor: Output the value of the attention
            attention_weight -> torch.Tensor: Output the weight of the attention
        """

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_to_check = att.clone()
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.proj(y)
        return y, att_to_check


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        """Initialize the Attention module.
        :param:
            dim       -> int : number of hidden dimension of attention
            depth     -> int : number of layer for the transformer
            heads     -> int : Number of heads
            mlp_dim   -> int : number of hidden dimension for mlp
            dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, n_recur: int = 12):
        """Forward pass through the Transformer Block.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            x -> torch.Tensor: Output of the Transformer
            l_attn (Optional) -> list(torch.Tensor): list of the attention
        """

        for attn, ff in self.layers:
            for it in range(1, n_recur + 1):
                attention_value, attention_weight = attn(x)
                x = attention_value + x
                x = ff(x) + x

        return x


class MaskTransformerUncond(nn.Module):
    def __init__(
        self,
        model: str = "hyper-set",
        img_size: int = 256,
        codebook_size: int = 1024,
        n_embd: int = 768,
        n_layer: int = 1,
        n_recur: int = 12,
        head: int = 8,
        multiplier: int = 4,
        dropout: float = 0.1,
        nclass: int = 10,
        phi_func: Union[str, None] = None,
        adaptive_mode: Union[str, None] = None,
    ):
        super().__init__()
        self.model = model
        self.n_recur = n_recur
        self.nclass = nclass
        self.patch_size = img_size // 16
        self.codebook_size = codebook_size

        self.tok_emb = nn.Embedding(codebook_size + 1, n_embd)
        self.pos_emb = (
            torch.from_numpy(get_2d_sincos_pos_embed(n_embd, self.patch_size, cls_token=False, extra_tokens=0))
            .float()
            .unsqueeze(0)
        )

        if model == "hyper-set":
            self.transformer = HyperSETBlock(
                dim=n_embd,
                multiplier=multiplier,
                heads=head,
                dim_head=n_embd // head,
                phi_func=phi_func,
                adaptive_mode=adaptive_mode,
            )
        elif model == "transformer":
            self.transformer = TransformerBlock(
                dim=n_embd,
                depth=n_layer,
                heads=head,
                mlp_dim=n_embd * multiplier,
                dropout=dropout,
            )
        else:
            raise NotImplementedError

        self.lm_head = nn.Sequential(nn.RMSNorm(n_embd), nn.Linear(n_embd, codebook_size))

    def forward(self, img_token, y=None, drop_label=None, n_recur=12):
        """Forward.
        :param:
            img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            y              -> torch.LongTensor: condition class to generate
            drop_label     -> torch.BoolTensor: either or not to drop the condition
            return_attn    -> Bool: return the attn for visualization
        :return:
            logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
            attn:          -> list(torch.FloatTensor) (Optional): list of attention for visualization
        """
        b, w, h = img_token.size()

        input = img_token.view(b, -1)

        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb.to(tok_embeddings.device)

        x = tok_embeddings + pos_embeddings

        if self.model == "hyper-set":
            x = self.transformer(x, attention_mode="bisoftmax", feedforward_mode="relu", n_recur=n_recur)
        elif self.model == "transformer":
            x = self.transformer(x, n_recur=n_recur)

        logit = self.lm_head(x)
        # return logit[:, :self.patch_size*self.patch_size, :self.codebook_size+1]
        return logit


if __name__ == "__main__":
    b, n, f = 4, 16, 128
    layer = TransformerBlock(dim=f, depth=1, heads=2, mlp_dim=f * 4, dropout=0.1)
    img_token = torch.randn((b, n, f))
    logit = layer(img_token, 1)
    print(logit.shape)
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"Number of parameters: {n_params / 1e6:.2f}M")
