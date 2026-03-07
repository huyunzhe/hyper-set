import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.components.norm import EnergyLayerNorm
from layers.crate import CRATELayer, CRATETLayer
from layers.et import ETBlock
from layers.hyperset import (
    HyperSETAlterAttentionLayer,
    HyperSETAlterFeedforwardLayer,
    HyperSETFixedStepSizeLayer,
    HyperSETLayer,
    HyperSETLayerLoRA,
)
from layers.pos_embed import (
    SinusoidalPosEmb,
    sinusoidal_positional_embedding,
)
from layers.transformer import TransformerLayer, TransformerLayerSharedQKV


class ViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 12,
        n_embd: int = 384,
        mlp_hidden: int = 384 * 4,
        head: int = 8,
        use_cls_token: bool = True,
    ):
        super(ViT, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_layer = n_layer
        self.n_recur = n_recur
        self.n_embd = n_embd
        self.emb = nn.Linear(f, n_embd)
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [TransformerLayer(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(self.n_layer)]
        )

        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)
        for enc in self.enc:
            for _ in range(self.n_recur):
                out = enc(out)
        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class ViTSharedQKV(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        mlp_hidden: int = 384 * 4,
        head: int = 8,
        use_cls_token: bool = True,
    ):
        super(ViTSharedQKV, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_recur = n_recur
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.emb = nn.Linear(f, n_embd)
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [
                TransformerLayerSharedQKV(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
                for _ in range(self.n_layer)
            ]
        )
        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)
        for enc in self.enc:
            for _ in range(self.n_recur):
                out = enc(out)
        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class ET(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        head: int = 8,
        use_cls_token: bool = True,
    ):
        super(ET, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_layer = n_layer
        self.n_recur = n_recur
        self.n_embd = n_embd
        self.emb = nn.Linear(f, n_embd)
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        EnergyLayerNorm(n_embd),
                        ETBlock(
                            n_embd,
                            n_embd // head,
                            head,
                            4.0,
                            None,
                            False,
                            False,
                        ),
                    ]
                )
                for _ in range(self.n_layer)
            ]
        )

        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

    def forward(self, x, alpha=1.0):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)
        for norm, block in self.enc:
            for _ in range(self.n_recur):
                out_norm = norm(out)
                dEdg, E = torch.func.grad_and_value(block)(out_norm)
                out = out - alpha * dEdg
        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class CRATE(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        mlp_hidden: int = 384 * 4,
        head: int = 8,
        use_cls_token: bool = True,
    ):
        super(CRATE, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_layer = n_layer
        self.n_recur = n_recur
        self.n_embd = n_embd
        self.emb = nn.Linear(f, n_embd)
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [CRATELayer(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(self.n_layer)]
        )

        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)
        for enc in self.enc:
            for _ in range(self.n_recur):
                out = enc(out)
        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class CRATET(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        mlp_hidden: int = 384 * 4,
        head: int = 8,
        use_cls_token: bool = True,
    ):
        super(CRATET, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_layer = n_layer
        self.n_recur = n_recur
        self.n_embd = n_embd
        self.emb = nn.Linear(f, n_embd)
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [CRATETLayer(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(self.n_layer)]
        )

        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)
        for enc in self.enc:
            for _ in range(self.n_recur):
                out = enc(out)
        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class HyperSETBasic(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        head: int = 8,
        mlp_hidden: int = 384 * 4,
        use_cls_token: bool = True,
        input_cond: bool = False,
        time_embed: int = 512,
    ):
        super(HyperSETBasic, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_recur = n_recur
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.emb = nn.Sequential(*[nn.LayerNorm(f), nn.Linear(f, n_embd), nn.LayerNorm(n_embd)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [HyperSETLayer(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(self.n_layer)]
        )

        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

        self.input_cond = input_cond
        self.time_embed = time_embed
        sinu_pos_emb = SinusoidalPosEmb(time_embed)

        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    sinu_pos_emb,
                    nn.Linear(time_embed, n_embd),
                    nn.RMSNorm(n_embd),
                    nn.GELU(),
                    nn.RMSNorm(n_embd),
                    nn.Linear(n_embd, n_embd),
                    nn.RMSNorm(n_embd),
                ),
                nn.Sequential(
                    nn.RMSNorm(n_embd),
                    nn.GELU(),
                    nn.Linear(n_embd, 2 * n_embd),
                ),
            ]
        )
        nn.init.constant_(self.time_mlp[0][1].weight, 0)
        nn.init.constant_(self.time_mlp[0][1].bias, 0)
        nn.init.constant_(self.time_mlp[0][5].weight, 0)
        nn.init.constant_(self.time_mlp[0][5].bias, 0)

        nn.init.constant_(self.time_mlp[1][2].weight, 0)
        nn.init.constant_(self.time_mlp[1][2].bias, 0)

        self.ln_inp = nn.RMSNorm(n_embd)

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)

        time = torch.arange(1, self.n_recur + 1, device=x.device)
        time_emb = self.time_mlp[0](time)
        inp = out
        for enc in self.enc:
            for recur in range(self.n_recur):
                c = time_emb[recur].expand_as(out)
                c = self.time_mlp[1](c + self.ln_inp(out) if not self.input_cond else c + self.ln_inp(inp))
                out = enc(out, c)

        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class HyperSET(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        head: int = 8,
        mlp_hidden: int = 384 * 4,
        use_cls_token: bool = True,
        input_cond: bool = False,
        time_embed: int = 512,
    ):
        super(HyperSET, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_recur = n_recur
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.emb = nn.Sequential(*[nn.LayerNorm(f), nn.Linear(f, n_embd), nn.LayerNorm(n_embd)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [HyperSETLayer(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(self.n_layer)]
        )

        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

        self.input_cond = input_cond
        self.time_embed = time_embed
        sinu_pos_emb = SinusoidalPosEmb(time_embed)

        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    sinu_pos_emb,
                    nn.Linear(time_embed, n_embd),
                    nn.RMSNorm(n_embd),
                    nn.GELU(),  # add skip connection
                    nn.RMSNorm(n_embd),
                    nn.Linear(n_embd, n_embd),
                    nn.RMSNorm(n_embd),
                ),
                nn.Sequential(
                    # nn.RMSNorm(n_embd),  # remove this rms norm
                    nn.GELU(),
                    nn.Linear(n_embd, 2 * n_embd),
                ),
            ]
        )
        nn.init.constant_(self.time_mlp[0][1].weight, 0)
        nn.init.constant_(self.time_mlp[0][1].bias, 0)
        nn.init.constant_(self.time_mlp[0][5].weight, 0)
        nn.init.constant_(self.time_mlp[0][5].bias, 0)

        nn.init.constant_(self.time_mlp[1][1].weight, 0)
        nn.init.constant_(self.time_mlp[1][1].bias, 0)

        self.ln_inp = nn.RMSNorm(n_embd)

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)

        time = torch.arange(1, self.n_recur + 1, device=x.device)
        time_emb = self.time_mlp[0](time)
        inp = out
        for enc in self.enc:
            for recur in range(self.n_recur):
                c = time_emb[recur].expand_as(out)
                emb = self.time_mlp[1](c + self.ln_inp(out) if not self.input_cond else c + self.ln_inp(inp))
                out = enc(out, emb)

        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class HyperSETLoRA(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        head: int = 8,
        mlp_hidden: int = 384 * 4,
        use_cls_token: bool = True,
        input_cond: bool = False,
        time_embed: int = 512,
        r: int = 4,
        alpha: int = 8,
    ):
        super(HyperSETLoRA, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_recur = n_recur
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.emb = nn.Sequential(*[nn.LayerNorm(f), nn.Linear(f, n_embd), nn.LayerNorm(n_embd)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if use_cls_token else None

        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [HyperSETLayerLoRA(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(self.n_layer)]
        )

        self.r = r
        self.alpha = alpha

        self.deltaW_attention_B = nn.ParameterList(
            [nn.Parameter(torch.randn(n_embd, r) * 0.02) for _ in range(self.n_recur)]
        )

        self.deltaW_attention_A = nn.ParameterList(
            [nn.Parameter(torch.randn(r, n_embd) * 0.02) for _ in range(self.n_recur)]
        )

        self.deltaW_ff_B = nn.ParameterList([nn.Parameter(torch.randn(n_embd, r) * 0.02) for _ in range(self.n_recur)])

        self.deltaW_ff_A = nn.ParameterList([nn.Parameter(torch.randn(r, n_embd) * 0.02) for _ in range(self.n_recur)])

        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

        self.input_cond = input_cond
        self.time_embed = time_embed
        sinu_pos_emb = SinusoidalPosEmb(time_embed)

        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    sinu_pos_emb,
                    nn.Linear(time_embed, n_embd),
                    nn.RMSNorm(n_embd),
                    nn.GELU(),
                    nn.RMSNorm(n_embd),
                    nn.Linear(n_embd, n_embd),
                    nn.RMSNorm(n_embd),
                ),
                nn.Sequential(
                    # nn.RMSNorm(n_embd),  # remove this rms norm
                    nn.GELU(),
                    nn.Linear(n_embd, 2 * n_embd),
                ),
            ]
        )
        nn.init.constant_(self.time_mlp[0][1].weight, 0)
        nn.init.constant_(self.time_mlp[0][1].bias, 0)
        nn.init.constant_(self.time_mlp[0][5].weight, 0)
        nn.init.constant_(self.time_mlp[0][5].bias, 0)

        nn.init.constant_(self.time_mlp[1][1].weight, 0)
        nn.init.constant_(self.time_mlp[1][1].bias, 0)

        self.ln_inp = nn.RMSNorm(n_embd)

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)

        time = torch.arange(1, self.n_recur + 1, device=x.device)
        time_emb = self.time_mlp[0](time)
        inp = out
        for enc in self.enc:
            for recur in range(self.n_recur):
                c = time_emb[recur].expand_as(out)
                c = self.time_mlp[1](c + self.ln_inp(out) if not self.input_cond else c + self.ln_inp(inp))
                out = enc(
                    out,
                    c,
                    (self.alpha / self.r) * self.deltaW_attention_B[recur] @ self.deltaW_attention_A[recur],
                    (self.alpha / self.r) * self.deltaW_ff_B[recur] @ self.deltaW_ff_A[recur],
                )

        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class HyperSETAlternativeFeedforward(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        head: int = 8,
        mlp_hidden: int = 384 * 4,
        use_cls_token: bool = True,
        input_cond: bool = False,
        time_embed: int = 512,
        ff: str = "relu",
    ):
        super(HyperSETAlternativeFeedforward, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_recur = n_recur
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.emb = nn.Sequential(*[nn.LayerNorm(f), nn.Linear(f, n_embd), nn.LayerNorm(n_embd)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if self.use_cls_token else None

        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [
                HyperSETAlterFeedforwardLayer(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head, ff=ff)
                for _ in range(self.n_layer)
            ]
        )
        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

        self.input_cond = input_cond
        self.time_embed = time_embed
        sinu_pos_emb = SinusoidalPosEmb(time_embed)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(time_embed, n_embd),
            nn.RMSNorm(n_embd),
            nn.GELU(),  # add skip connection
            nn.RMSNorm(n_embd),
            nn.Linear(n_embd, n_embd),
            nn.RMSNorm(n_embd),
        )

        nn.init.constant_(self.time_mlp[1].weight, 0)
        nn.init.constant_(self.time_mlp[1].bias, 0)
        nn.init.constant_(self.time_mlp[5].weight, 0)
        nn.init.constant_(self.time_mlp[5].bias, 0)

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)

        time = torch.arange(1, self.n_recur + 1, device=x.device)
        time_emb = self.time_mlp(time)
        inp = out
        for enc in self.enc:
            for recur in range(self.n_recur):
                c = time_emb[recur].expand_as(out)
                out = enc(out if not self.input_cond else inp, c)

        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class HyperSETAlternativeAttention(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        head: int = 8,
        mlp_hidden: int = 384 * 4,
        use_cls_token: bool = True,
        input_cond: bool = False,
        time_embed: int = 512,
        attention: str = "bisoftmax",
    ):
        super(HyperSETAlternativeAttention, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_recur = n_recur
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.emb = nn.Sequential(*[nn.LayerNorm(f), nn.Linear(f, n_embd), nn.LayerNorm(n_embd)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if self.use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [
                HyperSETAlterAttentionLayer(
                    n_embd,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                    head=head,
                    attention=attention,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))

        self.input_cond = input_cond
        self.time_embed = time_embed
        sinu_pos_emb = SinusoidalPosEmb(time_embed)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(time_embed, n_embd),
            nn.RMSNorm(n_embd),
            nn.GELU(),  # add skip connection
            nn.RMSNorm(n_embd),
            nn.Linear(n_embd, n_embd),
            nn.RMSNorm(n_embd),
        )
        nn.init.constant_(self.time_mlp[1].weight, 0)
        nn.init.constant_(self.time_mlp[1].bias, 0)
        nn.init.constant_(self.time_mlp[5].weight, 0)
        nn.init.constant_(self.time_mlp[5].bias, 0)

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)

        time = torch.arange(1, self.n_recur + 1, device=x.device)
        time_emb = self.time_mlp(time)
        inp = out
        for enc in self.enc:
            for recur in range(self.n_recur):
                c = time_emb[recur].expand_as(out)
                out = enc(out if not self.input_cond else inp, c)

        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class HyperSETFixedStepSize(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        n_layer: int = 1,
        n_recur: int = 7,
        n_embd: int = 384,
        mlp_hidden: int = 384 * 4,
        head: int = 8,
        use_cls_token: bool = True,
        stepsize: float = 1.0,
    ):
        super(HyperSETFixedStepSize, self).__init__()

        self.patch = patch
        self.use_cls_token = use_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch**2) + 1 if self.use_cls_token else (self.patch**2)
        self.num_tokens = num_tokens
        self.n_recur = n_recur
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.emb = nn.Sequential(*[nn.LayerNorm(f), nn.Linear(f, n_embd), nn.LayerNorm(n_embd)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd)) if self.use_cls_token else None
        self.pos_emb = sinusoidal_positional_embedding(self.num_tokens, self.n_embd).unsqueeze(0)

        self.enc = nn.ModuleList(
            [
                HyperSETFixedStepSizeLayer(n_embd, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
                for _ in range(self.n_layer)
            ]
        )
        self.fc = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_classes))
        self.step_size = stepsize

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.use_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb.to(x.device)

        for enc in self.enc:
            for _ in range(self.n_recur):
                out = enc(out, self.step_size)

        if self.use_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out

    @torch.no_grad()
    def compute_effective_rank(self, x):
        # Reshape to (B*H, N, d) for SVD
        print(x.shape)
        x_ = x.reshape(-1, x.size(-2), x.size(-1))
        print(x_.shape)
        s = torch.linalg.svdvals(x_)  # Shape: (B*H, min(N, d))
        normalized_s = s / (s.sum(dim=-1, keepdim=True) + 1e-10)  # Avoid division by zero
        normalized_s = torch.clamp(normalized_s, min=1e-10)  # Avoid log(0)
        entropy = (-normalized_s * normalized_s.log()).sum(dim=-1, keepdim=True)
        result = entropy.exp()
        print(result.shape)
        return result.reshape(x_.size(0) // x.size(1), x.size(1), 1)  # Reshape to (B, H, 1)

    @torch.no_grad()
    def compute_rank(self, x):
        x_ = x.reshape(-1, x.size(-2), x.size(-1))  # Reshape to (B*H, N, d)
        rank = torch.linalg.matrix_rank(x_, tol=1e-10).float().unsqueeze(-1)
        return rank.reshape(x_.size(0) // x.size(1), x.size(1), 1)  # Reshape to (B, H, 1)

    @torch.no_grad()
    def compute_average_angle(self, vectors):
        # Handle edge case: N=1 (no pairs to compare)
        if vectors.size(-2) <= 1:
            return torch.zeros(vectors.size(0), vectors.size(1), 1, device=vectors.device)

        # Normalize vectors
        normalized_vectors = torch.nn.functional.normalize(vectors, dim=-1)

        # Compute cosine similarity
        alignment = torch.matmul(normalized_vectors, normalized_vectors.transpose(-1, -2))
        alignment = torch.clamp(alignment, min=-1.0 + 1e-10, max=1.0 - 1e-10)

        # Compute angles in radians
        angles_rad = torch.acos(alignment)  # Shape: (B, H, N, N)

        # Mask out diagonal (self-angles) using upper triangular indices
        triu_mask = torch.triu(torch.ones_like(angles_rad), diagonal=1).bool()
        angles_rad_mask = angles_rad[triu_mask].view(vectors.size(0), vectors.size(1), -1)

        # Compute average angle in radians and convert to degrees
        average_angle_rad = angles_rad_mask.mean(dim=-1, keepdim=True)
        average_angle_deg = torch.rad2deg(average_angle_rad)

        return average_angle_deg


if __name__ == "__main__":
    # b, c, h, w = 1, 3, 32, 32
    b, c, h, w = 1, 3, 224, 224
    x = torch.randn(b, c, h, w)
    net = HyperSET(
        num_classes=1000,
        img_size=h,
        patch=16,
        dropout=0.0,
        n_layer=1,
        n_recur=12,
        n_embd=512,
        head=8,
        mlp_hidden=512,
        use_cls_token=True,
    ).cuda()

    # import torchsummary
    import torchinfo

    torchinfo.summary(net, (b, c, h, w))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {n_params / 1e6:.2f}M")
