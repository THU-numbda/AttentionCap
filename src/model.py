import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attention_type = config.attention_type
        if self.attention_type == 'symmetric':
            self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(F, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        if self.attention_type == 'symmetric':
            q, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if attention_mask is not None:
            reshaped_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if self.flash:
            flash_attn_mask = (reshaped_mask != 0) if attention_mask is not None else None
            if self.attention_type == 'symmetric':
                y = torch.nn.functional.scaled_dot_product_attention(q, q, v, attn_mask=flash_attn_mask, dropout_p=self.dropout if self.training else 0)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=flash_attn_mask, dropout_p=self.dropout if self.training else 0)
        else:
            if self.attention_type == 'symmetric':
                att = (q @ q.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                att = att.masked_fill(reshaped_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class LaplacianMatrixHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.RMSNorm(config.n_embd) if config.norm_type == "rmsnorm" else nn.LayerNorm(config.n_embd, bias=config.bias)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.head_activation = config.head_activation

    def forward(self, x, attention_mask=None):
        x = self.ln_f(x)
        q = self.c_attn(x)
        att = (q @ q.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        if self.head_activation == 'relu':
            att = F.relu(att)
        elif self.head_activation == 'exp':
            att = torch.exp(att)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            att = att.masked_fill(attention_mask == 0, 0.0)
        laplacian_matrix = torch.diag_embed(att.sum(dim=-1)) - att
        return laplacian_matrix


class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.exp_ratio * config.n_embd if config.exp_ratio is not None else 4 * config.n_embd
        h = int(math.ceil(hidden_dim * 2 / 3 / 8) * 8)
        self.gate = nn.Linear(config.n_embd, h, bias=config.bias)
        self.up = nn.Linear(config.n_embd, h, bias=config.bias)
        self.down = nn.Linear(h, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        a = F.silu(self.gate(x))
        b = self.up(x)
        x = a * b
        x = self.down(x)
        return self.drop(x)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.exp_ratio * config.n_embd if config.exp_ratio is not None else 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_type = config.norm_type
        if norm_type == "rmsnorm":
            self.ln_1 = nn.RMSNorm(config.n_embd)
            self.ln_2 = nn.RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.attn = SelfAttention(config)

        ffn_type = config.ffn_type
        if ffn_type == 'swiglu':
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class BlockNoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_type = config.norm_type
        if norm_type == "rmsnorm":
            self.ln_2 = nn.RMSNorm(config.n_embd)
        else:
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        ffn_type = config.ffn_type
        if ffn_type == 'swiglu':
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    input_dim: int = 4
    dropout: float = 0.0
    bias: bool = True
    exp_ratio: float = 4
    use_transformer: bool = False
    input_feature: str = 'linear'
    attention_type: str = 'standard'
    norm_type: str = 'layernorm'
    ffn_type: str = 'swiglu'
    head_activation: str = 'relu'
    head_mode: str = 'matrix'


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.head_mode not in ("matrix", "first_row"):
            raise ValueError(f"Unknown head mode: {config.head_mode}")
        self.config = config
        self.input_embedding = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)

        self.extra_embedding = nn.Embedding(10, config.n_embd)
        if config.use_transformer:
            self.transformer = nn.ModuleDict(dict(
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ))
        else:
            self.mlp = nn.ModuleList([BlockNoAttention(config) for _ in range(config.n_layer)])

        self.head = LaplacianMatrixHead(config)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _calculate_loss(self, pred, targets, attention_mask=None, loss_f=None):
        if self.config.head_mode == "first_row":
            if targets.ndim != 2:
                raise ValueError(f"Expected first-row targets, got shape {tuple(targets.shape)}")
            loss = ((pred[:, 0] - targets) / (targets[:, :1] + 1e-9)).square()
            if attention_mask is not None:
                loss *= attention_mask
            return loss.sum() / pred.size(0)

        if targets.ndim != 3:
            raise ValueError(f"Expected matrix targets, got shape {tuple(targets.shape)}")
        if loss_f == "laplacian":
            diagonal = torch.diagonal(targets, dim1=1, dim2=2)
            normalizer = (diagonal + 1e-9).pow(-0.5)
            loss = ((pred - targets) * normalizer.unsqueeze(-1) * normalizer.unsqueeze(-2)).square()
        else:
            loss = (pred - targets).square()
        if attention_mask is not None:
            loss = loss * attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
            return loss.sum() / attention_mask.sum()
        return loss.sum() / (pred.size(0) * pred.size(1))

    def forward(self, input, targets=None, attention_mask=None, loss_f=None):
        c = input.size(-1)
        if c != self.config.input_dim:
            assert c - 1 == self.config.input_dim, f"Input feature dimension {c} does not match config.input_dim {self.config.input_dim}"
            extra = self.extra_embedding(input[..., 0, -1].long())
            tok_emb = self.input_embedding(input[..., :c - 1]) + extra.unsqueeze(1)
        else:
            tok_emb = self.input_embedding(input)
        if hasattr(self, "transformer"):
            x = self.transformer.drop(tok_emb)
            for block in self.transformer.h:
                x = block(x, attention_mask=attention_mask)
        else:
            x = tok_emb
            for block in self.mlp:
                x = block(x)

        pred = self.head(x, attention_mask)

        if targets is not None:
            loss = self._calculate_loss(pred, targets, attention_mask, loss_f=loss_f)
        else:
            loss = None
        return pred, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
