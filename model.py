import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.attention_type = getattr(config, 'attention_type', 'standard')
        if self.attention_type == 'symmetric':
            self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.last_attn = None # for visualization, will be (B, nh, T, T)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.attention_type == 'symmetric':
            q, v  = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        # Reshape mask for broadcasting if it's provided
        if attention_mask is not None:
            # from (B, T) to (B, 1, 1, T)
            reshaped_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            flash_attn_mask = (reshaped_mask != 0) if attention_mask is not None else None
            # efficient attention using Flash Attention CUDA kernels
            if self.attention_type == 'symmetric':
                y = torch.nn.functional.scaled_dot_product_attention(q, q, v, attn_mask=flash_attn_mask, dropout_p=self.dropout if self.training else 0)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=flash_attn_mask, dropout_p=self.dropout if self.training else 0)
        else:
            # manual implementation of attention
            if self.attention_type == 'symmetric':
                att = (q @ q.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                att = att.masked_fill(reshaped_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            self.last_attn = att.detach()
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class LaplacianMatrixHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.RMSNorm(config.n_embd) if config.norm_type == "rmsnorm" else nn.LayerNorm(config.n_embd, bias=config.bias)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.head_activation = getattr(config, 'head_activation', 'relu')


    def forward(self, x, attention_mask=None):
        x = self.ln_f(x)
        q = self.c_attn(x)
        att = (q @ q.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1))) # (B, T, T)
        if self.head_activation == 'relu':
            att = F.relu(att)
        elif self.head_activation == 'exp':
            att = torch.exp(att)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # (B, 1, T) only last columns are zeroed out
            att = att.masked_fill(attention_mask == 0, 0.0)
        laplacian_matrix = torch.diag_embed(att.sum(dim=-1)) - att
        return laplacian_matrix

    
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.exp_ratio * config.n_embd if config.exp_ratio is not None else 4 * config.n_embd
        h = int(math.ceil(hidden_dim*2/3/8) * 8)  # round to 8 for tensor cores
        self.gate = nn.Linear(config.n_embd, h, bias=config.bias)  # W_gate
        self.up   = nn.Linear(config.n_embd, h, bias=config.bias)  # W_up
        self.down = nn.Linear(h, config.n_embd, bias=config.bias)  # W_down
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        a = F.silu(self.gate(x))   # SiLU(gate)
        b = self.up(x)             # value
        x = a * b                  # SwiGLU
        x = self.down(x)
        return self.drop(x)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.exp_ratio * config.n_embd if config.exp_ratio is not None else 4 * config.n_embd
        self.c_fc    = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
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
        norm_type = getattr(config, 'norm_type', "rmsnorm")
        if norm_type == "rmsnorm":
            self.ln_1 = nn.RMSNorm(config.n_embd)
            self.ln_2 = nn.RMSNorm(config.n_embd) 
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.attn = SelfAttention(config)
        
        ffn_type = getattr(config, 'ffn_type', 'swiglu')
        if ffn_type == 'swiglu':
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, extra_idx=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class BlockNoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_type = getattr(config, 'norm_type', "rmsnorm")
        if norm_type == "rmsnorm":
            self.ln_2 = nn.RMSNorm(config.n_embd) 
        else:
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        ffn_type = getattr(config, 'ffn_type', 'swiglu')
        if ffn_type == 'swiglu':
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, extra_idx=None):
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


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.input_embedding = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        
        self.extra_embedding = nn.Embedding(10, config.n_embd)
        if getattr(config, 'use_transformer', True):        
            self.transformer = nn.ModuleDict(dict(
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ))
        else:
            self.mlp = nn.ModuleList([BlockNoAttention(config) for _ in range(config.n_layer)])

        self.head = LaplacianMatrixHead(config)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _calculate_loss(self, pred, targets, attention_mask=None, loss_f=None):
        epsilon = 1e-9
        b = pred.size(0)
        t = pred.size(1)
        if len(targets.shape) == 3:
            # pred, targets are (b, t, t)
            if loss_f == "laplacian":
                diag = torch.diagonal(targets, dim1=1, dim2=2)
                normalizer = (diag+epsilon) ** (-0.5) # (b, t)
                matloss = (((pred - targets) * normalizer.unsqueeze(-1)) * normalizer.unsqueeze(-2))**2
            else:
                matloss = (pred - targets)**2
            if attention_mask is not None:
                # (b, t)
                matloss = (matloss * attention_mask.unsqueeze(-1)) * attention_mask.unsqueeze(-2)
                loss = (matloss.sum()) / attention_mask.sum()
            else:
                loss = (matloss.sum()) / (b*t)
        elif len(targets.shape) == 2:
            if len(pred.shape) == 3:
                pred = pred[..., 0, :]
            # pred, targets are (b, t)
            loss = ((pred - targets) / targets[..., 0:1])**2
            loss = loss.sum() / b
        else:
            raise ValueError(f"Unknown head_mode")
        return loss
    

    def forward(self, input, targets=None, attention_mask=None, loss_f=None):
        # input (b, t, 4) 
        # targets (b, t, t)
        # attention_mask (b, t) 1 for real tokens, 0 for padding
        b, t, c = input.size()
        extra_idx = input[..., 0, -1].long() # (b,)
        if c != self.config.input_dim:
            assert c-1 == self.config.input_dim, f"Input feature dimension {c} does not match config.input_dim {self.config.input_dim}"
            extra = self.extra_embedding(input[..., 0, -1].long()) # (b,) -> (b, n_embd)
            tok_emb = self.input_embedding(input[..., :c-1]) + extra.unsqueeze(1)
        else:
            tok_emb = self.input_embedding(input)
        if hasattr(self, "transformer"):
            x = self.transformer.drop(tok_emb)
            for id, block in enumerate(self.transformer.h):
                x = block(x, attention_mask=attention_mask, extra_idx=None) # (b, t, n_embd)
        else:
            x = tok_emb
            for id, block in enumerate(self.mlp):
                x = block(x, attention_mask=attention_mask, extra_idx=None)

        pred = self.head(x, attention_mask)

        if targets is not None:
            loss = self._calculate_loss(pred, targets, attention_mask, loss_f=loss_f)
        else:
            loss = None
        return pred, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer