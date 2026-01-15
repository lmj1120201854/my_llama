import torch
from torch import nn
import torch.nn.functional as F
from tools import apply_rotary_emb, repeat_kv, RMSNorm, precompute_freqs_cis
from config import ModelConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import math
import typing
from typing import Optional, Tuple

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads else cfg.n_heads
        assert cfg.n_heads % self.n_kv_heads == 0
        
        model_parallel_size = 1
        self.n_local_heads = cfg.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.d_model = cfg.dim
        self.d_head = self.d_model // self.n_local_heads

        self.wq = nn.Linear(self.d_model, self.d_head * self.n_local_heads, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_head * self.n_local_kv_heads, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_head * self.n_local_kv_heads, bias=False)
        self.wo = nn.Linear(self.d_head * self.n_local_heads, self.d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.o_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.full((1, 1, cfg.max_seq_len, cfg.max_seq_len), float("-inf"))
            mask = torch.tril(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        bs, slen, d_model = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, slen, self.n_local_heads, self.d_head)
        xk = xk.view(bs, slen, self.n_local_kv_heads, self.d_head)
        xv = xv.view(bs, slen, self.n_local_kv_heads, self.d_head)

        xq_rope, xk_rope = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        xk_rope = repeat_kv(xk_rope, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq_rope = xq_rope.transpose(1, 2) # bs nh slen dh
        xk_rope = xk_rope.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq_rope, xk_rope, xv, 
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            attn_score = torch.matmul(xq_rope, xk_rope.transpose(2, 3)) / math.sqrt(self.d_head)
            assert hasattr(self, "mask")
            attn_score = torch.nn.functional.softmax(attn_score + self.mask[:, :, :slen, :slen], dim=-1)
            attn_score = self.attn_dropout(attn_score)
            output = torch.matmul(attn_score, xv) # bs nh slen dh

        output = output.transpose(1, 2).contiguous().view(bs, slen, d_model)
        output = self.o_dropout(self.wo(output))
        return output

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if not hidden_dim:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        gate = F.silu(self.w1(x))
        hidden_state = gate * self.w3(x)
        output = self.w2(hidden_state)
        return self.dropout(output)
    
class Decoder(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = cfg.dim
        self.d_head = self.d_model // cfg.n_heads

        self.attention = Attention(cfg)
        self.ffn = MLP(cfg.dim, cfg.hidden_dim, cfg.multiple_of, cfg.dropout)

        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.ffn.forward(self.ffn_norm(h))
        return out
    
class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.n_layers
        
        # 各个模块
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.cfg.dim) # 词嵌入层
        self.dropout = nn.Dropout(cfg.dropout)  # dropout
        self.layers = nn.ModuleList() # decoder堆叠
        for layer_id in range(self.n_layers):
            self.layers.append(Decoder(layer_id, cfg))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps) # norm
        self.output_layer = nn.Linear(cfg.dim, self.vocab_size, bias=False) # 输出
        self.tok_embeddings.weight = self.output_layer.weight # 将收尾矩阵绑定为一个，节省空间

        # 位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # 权重初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight') or pn.endswith('wo.weight'): # TODO 这里可能有问题，w2？w3？
                torch.nn.init.normal_(p, mean=0, std=0.02/math.sqrt(2*cfg.n_layers))
        
        # 初始化一些记录
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor]=None) -> torch.Tensor:
        bs, slen = tokens.shape

        h = self.dropout(self.tok_embeddings(tokens))

        freqs_cos = self.freqs_cos[:slen]
        freqs_sin = self.freqs_sin[:slen]
        for decoder in self.layers:
            h = decoder(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            logits = self.output_layer(h)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), 
                targets.view(-1),
                ignore_index=0, # 忽略0号token，通常是padding
                reduce="none")
        else:
            # 推理阶段使用，只需要预测最后一个词的输出，也就是下一个词的logit
            logits = self.output_layer(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("last_loss", self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1, top_k=None):
        # TODO 未使用kv cache
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]
        
            if temperature == 0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == stop_id:
                break

            idx = torch.concat((idx, idx_next), dim=1)
        return idx[:, index:]

