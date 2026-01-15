import torch
from torch import nn
from typing import Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(dim)
        )
    
    def _norm(self, x):
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True)
            + self.eps
            )

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :] # 在头维度前加一个新维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float=10000):
    # 1 / (theta ** (2di / d_model))
    # exp(- (2di / d_model) * log(theta))
    freq = torch.exp(- (2 * torch.arange(0, dim, 2) / dim) * torch.log(torch.tensor(theta)))
    t = torch.arange(0, end, device=freq.device)
    freq = torch.outer(t, freq)
    # print(freq.shape, t.shape)
    # freq = t * freq
    freq_cos = torch.cos(freq)
    freq_sin = torch.sin(freq)
    return freq_cos, freq_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # slen d
    # x的第二维度与最后一维度保持不变，其他的为1，换句话说拓展了f维度
    # 注意，unbind并不会保留最后一维度，所以是ndim-1
    shape = [d if i==1 or i==ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    # bs slen nhead dh/2 2 ->bs slen nhead dh/2, bs slen nhead dh/2

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r) # 1 slen 1 dh/2
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_i)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_i * freqs_cos + xq_r * freqs_sin
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_i * freqs_cos + xk_r * freqs_sin
    # bs slen nh dh/2 * 1 slen 1 dh/2 -> bs slen nh dh/2

    xq_rotary = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_rotary = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    # bs slen nhead dh/2 -> bs slen nhead dh/2 2 -> bs slen nhead dh
    return xq_rotary.type_as(xq), xk_rotary.type_as(xk)

