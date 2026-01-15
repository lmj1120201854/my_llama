import torch
from torch import nn
import models
from models import *
from config import ModelConfig
from tools import apply_rotary_emb, precompute_freqs_cis

x = torch.randn((1, 50, 16, 768//16))

# # 测试RMSNorm
# from tools import RMSNorm
# norm = RMSNorm(768, 1e-5)
# output = norm(x)
# print(output.shape)

# # 测试repeat kv
# from tools import repear_kv
# output = repear_kv(x, 2)
# print(output.shape)

# # 测试rope,precompute
# from tools import precompute_freqs_cis
# c, s = precompute_freqs_cis(768//16, 50)
# print(c.shape, s.shape)

# # 测试rope
# from tools import apply_rotary_emb, precompute_freqs_cis
# xq = torch.randn((4, 50, 8, 128))
# xk = torch.randn((4, 50, 8, 128))
# freq_cos, freq_sin = precompute_freqs_cis(128, 50)
# print(freq_cos.shape)
# xq_, xk_ = apply_rotary_emb(xq, xk, freq_cos, freq_sin)
# print(xq_.shape, xk_.shape)

# # 测试Attention
# import models
# from models import *
# from config import ModelConfig
# from tools import apply_rotary_emb, precompute_freqs_cis
# cfg = ModelConfig()
# attention = Attention(cfg)
# x = torch.randn((4, 50, 768))
# freq_cos, freq_sin = precompute_freqs_cis(768 // cfg.n_heads, 50)
# out = attention(x, freq_cos, freq_sin)
# print(out.shape)

# # 测试MLP
# cfg = ModelConfig()
# mlp = MLP(cfg.dim, cfg.hidden_dim, cfg.multiple_of, cfg.dropout)
# x = torch.randn(1, 50, cfg.dim)
# output = mlp(x)
# print(output.shape)

# # 测试Decoder
# cfg = ModelConfig()
# decoderlayer = Decoder(0, cfg)
# dim = cfg.dim
# seq_len = 50
# x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]
# freqs_cos, freqs_sin = precompute_freqs_cis(dim//cfg.n_heads, seq_len)
# out = decoderlayer(x, freqs_cos, freqs_sin)
# print(out.shape) # 形状和输入的x一样 [batch_size, seq_len, dim]

# # 测试Transformer 
# cfg = ModelConfig()
# x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
# model = Transformer(cfg)
# num_params = sum(p.numel() for p in model.parameters())
# print('Number of parameters:', num_params)
# out = model(x)
# print(out.logits.shape) # [batch_size, 1, vocab_size]

