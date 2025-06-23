"""
RWKV "x051a" model - does not require custom CUDA kernel to train :)

References:
https://github.com/BlinkDL/RWKV-LM

Inference:
Always fast, and VRAM will not grow, because RWKV does not need KV cache.

Training:
Because we are not using custom CUDA kernel here, training is slightly slower than gpt+flash_attn when ctxlen is short.
Training becomes faster than gpt+flash_attn when ctxlen is long.
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from einops import rearrange
from embed import DataEmbedding

import math, warnings
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 添加RWKV-7所需的辅助函数
def lerp(a, b, x):
    """Linear interpolation: a + (b - a) * x"""
    return a + (b - a) * x

class LoRAMLP(nn.Module):
    """Low-rank MLP for dynamic parameter generation"""
    def __init__(self, input_dim, hidden_dim, bias=False):
        super().__init__()
        self.A = nn.Linear(input_dim, hidden_dim, bias=False)
        self.B = nn.Linear(hidden_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim)) if bias else None
        
    def forward(self, x):
        result = self.B(self.A(x))
        if self.bias is not None:
            result = result + self.bias
        return result



# @dataclass
# class RWKVConfig:
#     block_size: int = 1024
#     vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
#     n_layer: int = 2
#     n_head: int = 2
#     n_embd: int = 128
#     dropout: float = 0.0
#     bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster



@dataclass
class RWKVConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    # 添加RWKV-7特有的配置
    head_size_divisor: int = 1  # 确保head_size是整数

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RWKV_TimeMix_v7(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.layer_id = layer_id
        self.eps = 1e-8
        
        # Token shift参数
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            
            self.mu_r = nn.Parameter(torch.full((config.n_embd,), 0.5 * ratio_1_to_almost0))
            self.mu_k = nn.Parameter(torch.full((config.n_embd,), ratio_1_to_almost0))
            self.mu_v = nn.Parameter(torch.full((config.n_embd,), ratio_1_to_almost0 + 0.3 * ratio_0_to_1))
            self.mu_g = nn.Parameter(torch.full((config.n_embd,), 0.5 * ratio_1_to_almost0))
            self.mu_a = nn.Parameter(torch.full((config.n_embd,), 0.5 * ratio_1_to_almost0))
            self.mu_d = nn.Parameter(torch.full((config.n_embd,), 0.5 * ratio_1_to_almost0))
        
        # LoRA MLPs
        lora_dim = max(32, config.n_embd // 16)
        self.decay_lora = LoRAMLP(config.n_embd, lora_dim, bias=True)
        self.iclr_lora = LoRAMLP(config.n_embd, lora_dim, bias=True)
        self.gate_lora = LoRAMLP(config.n_embd, lora_dim, bias=False)
        self.value_residual_lora = LoRAMLP(config.n_embd, lora_dim//2, bias=True) if layer_id > 0 else None
        
        # 线性层
        self.W_receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # RWKV-7关键参数
        self.removal_key_multiplier = nn.Parameter(torch.randn(config.n_embd) * 0.1)  # ξ
        self.iclr_mix_amt = nn.Parameter(torch.full((config.n_embd,), 0.5))  # α  
        self.bonus_multiplier = nn.Parameter(torch.ones(config.n_embd))  # ρ
        
        # GroupNorm
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd, eps=1e-5 * self.n_head)
        self.dropout = nn.Dropout(config.dropout)
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x, vprime_0=None):
        B, T, C = x.size()
        H, N = self.n_head, self.head_size
        
        # Token shift
        x_shifted = self.time_shift(x)
        
        # Token mixing
        x_receptance = lerp(x, x_shifted, self.mu_r)
        x_key = lerp(x, x_shifted, self.mu_k)
        x_value = lerp(x, x_shifted, self.mu_v)
        x_gate = lerp(x, x_shifted, self.mu_g)
        x_iclr = lerp(x, x_shifted, self.mu_a)
        x_decay = lerp(x, x_shifted, self.mu_d)
        
        # 基础变换
        r = self.W_receptance(x_receptance)
        k = self.W_key(x_key)
        vprime = self.W_value(x_value)
        
        # 动态参数生成
        gate = torch.sigmoid(self.gate_lora(x_gate))
        iclr = torch.sigmoid(self.iclr_lora(x_iclr))
        decay_precursor = torch.tanh(self.decay_lora(x_decay))
        
        # Value residual learning
        if self.layer_id == 0:
            v = vprime_0 = vprime
        else:
            value_residual_gate = torch.sigmoid(self.value_residual_lora(x_value))
            v = lerp(vprime, vprime_0, value_residual_gate)
        
        # Decay计算
        decay = torch.exp(-math.exp(-0.5) * torch.sigmoid(decay_precursor))
        
        # Key processing
        removal_k = k * self.removal_key_multiplier
        replacement_k = k * lerp(torch.ones_like(iclr), iclr, self.iclr_mix_amt)
        
        # 重塑
        r = r.view(B, T, H, N)
        removal_k = removal_k.view(B, T, H, N)
        replacement_k = replacement_k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        decay = decay.view(B, T, H, N)
        iclr = iclr.view(B, T, H, N)
        
        # 正确的归一化
        removal_k_norm = F.normalize(removal_k, dim=-1)
        
        # WKV计算 - 修正版本
        wkv_state = torch.zeros(B, H, N, N, device=x.device, dtype=x.dtype)
        output = torch.zeros(B, T, H, N, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            decay_t = decay[:, t]
            iclr_t = iclr[:, t]
            removal_k_norm_t = removal_k_norm[:, t]
            replacement_k_t = replacement_k[:, t]
            v_t = v[:, t]
            r_t = r[:, t]
            
            # Transition matrix构建
            diag_decay = torch.diag_embed(decay_t)  # [B, H, N, N]
            
            # κ̂_t^T(at ⊙ κ̂_t)
            weighted_removal = iclr_t * removal_k_norm_t
            removal_outer = torch.einsum('bhi,bhj->bhij', removal_k_norm_t, weighted_removal)
            
            G_t = diag_decay - removal_outer
            
            # 状态更新：正确的矩阵乘法
            wkv_state = torch.bmm(G_t.view(B*H, N, N), wkv_state.view(B*H, N, N)).view(B, H, N, N)
            
            # 添加新信息
            update = torch.einsum('bhi,bhj->bhij', v_t, replacement_k_t)
            wkv_state = wkv_state + update
            
            # 输出计算
            output[:, t] = torch.einsum('bhij,bhj->bhi', wkv_state, r_t)
        
        
        bonus_scalar = torch.sum(r * (self.bonus_multiplier.view(H, N) * replacement_k), dim=-1, keepdim=True)
        bonus = bonus_scalar * v
        output = output + bonus
        
        # 归一化和输出
        output = output.view(B, T, C)
        output = self.ln_x(output.view(B*T, C)).view(B, T, C)
        output = output * gate
        output = self.dropout(self.W_output(output))
        
        return output, vprime_0






class RWKV_ChannelMix_v7(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        
        # Token shift参数
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            self.mu_k = nn.Parameter(torch.full((config.n_embd,), ratio_1_to_almost0))
        
        # 4x扩展，移除gating
        self.W_key = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.W_value = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
    
    def forward(self, x):
        x_shifted = self.time_shift(x)
        xk = lerp(x, x_shifted, self.mu_k)
        
        k = self.W_key(xk)
        k = F.relu(k).square()  # ReLU^2激活
        return self.W_value(k)



class Block_v7(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.tmix = RWKV_TimeMix_v7(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cmix = RWKV_ChannelMix_v7(config, layer_id)

    def forward(self, x, vprime_0=None):
        # TimeMix with value residual
        tmix_out, vprime_0 = self.tmix(self.ln_1(x), vprime_0)
        x = x + tmix_out
        
        # ChannelMix
        x = x + self.cmix(self.ln_2(x))
        
        return x, vprime_0



class RWKV7_TS_3(nn.Module):
    def __init__(self, configs, device):
        super(RWKV7_TS_3, self).__init__()
        self.patch_size = configs.patch_size
        self.pretrain = getattr(configs, 'pretrain', False)
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        rwkv_config = RWKVConfig(
            n_layer=configs.gpt_layers,
            n_head=configs.n_heads,
            n_embd=configs.d_model
        )
        self.rwkv = nn.ModuleList([Block_v7(rwkv_config, i) for i in range(rwkv_config.n_layer)])
        print("rwkv = {}".format(self.rwkv))

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        for layer in (self.rwkv, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forward(self, x, itr=None):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        vprime_0 = None
        for block in self.rwkv:
            outputs, vprime_0 = block(outputs, vprime_0)

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs