#!/usr/bin/env python3
"""
diff_attention.py

This module implements a Differential Attention Vision Transformer.
The key idea is to replace the standard softmax attention with a
differential attention mechanism as described in the paper:

    DiffAttn(X) = (softmax(Q₁K₁ᵀ/√d) − λ · softmax(Q₂K₂ᵀ/√d)) · V

where the query and key projections are split as:
    [Q₁; Q₂] = X Wᵠ,   [K₁; K₂] = X Wᵏ,
and V = X Wᵛ.

The learnable scalar λ is re-parameterized as:
    λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init

The multi-head formulation uses "effective heads" computed as:
    h_effective = (num_heads // 2)
with the per-head dimension d_head = d_model / num_heads. Note that the value
projection is not split (it remains of dimension d_model), so that its per-head shape
is (2·d_head), aligning with the fact that Q and K are split into two parts.

The overall transformer block is:
    Y = X + DropPath(LayerScale(DiffAttention(LN(X))))
    X' = Y + DropPath(LayerScale(MLP(LN(Y))))

The DifferentialVisionTransformer class below inherits from timm's VisionTransformer
and replaces its transformer blocks with blocks using Differential Attention.
A registration function diff_vit_base_patch16_224 is provided with the same default
parameters as ViT-Base (patch16, 224).

References:
    - Vision Transformer: https://arxiv.org/abs/2010.11929
    - Differential Transformers: (see paper)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Register model decorator: Try to import timm's version; if unavailable, use a dummy.
try:
    from timm.models.registry import register_model
except ImportError:
    def register_model(fn):
        return fn

# Import timm's VisionTransformer and common layers.
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath, Mlp

# ---------------------------------------------------------
# RMSNorm (as used in the differential attention paper)
# ---------------------------------------------------------
class RMSNorm(nn.Module):
    r"""
    RMSNorm normalizes the input tensor by its root-mean-square (RMS) value.

    Given an input x ∈ ℝ^(...×d), it computes:

        RMS(x) = sqrt(mean(x², dim=-1, keepdim=True) + ε)
        output = x / RMS(x)

    Optionally, a learnable weight is applied if elementwise_affine is True.

    Args:
        dim (int): Dimension to normalize.
        eps (float): A value added for numerical stability.
        elementwise_affine (bool): If True, multiply by a learnable weight.
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}'

# ---------------------------------------------------------
# Differential Attention Module
# ---------------------------------------------------------
class DiffAttention(nn.Module):
    r"""
    Differential Attention Module.

    Given an input tensor X ∈ ℝ^(B×N×d_model), we first compute the linear projections:

        Q = X Wᵠ,   K = X Wᵏ,   V = X Wᵛ

    The queries and keys are then reshaped and split into two parts:
        Q → [Q₁; Q₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
        K → [K₁; K₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
    with h_effective = num_heads // 2 and d_head = d_model / num_heads.

    The value projection is reshaped to:
        V ∈ ℝ^(B, N, h_effective, 2·d_head)

    We then compute two attention maps:
        A₁ = softmax((Q₁ K₁ᵀ) / √d_head)
        A₂ = softmax((Q₂ K₂ᵀ) / √d_head)

    A learnable scalar λ is computed via:
        λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init

    Finally, the differential attention output is:
        DiffAttn(X) = (A₁ − λ · A₂) · V

    The per-head outputs are then normalized headwise with RMSNorm and projected back to d_model.

    Args:
        dim (int): Embedding dimension (d_model).
        num_heads (int): Number of heads in the original transformer (must be even).
        qkv_bias (bool): If True, add a bias term to the Q, K, V projections.
        attn_drop (float): Dropout probability after softmax.
        proj_drop (float): Dropout probability after the output projection.
        lambda_init (float): Initial constant for lambda re-parameterization.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = dim
        self.num_heads = num_heads           # original number of heads
        self.effective_heads = num_heads // 2  # differential attention operates on half as many heads
        self.head_dim = dim // num_heads       # per-head dimension
        self.scaling = self.head_dim ** -0.5

        # Linear projections for Q, K, V: mapping from dim → dim.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)  # final output projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RMSNorm for headwise normalization on outputs (each head's output has dimension 2·head_dim)
        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable lambda parameters (shared across all heads)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, d_model).

        Returns:
            Tensor of shape (B, N, d_model) after applying differential attention.
        """
        B, N, _ = x.shape

        # Compute Q, K, V projections.
        q = self.q_proj(x)  # shape: (B, N, d_model)
        k = self.k_proj(x)  # shape: (B, N, d_model)
        v = self.v_proj(x)  # shape: (B, N, d_model)

        # Reshape Q and K into (B, N, 2 * h_effective, head_dim)
        q = q.view(B, N, 2 * self.effective_heads, self.head_dim)
        k = k.view(B, N, 2 * self.effective_heads, self.head_dim)
        # Reshape V into (B, N, h_effective, 2 * head_dim)
        v = v.view(B, N, self.effective_heads, 2 * self.head_dim)

        # Transpose to bring head dimension forward.
        # q, k: (B, 2 * h_effective, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # v: (B, h_effective, N, 2 * head_dim)
        v = v.transpose(1, 2)

        # Scale Q.
        q = q * self.scaling

        # Compute raw attention scores: (B, 2 * h_effective, N, N)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        # Compute attention probabilities.
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Reshape to separate the two halves: (B, h_effective, 2, N, N)
        attn_probs = attn_probs.view(B, self.effective_heads, 2, N, N)

        # Compute lambda via re-parameterization.
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: subtract the second attention map scaled by lambda_full.
        diff_attn = attn_probs[:, :, 0, :, :] - lambda_full * attn_probs[:, :, 1, :, :]  # shape: (B, h_effective, N, N)

        # Multiply the differential attention weights with V.
        attn_output = torch.matmul(diff_attn, v)  # shape: (B, h_effective, N, 2 * head_dim)

        # Apply RMSNorm (headwise normalization) and scale by (1 - lambda_init)
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        # Concatenate heads: reshape from (B, h_effective, N, 2 * head_dim) → (B, N, 2 * h_effective * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, 2 * self.effective_heads * self.head_dim)

        # Final linear projection.
        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)
        return x_out

# ---------------------------------------------------------
# LayerScale module (optional scaling of sublayer outputs)
# ---------------------------------------------------------
class LayerScale(nn.Module):
    r"""
    LayerScale scales the output of a sublayer by a learnable parameter.

    Equation:
        Output = x * γ

    Args:
        dim (int): Dimension of the sublayer output.
        init_values (float): Initial value for scaling parameter γ.
        inplace (bool): Whether to perform the multiplication in-place.
    """
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

# ---------------------------------------------------------
# Transformer Block with Differential Attention
# ---------------------------------------------------------
class DiffBlock(nn.Module):
    r"""
    Transformer Block with Differential Attention.

    The block consists of two main sublayers:

      1. Differential Attention sublayer:
         Y = X + DropPath(LayerScale(DiffAttention(LayerNorm(X))))

      2. MLP sublayer:
         X' = Y + DropPath(LayerScale(MLP(LayerNorm(Y))))

    Equations:
        Y = X + DropPath(LS₁(DiffAttention(LN₁(X))))
        X' = Y + DropPath(LS₂(MLP(LN₂(Y))))

    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of heads in the original transformer (must be even).
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        qkv_bias (bool): If True, add bias in Q, K, V projections.
        drop (float): Dropout probability.
        attn_drop (float): Attention dropout probability.
        drop_path (float): Stochastic depth rate.
        init_values (float or None): Initial value for LayerScale. If None, LayerScale is disabled.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        lambda_init (float): Initial lambda value for differential attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, lambda_init=0.8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DiffAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                  lambda_init=lambda_init)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# ---------------------------------------------------------
# Differential Vision Transformer
# ---------------------------------------------------------
class DifferentialVisionTransformer(VisionTransformer):
    r"""
    Vision Transformer with Differential Attention.

    This model is a modification of the standard VisionTransformer (timm)
    where the self-attention mechanism is replaced with Differential Attention.

    In each transformer block, the attention sublayer is computed as:

        DiffAttn(X) = (softmax(Q₁K₁ᵀ/√d_head) − λ · softmax(Q₂K₂ᵀ/√d_head)) · V

    with the λ re-parameterization:
        λ = exp(λ_{q1}⋅λ_{k1}) − exp(λ_{q2}⋅λ_{k2}) + λ_init

    The overall block structure is:
        Y = X + DropPath(LayerScale(DiffAttention(LayerNorm(X))))
        X' = Y + DropPath(LayerScale(MLP(LayerNorm(Y))))

    Args:
        All arguments are as in timm's VisionTransformer.
        lambda_init (float): Initial lambda value for differential attention.
    """
    def __init__(self, *args, lambda_init=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        depth = kwargs.get('depth', 12)
        embed_dim = self.embed_dim  # d_model from VisionTransformer
        num_heads = kwargs.get('num_heads', 12)
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', 0.)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        init_values = kwargs.get('init_values', None)
        norm_layer = kwargs.get('norm_layer', None) or nn.LayerNorm
        act_layer = kwargs.get('act_layer', None) or nn.GELU

        # Create stochastic depth schedule.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            blocks.append(
                DiffBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    init_values=init_values,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    lambda_init=lambda_init  # same for all blocks (or can vary with depth)
                )
            )
        self.blocks = nn.Sequential(*blocks)
        if hasattr(self, 'norm'):
            self.norm = norm_layer(embed_dim)

# ---------------------------------------------------------
# Model Registration: Differential ViT-Base (Patch16, 224)
# ---------------------------------------------------------
@register_model
def diff_vit_base_patch16_224(pretrained: bool = False, **kwargs) -> DifferentialVisionTransformer:
    """
    Differential ViT-Base (ViT-B/16) with Differential Attention.
    
    The defaults are set to match the original ViT-Base (patch16, 224):
        - patch_size = 16
        - embed_dim = 768
        - depth = 12
        - num_heads = 12
        - lambda_init = 0.8

    Args:
        pretrained (bool): If True, load pretrained weights (not implemented here).
        **kwargs: Additional keyword arguments.

    Returns:
        DifferentialVisionTransformer model.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        lambda_init=0.8,
    )
    # Merge additional kwargs with defaults.
    model = DifferentialVisionTransformer(**dict(model_args, **kwargs))
    if pretrained:
        # Code to load pretrained weights can be added here.
        pass
    return model

# ---------------------------------------------------------
# Main test function
# ---------------------------------------------------------
if __name__ == "__main__":
    # Instantiate the Differential ViT-Base (patch16, 224) with default parameters.
    model = diff_vit_base_patch16_224()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print("Differential ViT-Base output shape:", output.shape)

    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")

    from timm.models.vision_transformer import vit_base_patch16_224
    model = vit_base_patch16_224()
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")

