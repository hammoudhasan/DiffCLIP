#!/usr/bin/env python3
"""
diff_clip.py

This module implements a differential version of CLIP's text encoder.
The idea is to replace the standard softmax multi-head attention with a
differential attention mechanism. The differential attention is defined as:

    DiffAttn(X) = (softmax(Q₁ K₁ᵀ / √d) − λ · softmax(Q₂ K₂ᵀ / √d)) · V

where the input X ∈ ℝ^(L×N×d) (with L = sequence length, N = batch size, and d = embed_dim)
is projected to query, key, and value as:
    
    Q = X W^Q,  K = X W^K,  V = X W^V

and Q and K are split along the head dimension:
    
    [Q₁; Q₂] ∈ ℝ^(L, N, 2·h_eff, d_head)   where h_eff = num_heads // 2 and d_head = d / num_heads.

A learnable scalar λ is computed via:
    
    λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init

The final multi-head differential attention is then computed as:

    MultiHeadDiffAttn(X) = Concat( LN( DiffAttn₁(X) ), …, LN( DiffAttn_h_eff(X) ) ) · W^O

The overall block in the text transformer is structured as:
    
    X' = X + DifferentialAttention(LayerNorm(X))
    X'' = X' + MLP(LayerNorm(X'))

This file defines:
  1. DifferentialMultiheadAttention – a drop-in replacement for nn.MultiheadAttention.
  2. DifferentialResidualAttentionBlock – a residual block using differential attention.
  3. DifferentialTextTransformer – a stack of differential residual attention blocks.
  4. DiffCLIP – a version of CLIP that uses DifferentialTextTransformer for text encoding.

References:
    - CLIP from OpenAI (modified from github.com/openai/CLIP)
    - Differential Transformers (see paper)
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_attention import diff_vit_base_patch16_224

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
# Utility Layers (LayerNorm and QuickGELU)
# ---------------------------------------------------------
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# ---------------------------------------------------------
# Differential Multihead Attention for Text
# ---------------------------------------------------------
class DifferentialMultiheadAttention(nn.Module):
    r"""
    Differential Multihead Attention for text inputs.

    This module implements the differential attention mechanism with the following steps:
    
      1. Given input X ∈ ℝ^(L×N×d) (L: sequence length, N: batch size, d: embed_dim),
         compute linear projections:
             Q = X W^Q,  K = X W^K,  V = X W^V.
      
      2. Permute the input to shape (N, L, d) so that we treat the batch dimension as B.
      
      3. Reshape Q and K to shape (B, L, 2·h_eff, d_head) and then transpose to
         (B, 2·h_eff, L, d_head), where h_eff = num_heads // 2 and d_head = d / num_heads.
      
      4. Reshape V to (B, L, h_eff, 2·d_head) and then transpose to (B, h_eff, L, 2·d_head).
      
      5. Compute the scaled dot-product attention scores for both splits:
             A₁ = softmax((Q₁ K₁ᵀ) / √d_head)
             A₂ = softmax((Q₂ K₂ᵀ) / √d_head)
      
      6. Compute a learnable scalar:
             λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init
      
      7. The differential attention output is:
             DiffAttn(X) = (A₁ − λ · A₂) · V
      
      8. After applying headwise RMSNorm and a final linear projection, the output is
         permuted back to (L, N, d).

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of attention heads (must be even).
        qkv_bias (bool): If True, add a bias to the Q, K, V projections.
        attn_drop (float): Dropout rate after softmax.
        proj_drop (float): Dropout rate after the output projection.
        lambda_init (float): Initial scalar for λ.
    """
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.effective_heads = num_heads // 2  # differential attention uses half the heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Linear layers for Q, K, V projections.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RMSNorm for headwise normalization; each head's output has dimension 2 * head_dim.
        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable lambda parameters (shared across heads).
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query, key, value (Tensor): Input tensors of shape (L, N, embed_dim),
                                        where L is sequence length and N is batch size.
            attn_mask (Tensor, optional): Additive attention mask of shape (L, L).

        Returns:
            Tensor of shape (L, N, embed_dim) after applying differential multi-head attention.
        """
        # Permute input from (L, N, embed_dim) to (N, L, embed_dim)
        x = query.transpose(0, 1)  # now (N, L, embed_dim)
        B, L, _ = x.shape

        # Compute projections.
        q = self.q_proj(x)  # (B, L, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape Q and K to (B, L, 2*h_eff, head_dim)
        q = q.view(B, L, 2 * self.effective_heads, self.head_dim)
        k = k.view(B, L, 2 * self.effective_heads, self.head_dim)
        # Reshape V to (B, L, h_eff, 2*head_dim)
        v = v.view(B, L, self.effective_heads, 2 * self.head_dim)

        # Transpose Q and K to (B, 2*h_eff, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # Transpose V to (B, h_eff, L, 2*head_dim)
        v = v.transpose(1, 2)

        # Scale Q.
        q = q * self.scaling

        # Compute raw attention scores: (B, 2*h_eff, L, L)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        # If an attention mask is provided, add it.
        if attn_mask is not None:
            # attn_mask is expected to be of shape (L, L)
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)

        # Compute attention probabilities.
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Reshape to separate the two halves: (B, h_eff, 2, L, L)
        attn_probs = attn_probs.view(B, self.effective_heads, 2, L, L)

        # Compute λ via re-parameterization.
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: subtract the second attention map scaled by λ.
        diff_attn = attn_probs[:, :, 0, :, :] - lambda_full * attn_probs[:, :, 1, :, :]  # (B, h_eff, L, L)

        # Compute weighted sum with V.
        out = torch.matmul(diff_attn, v)  # (B, h_eff, L, 2*head_dim)
        # Apply RMSNorm (headwise normalization) and scale by (1 - lambda_init).
        out = self.diff_norm(out) * (1 - self.lambda_init)

        # Concatenate heads: transpose to (B, L, h_eff, 2*head_dim) and then reshape to (B, L, embed_dim)
        out = out.transpose(1, 2).reshape(B, L, 2 * self.effective_heads * self.head_dim)
        # Final linear projection.
        out = self.out_proj(out)
        out = self.proj_drop(out)
        # Permute back to (L, N, embed_dim)
        out = out.transpose(0, 1)
        return out

# ---------------------------------------------------------
# Differential Residual Attention Block for Text
# ---------------------------------------------------------
class DifferentialResidualAttentionBlock(nn.Module):
    r"""
    Residual Attention Block using Differential Multihead Attention.

    This block first applies layer normalization to the input, then
    differential multihead attention, and adds the result to the input.
    Then it applies another layer normalization, a feed-forward MLP, and adds
    the result again.

    Equations:
        X' = X + DiffMultiheadAttention(LN(X))
        Y = X' + MLP(LN(X'))

    Args:
        d_model (int): Embedding dimension.
        n_head (int): Number of attention heads (must be even).
        attn_mask (Tensor, optional): Additive attention mask of shape (L, L).
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = DifferentialMultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        # DifferentialMultiheadAttention returns output of shape (L, N, d_model)
        return self.attn(x, x, x, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ---------------------------------------------------------
# Differential Text Transformer
# ---------------------------------------------------------
class DifferentialTextTransformer(nn.Module):
    r"""
    Transformer for text built from Differential Residual Attention Blocks.

    Args:
        width (int): Embedding dimension.
        layers (int): Number of transformer layers.
        heads (int): Number of attention heads (must be even).
        attn_mask (Tensor, optional): Additive attention mask of shape (L, L).
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            DifferentialResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# ---------------------------------------------------------
# DiffCLIP: Differential CLIP Model
# ---------------------------------------------------------
class DiffCLIP(nn.Module):
    r"""
    DiffCLIP implements a differential version of CLIP, where the text encoder is modified
    to use DifferentialTextTransformer.

    The overall architecture is similar to CLIP:
      - A vision encoder (can be any vision model, e.g., a differential ViT).
      - A text encoder that tokenizes text, adds positional embeddings, and passes through
        a stack of differential transformer blocks.
      - Final projections for image and text features.
    
    Args:
        embed_dim (int): Dimension of the joint embedding space.
        vision_width (int): Width (output dimension) of the vision encoder.
        vision_model (nn.Module): Vision encoder model.
        context_length (int): Maximum text sequence length.
        vocab_size (int): Vocabulary size for the text encoder.
        transformer_width (int): Embedding dimension of the text transformer.
        transformer_heads (int): Number of heads in the text transformer (must be even).
        transformer_layers (int): Number of layers in the text transformer.
    """
    def __init__(
        self,
        embed_dim: int,
        vision_width: int,
        vision_model: nn.Module,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        **kwargs,
    ):
        super().__init__()
        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        # Use DifferentialTextTransformer instead of the standard Transformer.
        self.transformer = DifferentialTextTransformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Initialize transformer parameters.
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            # For our differential attention blocks, initialize the linear layers.
            nn.init.normal_(block.attn.q_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.k_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.v_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)  # c_fc layer
            nn.init.normal_(block.mlp[2].weight, std=proj_std)  # c_proj layer

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # Create a causal attention mask for text.
        # The mask is of shape (context_length, context_length) with -inf above the diagonal.
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection
        return x

    def encode_text(self, text):
        # text: (batch_size, context_length)
        x = self.token_embedding(text)  # (batch_size, context_length, transformer_width)
        x = x + self.positional_embedding
        # Permute to (context_length, batch_size, transformer_width)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # Permute back to (batch_size, context_length, transformer_width)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        # Extract the features at the position of the end-of-text token (assumed to be the max token index in each sequence).
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),
        }


def DiffCLIP_VITB16(**kwargs):
    """
    Factory function to build DiffCLIP with a ViT-B/16 vision encoder.
    This function creates a vision model using the differential vision transformer 
    "diff_vit_base_patch16_224" and then builds DiffCLIP.
    
    Args:
        **kwargs: Additional keyword arguments.
    
    Returns:
        DiffCLIP model.
    """
    # Create a vision model using the differential vision transformer
    vision_model = diff_vit_base_patch16_224(num_classes=0)
    model = DiffCLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,   # must be even
        transformer_layers=12,
        **kwargs,
    )
    return model


# ---------------------------------------------------------
# Main test function
# ---------------------------------------------------------
if __name__ == "__main__":
    # Create dummy inputs.
    dummy_image = torch.randn(2, 3, 224, 224)  # batch of 2 images
    # Create dummy text tokens (e.g., 77 tokens per sequence). For simplicity, we simulate tokens as random integers.
    dummy_text = torch.randint(low=0, high=49408, size=(2, 77))
    
    # Instantiate DiffCLIP using the DiffCLIP_VITB16 factory function.
    model = DiffCLIP_VITB16()
    model.eval()
    
    with torch.no_grad():
        outputs = model(dummy_image, dummy_text)
    
    print("DiffCLIP output keys:", outputs.keys())
    print("Image embed shape:", outputs["image_embed"].shape)
    print("Text embed shape:", outputs["text_embed"].shape)
    print("Logit scale:", outputs["logit_scale"])

    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")
    
    # Note: Original CLIP comparison removed as it's not relevant for the public release

