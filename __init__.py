#!/usr/bin/env python3
"""
DiffCLIP Module

This package implements a differential version of CLIP, featuring:
- DifferentialVisionTransformer: A vision transformer with differential attention
- DiffCLIP: A CLIP model using differential attention in both vision and text encoders

Main components:
- diff_attention.py: Implements DifferentialVisionTransformer
- diff_clip.py: Implements DiffCLIP

For more details, see the individual module docstrings.
"""

from .diff_attention import (
    RMSNorm, 
    DiffAttention, 
    LayerScale, 
    DiffBlock, 
    DifferentialVisionTransformer, 
    diff_vit_base_patch16_224
)

from .diff_clip import (
    DifferentialMultiheadAttention,
    DifferentialResidualAttentionBlock,
    DifferentialTextTransformer,
    DiffCLIP,
    DiffCLIP_VITB16,
)

__all__ = [
    'RMSNorm',
    'DiffAttention',
    'LayerScale',
    'DiffBlock',
    'DifferentialVisionTransformer',
    'diff_vit_base_patch16_224',
    'DifferentialMultiheadAttention',
    'DifferentialResidualAttentionBlock',
    'DifferentialTextTransformer',
    'DiffCLIP',
    'DiffCLIP_VITB16',
] 
