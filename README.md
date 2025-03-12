<div align="center">
<h1>DiffCLIP: Differential Attention Meets CLIP</h1>
<p style="font-size: 1.5em; font-weight: bold;">Hasan Abed Al Kader Hammoud and Bernard Ghanem</p>
<p style="font-size: 1.2em; font-style: italic;">King Abdullah University of Science and Technology</p>


  <p>
<a href="https://arxiv.org/abs/2503.06626">
    <img src="https://img.shields.io/badge/arXiv-2503.06626-B31B1B.svg" alt="arXiv">
</a>
<a href="https://huggingface.co/collections/hammh0a/diffclip-67cd8d3b7c6e6ea1cc26cd93"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-yellow" alt="Hugging Face Collection"></a>
  </p>
</div>

<div align="center">
  <img src="./assets/images.png" width="100%">
</div>

## Abstract

We propose DiffCLIP, a novel vision-language model that extends the differential attention mechanism to CLIP architectures. Differential attention was originally developed for large language models to amplify relevant context while canceling out noisy information. In this work, we integrate this mechanism into CLIP's dual encoder (image and text) framework. With minimal additional parameters, DiffCLIP achieves superior performance on image-text understanding tasks. Across zero-shot classification, retrieval, and robustness benchmarks, DiffCLIP consistently outperforms baseline CLIP models. Notably, these gains come with negligible computational overhead, demonstrating that differential attention can significantly enhance multi-modal representations without sacrificing efficiency.

## What is Differential Attention?

Differential attention, proposed in [Differential Transformer](https://arxiv.org/abs/2410.05258), computes the difference between two attention maps:

```
DiffAttn(X) = (softmax(Q₁K₁ᵀ/√d) − λ · softmax(Q₂K₂ᵀ/√d)) · V
```

where the query and key projections are split as `[Q₁; Q₂] = X·Wᵠ` and `[K₁; K₂] = X·Wᵏ`, and λ is a learnable parameter. This mechanism allows the model to capture complementary information by explicitly modeling the differences between attention patterns, leading to richer multimodal representations.

## Structure

The repository contains two main components:

1. **DifferentialVisionTransformer** (in `diff_attention.py`): A Vision Transformer modified to use differential attention.

2. **DiffCLIP** (in `diff_clip.py`): A CLIP model that uses differential attention in both its vision and text encoders.

## How to Use

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DiffCLIP.git
cd DiffCLIP

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from diff_clip import DiffCLIP_VITB16

# Create model
model = DiffCLIP_VITB16()

# Process image and text
image = torch.randn(1, 3, 224, 224)
text = torch.randint(0, 49408, (1, 77))  # Tokenized text

# Get embeddings
with torch.no_grad():
    outputs = model(image, text)

print(outputs["image_embed"].shape)  # Should be [1, 512]
print(outputs["text_embed"].shape)   # Should be [1, 512]
```

### Zero-Shot Classification

You can use the provided `test_models.py` script to perform zero-shot classification:

```bash
# Download the model from Hugging Face and test on a COCO image
python test_models.py
```

This will:
1. Download the DiffCLIP_ViTB16_CC12M model from Hugging Face
2. Load a sample image from COCO
3. Perform zero-shot classification
4. Print the top-5 predicted classes

## References

```
@misc{hammoud2025diffclipdifferentialattentionmeets,
      title={DiffCLIP: Differential Attention Meets CLIP}, 
      author={Hasan Abed Al Kader Hammoud and Bernard Ghanem},
      year={2025},
      eprint={2503.06626},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06626}, 
}
```
