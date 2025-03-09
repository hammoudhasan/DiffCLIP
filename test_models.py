#!/usr/bin/env python3
"""
test_models.py

This script demonstrates zero-shot prediction with DiffCLIP. It:
1. Downloads the DiffCLIP_ViTB16_CC12M model from Hugging Face
2. Loads an image from COCO dataset
3. Performs zero-shot classification

Usage:
    python test_models.py
"""

import os
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download
from tokenizer import SimpleTokenizer

# Import the DiffCLIP models
from diff_clip import DiffCLIP_VITB16


def download_model():
    """
    Download the DiffCLIP_ViTB16_CC12M model from Hugging Face to a local folder.
    Returns the path to the checkpoint file.
    """
    print("Downloading model from Hugging Face...")
    model_id = "hammh0a/DiffCLIP_ViTB16_CC12M"
    local_dir = "./DiffCLIP_ViTB16_CC12M"
    
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the checkpoint file
    checkpoint_path = hf_hub_download(
        repo_id=model_id,
        filename="checkpoint_best.pt",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Model downloaded to {checkpoint_path}")
    return checkpoint_path


def load_model(checkpoint_path):
    """
    Load the DiffCLIP model and checkpoint.
    """
    print("Loading model...")
    # Create a model instance
    model = DiffCLIP_VITB16()
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # If the model was saved with DataParallel, we need to handle that
    if list(checkpoint["state_dict"].keys())[0].startswith("module."):
        # Create a new state_dict without the 'module.' prefix
        new_state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        load_status = model.load_state_dict(new_state_dict)
    else:
        load_status = model.load_state_dict(checkpoint["state_dict"])
    
    print(f"Model loaded with status: {load_status}")
    return model


def load_image_from_coco():
    """
    Load a sample image from COCO dataset.
    """
    print("Loading sample image from COCO...")
    # A sample image URL from COCO
    coco_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    # Download the image
    response = requests.get(coco_image_url)
    img = Image.open(BytesIO(response.content))
    
    # Resize and preprocess for the model
    img = img.convert("RGB").resize((224, 224))
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()  # (3, 224, 224)
    img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0), img  # Return tensor and PIL image


def zero_shot_prediction(model, image_tensor, tokenizer, classes):
    """
    Perform zero-shot prediction on an image with the provided model.
    """
    print("Performing zero-shot prediction...")
    model.eval()
    
    # Create text prompts
    prompts = [f"a photo of a {label}" for label in classes]
    
    # Tokenize text
    text_tokens = tokenizer(prompts)
    
    # Put everything on the same device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    text_tokens = text_tokens.to(device)
    
    # Get image and text features
    with torch.no_grad():
        outputs = model(image_tensor, text_tokens)
        image_features = outputs["image_embed"]
        text_features = outputs["text_embed"]
        logit_scale = outputs["logit_scale"]
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity scores
    similarity = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
    
    # Get top predictions
    values, indices = similarity[0].topk(min(5, len(classes)))
    
    # Return prediction results
    predictions = [(classes[idx], values[i].item()) for i, idx in enumerate(indices)]
    return predictions


def main():
    # Define some classes for zero-shot prediction
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", 
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", 
        "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", 
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
        "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Download and load the model
    checkpoint_path = download_model()
    model = load_model(checkpoint_path)
    
    # Load a sample image
    image_tensor, pil_image = load_image_from_coco()
    
    # Save the image for reference
    # pil_image.save("coco_sample.jpg")
    # print("Sample image saved as 'coco_sample.jpg'")
    
    # Perform zero-shot prediction
    predictions = zero_shot_prediction(model, image_tensor, tokenizer, coco_classes)
    
    # Print results
    print("\nZero-shot prediction results:")
    for i, (label, score) in enumerate(predictions):
        print(f"{i+1}. {label}: {score:.4f}")


if __name__ == "__main__":
    main() 