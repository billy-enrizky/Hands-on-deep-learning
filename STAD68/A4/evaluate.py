"""
Evaluation script for Pokemon image generation using decoder-only transformers.

Loads a trained checkpoint, generates images from the test set, and computes:
  - FID (Frechet Inception Distance) using torchmetrics
  - PDR (Pokemon Detection Rate) using a trained CNN classifier

Usage:
    uv run evaluate.py --checkpoint checkpoints/llama_medium/best_model.pth
    uv run evaluate.py --checkpoint checkpoints/llama_medium/best_model.pth --do_sample --temperature 0.8 --top_k 50
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    LlamaConfig,
    Qwen2Config,
    set_seed,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 20
TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 400
CONTEXT_PIXELS = 240
PREDICT_PIXELS = 160

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PixelSequenceDataset(Dataset):
    """Dataset for pixel-color sequences in train / dev / test modes."""

    def __init__(self, data: List[List[int]], mode: str = "train") -> None:
        self.data = data
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        sequence = self.data[idx]

        if self.mode == "train":
            input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
            labels = torch.tensor(sequence[1:], dtype=torch.long)
            return input_ids, labels
        elif self.mode == "dev":
            input_ids = torch.tensor(sequence[:-PREDICT_PIXELS], dtype=torch.long)
            labels = torch.tensor(sequence[-PREDICT_PIXELS:], dtype=torch.long)
            return input_ids, labels
        elif self.mode == "test":
            input_ids = torch.tensor(sequence, dtype=torch.long)
            return input_ids
        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Choose from 'train', 'dev', or 'test'."
            )


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def pixel_to_image(pixel_color: List[int], colormap: List[List[int]]) -> Image.Image:
    """Convert a list of pixel color indices to a 20x20 PIL RGB image."""
    pixel_color = list(pixel_color)
    while len(pixel_color) < TOTAL_PIXELS:
        pixel_color.append(0)
    pixel_data = [colormap[pixel] for pixel in pixel_color]
    image_array = np.array(pixel_data, dtype=np.uint8).reshape(
        IMAGE_SIZE, IMAGE_SIZE, 3
    )
    return Image.fromarray(image_array)


def images_to_tensor(images: List[Image.Image], size: int = 299) -> torch.Tensor:
    """Convert PIL images to uint8 tensor (N, 3, H, W) resized for InceptionV3."""
    tensors = []
    for img in images:
        arr = (
            torch.tensor(np.array(img, dtype=np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        resized = F.interpolate(
            arr, size=(size, size), mode="bilinear", align_corners=False
        )
        tensors.append(resized.clamp(0, 255).to(torch.uint8))
    return torch.cat(tensors, dim=0)


def save_image_grid(
    images: List[Image.Image], save_path: str, rows: int = 6, cols: int = 16
) -> None:
    """Save a grid of images to a file using matplotlib."""
    num_images = min(rows * cols, len(images))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Image grid saved to %s", save_path)


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

CONFIG_MAP = {
    "gpt2": GPT2Config,
    "llama": LlamaConfig,
    "qwen2": Qwen2Config,
}


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a model from a training checkpoint and return (model, checkpoint_dict)."""
    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type: str = checkpoint["model_type"]
    model_params: Dict[str, Any] = checkpoint["model_params"]
    num_classes: int = checkpoint["num_classes"]
    config_name: str = checkpoint.get("config_name", "unknown")

    logger.info("Model type: %s", model_type)
    logger.info("Config name: %s", config_name)
    logger.info("Num classes (vocab size): %d", num_classes)
    logger.info("Model params: %s", json.dumps(model_params, indent=2))

    if model_type not in CONFIG_MAP:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {list(CONFIG_MAP.keys())}"
        )

    config_cls = CONFIG_MAP[model_type]
    config = config_cls(
        vocab_size=num_classes,
        pad_token_id=None,
        eos_token_id=None,
        **model_params,
    )

    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    return model, checkpoint


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_sequences(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> List[List[int]]:
    """Generate completed pixel sequences from partial test inputs."""
    logger.info("Starting generation...")
    logger.info(
        "  do_sample=%s, temperature=%.2f, top_k=%d, top_p=%.2f",
        do_sample,
        temperature,
        top_k,
        top_p,
    )

    results: List[List[int]] = []

    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_dataloader):
            inputs = inputs.to(device)
            attention_mask = torch.ones_like(inputs)

            generated_outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=TOTAL_PIXELS,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            batch_results = generated_outputs.cpu().numpy().tolist()
            results.extend(batch_results)
            logger.info(
                "  Batch %d/%d complete (%d sequences so far)",
                batch_idx + 1,
                len(test_dataloader),
                len(results),
            )

    logger.info("Generation complete. Total sequences: %d", len(results))
    return results


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------


def compute_fid(
    real_images: List[Image.Image], generated_images: List[Image.Image]
) -> float:
    """Compute FID between real and generated image sets using torchmetrics."""
    logger.info(
        "Computing FID (real=%d images, generated=%d images)...",
        len(real_images),
        len(generated_images),
    )

    real_tensors = images_to_tensor(real_images)
    gen_tensors = images_to_tensor(generated_images)

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False)
    fid_metric.update(real_tensors, real=True)
    fid_metric.update(gen_tensors, real=False)
    fid_score = fid_metric.compute().item()

    logger.info("FID Score: %.4f", fid_score)
    return fid_score


# ---------------------------------------------------------------------------
# PDR computation (Pokemon Detection Rate)
# ---------------------------------------------------------------------------


class PokemonClassifier(nn.Module):
    """Simple CNN classifier for Pokemon vs non-Pokemon images (20x20 input)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_negatives(real_images: List[Image.Image]) -> List[np.ndarray]:
    """
    Generate negative samples by destroying image structure.

    Uses three strategies (cycled 1/3 each):
      0 - Random pixel permutation: shuffle all pixel positions
      1 - Block shuffle: divide 20x20 into 5x5 blocks and shuffle them
      2 - Random noise: uniform random pixel values
    """
    negatives: List[np.ndarray] = []
    for i, img in enumerate(real_images):
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3, 20, 20)
        strategy = i % 3

        if strategy == 0:
            # Random pixel permutation
            flat = arr.reshape(3, -1)
            perm_idx = np.random.permutation(flat.shape[1])
            neg = flat[:, perm_idx].reshape(3, IMAGE_SIZE, IMAGE_SIZE)
        elif strategy == 1:
            # Block shuffle (5x5 blocks)
            block_size = 5
            blocks = []
            for r in range(0, IMAGE_SIZE, block_size):
                for c in range(0, IMAGE_SIZE, block_size):
                    blocks.append(
                        arr[:, r : r + block_size, c : c + block_size].copy()
                    )
            np.random.shuffle(blocks)
            neg = np.zeros_like(arr)
            idx = 0
            for r in range(0, IMAGE_SIZE, block_size):
                for c in range(0, IMAGE_SIZE, block_size):
                    neg[:, r : r + block_size, c : c + block_size] = blocks[idx]
                    idx += 1
        else:
            # Random noise
            neg = np.random.rand(3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

        negatives.append(neg)
    return negatives


def train_classifier(
    real_images: List[Image.Image],
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> PokemonClassifier:
    """Train a Pokemon vs non-Pokemon CNN classifier."""
    logger.info("Training Pokemon classifier (epochs=%d, lr=%.4f)...", epochs, lr)

    # Prepare positive samples
    pos_data = [
        np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        for img in real_images
    ]
    # Generate negative samples
    neg_data = generate_negatives(real_images)

    cls_x = torch.tensor(np.array(pos_data + neg_data))
    cls_y = torch.tensor([1] * len(pos_data) + [0] * len(neg_data))

    cls_loader = DataLoader(
        TensorDataset(cls_x, cls_y), batch_size=batch_size, shuffle=True
    )

    classifier = PokemonClassifier().to(device)
    cls_opt = optim.Adam(classifier.parameters(), lr=lr)
    cls_loss_fn = nn.CrossEntropyLoss()

    classifier.train()
    for ep in range(epochs):
        correct = 0
        total = 0
        for bx, by in cls_loader:
            bx, by = bx.to(device), by.to(device)
            cls_opt.zero_grad()
            out = classifier(bx)
            loss = cls_loss_fn(out, by)
            loss.backward()
            cls_opt.step()
            correct += (out.argmax(1) == by).sum().item()
            total += by.size(0)
        if (ep + 1) % 5 == 0:
            logger.info(
                "  Classifier epoch %d/%d | Accuracy: %.4f",
                ep + 1,
                epochs,
                correct / total,
            )

    logger.info("Classifier training complete.")
    return classifier


def compute_pdr(
    classifier: PokemonClassifier,
    generated_images: List[Image.Image],
    device: torch.device,
) -> Tuple[int, int, float]:
    """
    Compute Pokemon Detection Rate using the trained classifier.

    Returns:
        (pokemon_count, total, pdr_ratio)
    """
    classifier.eval()
    pokemon_count = 0
    total = len(generated_images)

    with torch.no_grad():
        for img in generated_images:
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            t = torch.tensor(arr).unsqueeze(0).to(device)
            if classifier(t).argmax(1).item() == 1:
                pokemon_count += 1

    pdr = pokemon_count / total if total > 0 else 0.0
    logger.info("PDR: %d/%d = %.4f", pokemon_count, total, pdr)
    return pokemon_count, total, pdr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pokemon pixel-sequence transformer model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.pth).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling. 0 = disabled (default: 0).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus (top-p) sampling threshold (default: 1.0).",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Enable sampling-based generation (default: greedy).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for test DataLoader (default: 16).",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=False,
        help="Save a grid of generated images to the results directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load checkpoint and reconstruct model
    # ------------------------------------------------------------------
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    config_name = checkpoint.get("config_name", "unknown")

    logger.info("Checkpoint epoch: %d", checkpoint.get("epoch", -1))
    logger.info("Checkpoint train_loss: %.4f", checkpoint.get("train_loss", -1.0))
    logger.info(
        "Checkpoint recon_accuracy: %.4f", checkpoint.get("recon_accuracy", -1.0)
    )

    # ------------------------------------------------------------------
    # 2. Load dataset and colormap
    # ------------------------------------------------------------------
    logger.info("Loading dataset and colormap from HuggingFace...")
    pokemon_dataset = load_dataset("lca0503/ml2025-hw4-pokemon")
    colormap = list(
        load_dataset("lca0503/ml2025-hw4-colormap")["train"]["color"]
    )
    num_classes = len(colormap)
    logger.info("Colormap loaded: %d colors", num_classes)

    # Prepare test dataloader
    test_dataset = PixelSequenceDataset(
        pokemon_dataset["test"]["pixel_color"], mode="test"
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    logger.info("Test dataset: %d samples", len(test_dataset))

    # ------------------------------------------------------------------
    # 3. Generate images from test set
    # ------------------------------------------------------------------
    logger.info("--- Generation Settings ---")
    logger.info("  do_sample: %s", args.do_sample)
    logger.info("  temperature: %.2f", args.temperature)
    logger.info("  top_k: %d", args.top_k)
    logger.info("  top_p: %.2f", args.top_p)

    generated_sequences = generate_sequences(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Convert sequences to images
    generated_images = [
        pixel_to_image(seq, colormap) for seq in generated_sequences
    ]
    logger.info("Generated %d images", len(generated_images))

    # Build real (training) images for FID
    train_images = [
        pixel_to_image(data["pixel_color"], colormap)
        for data in pokemon_dataset["train"]
    ]
    logger.info("Real training images: %d", len(train_images))

    # ------------------------------------------------------------------
    # 4. Compute FID
    # ------------------------------------------------------------------
    fid_score = compute_fid(train_images, generated_images)

    # ------------------------------------------------------------------
    # 5. Train classifier and compute PDR
    # ------------------------------------------------------------------
    classifier = train_classifier(train_images, device)
    pokemon_count, total, pdr_score = compute_pdr(
        classifier, generated_images, device
    )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_summary = {
        "config_name": config_name,
        "model_type": checkpoint.get("model_type", "unknown"),
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch", -1),
        "train_loss": checkpoint.get("train_loss", -1.0),
        "recon_accuracy": checkpoint.get("recon_accuracy", -1.0),
        "generation_settings": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "fid_score": fid_score,
        "pdr": {
            "pokemon_count": pokemon_count,
            "total": total,
            "ratio": pdr_score,
        },
    }

    results_path = os.path.join(results_dir, f"{config_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Optionally save image grid
    if args.save_images:
        grid_path = os.path.join(results_dir, f"{config_name}_generated.png")
        save_image_grid(generated_images, grid_path, rows=6, cols=16)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Config:      %s", config_name)
    logger.info("  Model type:  %s", checkpoint.get("model_type", "unknown"))
    logger.info("  FID Score:   %.4f", fid_score)
    logger.info("  PDR:         %d/%d = %.4f", pokemon_count, total, pdr_score)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
