"""
Modal training script for Pokemon pixel sequence generation.

Trains decoder-only transformers (GPT-2, LLaMA, Qwen2) on 20x20 Pokemon pixel
sequences using next-token prediction. Each pixel is a color index from a
vocabulary of ~167 classes. Sequences are 400 tokens long.

Usage:
    modal run train_modal.py --config gpt2_baseline
    modal run train_modal.py --config all
"""

import logging
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import modal

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------
vol = modal.Volume.from_name("pokemon-training-vol", create_if_missing=True)
VOLUME_PATH = "/vol"
CHECKPOINT_ROOT = f"{VOLUME_PATH}/checkpoints"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "tqdm",
)

app = modal.App("pokemon-transformer-training")

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainParams:
    epochs: int = 100
    lr: float = 5e-4
    batch_size: int = 16
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    grad_clip: float = 1.0


@dataclass
class ModelConfig:
    name: str = ""
    model_type: str = "gpt2"  # gpt2 | llama | qwen2
    model_params: Dict[str, Any] = field(default_factory=dict)
    train_params: TrainParams = field(default_factory=TrainParams)


def build_configs() -> Dict[str, ModelConfig]:
    """Return the seven preset configurations. vocab_size is set dynamically."""
    configs: Dict[str, ModelConfig] = {}

    # --- GPT-2 baseline ---
    configs["gpt2_baseline"] = ModelConfig(
        name="gpt2_baseline",
        model_type="gpt2",
        model_params=dict(
            activation_function="gelu_new",
            initializer_range=0.02,
            layer_norm_epsilon=1e-05,
            model_type="gpt2",
            n_layer=2,
            n_head=2,
            n_embd=64,
            n_positions=400,
            n_ctx=400,
            attn_pdrop=0.1,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
        ),
        train_params=TrainParams(
            epochs=50, lr=1e-3, batch_size=16,
            weight_decay=0.1, warmup_epochs=5, grad_clip=1.0,
        ),
    )

    # --- GPT-2 scaled ---
    configs["gpt2_scaled"] = ModelConfig(
        name="gpt2_scaled",
        model_type="gpt2",
        model_params=dict(
            activation_function="gelu_new",
            initializer_range=0.02,
            layer_norm_epsilon=1e-05,
            model_type="gpt2",
            n_layer=6,
            n_head=8,
            n_embd=256,
            n_positions=400,
            n_ctx=400,
            attn_pdrop=0.05,
            embd_pdrop=0.05,
            resid_pdrop=0.05,
        ),
        train_params=TrainParams(
            epochs=100, lr=5e-4, batch_size=16,
            weight_decay=0.05, warmup_epochs=10, grad_clip=1.0,
        ),
    )

    # --- LLaMA small ---
    configs["llama_small"] = ModelConfig(
        name="llama_small",
        model_type="llama",
        model_params=dict(
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=400,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        ),
        train_params=TrainParams(
            epochs=100, lr=5e-4, batch_size=16,
            weight_decay=0.05, warmup_epochs=10, grad_clip=1.0,
        ),
    )

    # --- LLaMA medium ---
    configs["llama_medium"] = ModelConfig(
        name="llama_medium",
        model_type="llama",
        model_params=dict(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            max_position_embeddings=400,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        ),
        train_params=TrainParams(
            epochs=150, lr=3e-4, batch_size=16,
            weight_decay=0.05, warmup_epochs=15, grad_clip=1.0,
        ),
    )

    # --- LLaMA large ---
    configs["llama_large"] = ModelConfig(
        name="llama_large",
        model_type="llama",
        model_params=dict(
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=8,
            num_attention_heads=8,
            max_position_embeddings=400,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        ),
        train_params=TrainParams(
            epochs=200, lr=2e-4, batch_size=16,
            weight_decay=0.05, warmup_epochs=20, grad_clip=1.0,
        ),
    )

    # --- Qwen2 small ---
    configs["qwen2_small"] = ModelConfig(
        name="qwen2_small",
        model_type="qwen2",
        model_params=dict(
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=400,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        ),
        train_params=TrainParams(
            epochs=100, lr=5e-4, batch_size=16,
            weight_decay=0.05, warmup_epochs=10, grad_clip=1.0,
        ),
    )

    # --- Qwen2 medium ---
    configs["qwen2_medium"] = ModelConfig(
        name="qwen2_medium",
        model_type="qwen2",
        model_params=dict(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=400,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
        ),
        train_params=TrainParams(
            epochs=150, lr=3e-4, batch_size=16,
            weight_decay=0.05, warmup_epochs=15, grad_clip=1.0,
        ),
    )

    return configs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_dataset_class():
    """Build and return the PixelSequenceDataset class inside the remote context."""
    import torch
    from torch.utils.data import Dataset as _Dataset

    class PixelSequenceDataset(_Dataset):
        def __init__(self, data: List[List[int]], mode: str = "train"):
            self.data = data
            self.mode = mode

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int):
            sequence = self.data[idx]
            if self.mode == "train":
                return (
                    torch.tensor(sequence[:-1], dtype=torch.long),
                    torch.tensor(sequence[1:], dtype=torch.long),
                )
            elif self.mode == "dev":
                return (
                    torch.tensor(sequence[:-160], dtype=torch.long),
                    torch.tensor(sequence[-160:], dtype=torch.long),
                )
            elif self.mode == "test":
                return torch.tensor(sequence, dtype=torch.long)
            raise ValueError(f"Invalid mode: {self.mode}")

    return PixelSequenceDataset


# ---------------------------------------------------------------------------
# LR schedule helper
# ---------------------------------------------------------------------------

def _cosine_with_warmup(warmup_steps: int, total_steps: int):
    """Return a lr_lambda for cosine schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(model_type: str, model_params: Dict[str, Any], num_classes: int):
    """Instantiate a fresh model from config."""
    from transformers import (
        AutoModelForCausalLM,
        GPT2Config,
        LlamaConfig,
        Qwen2Config,
    )

    # Common overrides
    common = dict(vocab_size=num_classes, pad_token_id=None, eos_token_id=None)

    if model_type == "gpt2":
        params = {**model_params, **common}
        config = GPT2Config(**params)
    elif model_type == "llama":
        params = {**model_params, **common}
        config = LlamaConfig(**params)
    elif model_type == "qwen2":
        params = {**model_params, **common}
        config = Qwen2Config(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = AutoModelForCausalLM.from_config(config)
    return model, config


# ---------------------------------------------------------------------------
# Training function (runs on Modal GPU)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: vol},
    timeout=7200,
)
def train_config(config_name: str) -> Dict[str, Any]:
    """Train a single configuration on a Modal T4 GPU."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from tqdm import tqdm

    # Re-configure logging inside the remote container
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        force=True,
    )
    remote_logger = logging.getLogger(f"train.{config_name}")

    remote_logger.info("Starting training for config: %s", config_name)

    # ---- Build config ----
    all_configs = build_configs()
    if config_name not in all_configs:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(all_configs.keys())}"
        )
    cfg = all_configs[config_name]
    tp = cfg.train_params

    # ---- Load data ----
    remote_logger.info("Loading dataset from HuggingFace...")
    pokemon_dataset = load_dataset("lca0503/ml2025-hw4-pokemon")
    colormap = list(
        load_dataset("lca0503/ml2025-hw4-colormap")["train"]["color"]
    )
    num_classes = len(colormap)
    remote_logger.info("Number of color classes: %d", num_classes)

    PixelSequenceDataset = _build_dataset_class()

    train_dataset = PixelSequenceDataset(
        pokemon_dataset["train"]["pixel_color"], mode="train"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=tp.batch_size, shuffle=True
    )

    dev_dataset = PixelSequenceDataset(
        pokemon_dataset["dev"]["pixel_color"], mode="dev"
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=tp.batch_size, shuffle=False
    )

    # ---- Build model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    remote_logger.info("Device: %s", device)

    model, model_config = _build_model(cfg.model_type, cfg.model_params, num_classes)
    model.to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    remote_logger.info("Model: %s | Trainable params: %s", cfg.model_type, f"{trainable_params:,}")

    # ---- Optimizer, scheduler, loss ----
    optimizer = optim.AdamW(
        model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay
    )

    steps_per_epoch = len(train_dataloader)
    total_steps = tp.epochs * steps_per_epoch
    warmup_steps = tp.warmup_epochs * steps_per_epoch

    scheduler = LambdaLR(
        optimizer, lr_lambda=_cosine_with_warmup(warmup_steps, total_steps)
    )

    criterion = nn.CrossEntropyLoss()

    # ---- Checkpoint directory ----
    ckpt_dir = os.path.join(CHECKPOINT_ROOT, config_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Resume from latest checkpoint if available ----
    latest_path = os.path.join(ckpt_dir, "latest.pth")
    start_epoch = 0
    best_loss = float("inf")
    best_accuracy = 0.0

    # Try latest checkpoint first, fall back to best_model if corrupted
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    for ckpt_path in [latest_path, best_model_path]:
        if not os.path.exists(ckpt_path):
            continue
        try:
            remote_logger.info("Attempting to resume from: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            best_loss = ckpt["best_loss"]
            remote_logger.info(
                "Resumed at epoch %d with best_loss=%.4f",
                start_epoch, best_loss,
            )
            break
        except Exception as e:
            remote_logger.warning("Failed to load %s: %s. Trying next.", ckpt_path, e)

    # ---- Training loop ----
    for epoch in range(start_epoch, tp.epochs):
        model.train()
        epoch_loss = 0.0

        for input_ids, labels in tqdm(
            train_dataloader,
            desc=f"[{config_name}] Epoch {epoch + 1}/{tp.epochs}",
        ):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids).logits.view(-1, num_classes)
            loss = criterion(outputs, labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tp.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        model.eval()
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                dev_dataloader,
                desc=f"[{config_name}] Validating epoch {epoch + 1}",
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = torch.ones_like(inputs)

                generated = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=400,
                )
                generated_last = generated[:, -160:]

                accuracy = (generated_last == labels).float().mean().item()
                total_accuracy += accuracy
                num_batches += 1

        avg_accuracy = total_accuracy / max(num_batches, 1)
        best_accuracy = max(best_accuracy, avg_accuracy)

        remote_logger.info(
            "[%s] Epoch %d/%d | Loss: %.4f | LR: %.6f | Recon Acc: %.4f",
            config_name, epoch + 1, tp.epochs,
            avg_epoch_loss, current_lr, avg_accuracy,
        )

        # ---- Save checkpoints ----
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": min(best_loss, avg_epoch_loss),
            "config_name": config_name,
            "model_type": cfg.model_type,
            "model_params": cfg.model_params,
            "num_classes": num_classes,
            "train_loss": avg_epoch_loss,
            "recon_accuracy": avg_accuracy,
        }

        # Always save latest
        torch.save(checkpoint_data, latest_path)

        # Save best if train loss improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_data["best_loss"] = best_loss
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(checkpoint_data, best_path)
            remote_logger.info(
                "[%s] New best model saved (loss=%.4f)", config_name, best_loss
            )

        # Commit volume so checkpoints persist
        vol.commit()

    # ---- Final summary ----
    remote_logger.info("=" * 60)
    remote_logger.info("[%s] Training complete.", config_name)
    remote_logger.info("  Best train loss : %.4f", best_loss)
    remote_logger.info("  Best recon acc  : %.4f", best_accuracy)
    remote_logger.info("  Total params    : %s", f"{trainable_params:,}")
    remote_logger.info("=" * 60)

    return {
        "config_name": config_name,
        "best_loss": best_loss,
        "best_accuracy": best_accuracy,
        "total_params": trainable_params,
    }


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    timeout=300,
)
def list_checkpoints() -> List[str]:
    """List all checkpoint files on the volume."""
    results = []
    for root, _dirs, files in os.walk(CHECKPOINT_ROOT):
        for f in files:
            results.append(os.path.join(root, f))
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

ALL_CONFIG_NAMES = [
    "gpt2_baseline",
    "gpt2_scaled",
    "llama_small",
    "llama_medium",
    "llama_large",
    "qwen2_small",
    "qwen2_medium",
]


@app.local_entrypoint()
def main(config: str = "gpt2_baseline"):
    """
    Entry point for training Pokemon transformer models on Modal.

    Args:
        config: Name of a single config preset, or "all" to train all 7
                configs in parallel on separate GPUs.
    """
    logger.info("Pokemon Transformer Training on Modal")
    logger.info("Requested config: %s", config)

    if config == "all":
        logger.info("Training all %d configs in parallel...", len(ALL_CONFIG_NAMES))
        results = list(train_config.map(ALL_CONFIG_NAMES))
        logger.info("=" * 60)
        logger.info("All training runs complete. Results:")
        for r in results:
            logger.info(
                "  %s | loss=%.4f | acc=%.4f | params=%s",
                r["config_name"],
                r["best_loss"],
                r["best_accuracy"],
                f"{r['total_params']:,}",
            )
        logger.info("=" * 60)
    else:
        if config not in ALL_CONFIG_NAMES:
            logger.error(
                "Unknown config '%s'. Available: %s",
                config,
                ALL_CONFIG_NAMES,
            )
            return
        logger.info("Training single config: %s", config)
        result = train_config.remote(config)
        logger.info(
            "Done: %s | loss=%.4f | acc=%.4f | params=%s",
            result["config_name"],
            result["best_loss"],
            result["best_accuracy"],
            f"{result['total_params']:,}",
        )

    # ---- Download best checkpoints to local disk ----
    logger.info("Downloading checkpoints from Modal Volume...")

    if config == "all":
        configs_to_download = ALL_CONFIG_NAMES
    else:
        configs_to_download = [config]

    for cfg_name in configs_to_download:
        remote_path = f"{CHECKPOINT_ROOT}/{cfg_name}/best_model.pth"
        local_dir = pathlib.Path(f"checkpoints/{cfg_name}")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "best_model.pth"

        try:
            # Read from volume via a subprocess-style approach:
            # We read the file bytes from the volume using a helper.
            data = _download_checkpoint.remote(remote_path)
            if data is not None:
                local_path.write_bytes(data)
                logger.info(
                    "Downloaded: %s -> %s", remote_path, local_path
                )
            else:
                logger.warning(
                    "Checkpoint not found on volume: %s", remote_path
                )
        except Exception:
            logger.exception(
                "Failed to download checkpoint for %s", cfg_name
            )


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    timeout=300,
)
def _download_checkpoint(remote_path: str) -> Optional[bytes]:
    """Read a checkpoint file from the Modal Volume and return its bytes."""
    vol.reload()
    if os.path.exists(remote_path):
        with open(remote_path, "rb") as f:
            return f.read()
    return None
