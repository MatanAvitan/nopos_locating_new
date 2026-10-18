"""Common model loading utilities for analysis scripts."""

import sys
from pathlib import Path

import torch


def get_nanogpt_path():
    """Get path to nanoGPT directory."""
    return Path(__file__).parent.parent.parent / "nanoGPT"


def add_nanogpt_to_path():
    """Add nanoGPT to Python path."""
    nanogpt_path = str(get_nanogpt_path())
    if nanogpt_path not in sys.path:
        sys.path.insert(0, nanogpt_path)


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint .pt file
        device: Device to load the model on

    Returns:
        Tuple of (model, config)
    """
    add_nanogpt_to_path()
    from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    model_args = checkpoint.get("model_args", {})
    config = TwoLayerMechanismConfig(**model_args)

    # Create and load model
    model = TwoLayerMechanismModel(config)

    # Handle state dict with _orig_mod prefix (from torch.compile)
    state_dict = checkpoint["model"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            unwrapped_state_dict[k[len("_orig_mod."):]] = v
        else:
            unwrapped_state_dict[k] = v

    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    model.eval()

    return model, config


def load_model_config(checkpoint_path: str, device: str = "cuda"):
    """Load just the model config from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint.get("model_args", {})
