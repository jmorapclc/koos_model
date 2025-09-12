"""
Utility functions for KOOS-PS prediction model.

This module contains helper functions for data processing, model utilities,
and other common operations.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
import yaml
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device

def save_config(config: Any, filepath: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        filepath: Path to save configuration
    """
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
    
    with open(filepath, 'w') as f:
        if filepath.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False)
        else:
            json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        if filepath.endswith('.json'):
            config = json.load(f)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    logger.info(f"Configuration loaded from {filepath}")
    return config

def create_directory_structure(base_path: str, structure: Dict[str, Any]):
    """
    Create directory structure.
    
    Args:
        base_path: Base directory path
        structure: Directory structure dictionary
    """
    base_path = Path(base_path)
    
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_directory_structure(str(path), content)
        else:
            path.mkdir(parents=True, exist_ok=True)

def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Class weights tensor
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    num_classes = len(unique_labels)
    
    weights = []
    for i in range(num_classes):
        weight = total_samples / (num_classes * counts[i])
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def normalize_metadata(metadata: np.ndarray, stats: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Normalize metadata features.
    
    Args:
        metadata: Metadata array
        stats: Optional normalization statistics
        
    Returns:
        Normalized metadata and statistics
    """
    if stats is None:
        stats = {
            'mean': np.mean(metadata, axis=0),
            'std': np.std(metadata, axis=0)
        }
    
    normalized_metadata = (metadata - stats['mean']) / (stats['std'] + 1e-8)
    
    return normalized_metadata, stats

def denormalize_metadata(normalized_metadata: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Denormalize metadata features.
    
    Args:
        normalized_metadata: Normalized metadata array
        stats: Normalization statistics
        
    Returns:
        Denormalized metadata
    """
    return normalized_metadata * stats['std'] + stats['mean']

def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE plot
    axes[0, 1].plot(history['train_mae'], label='Train MAE')
    axes[0, 1].plot(history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Training and Validation MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate plot
    axes[1, 0].plot(history['learning_rate'])
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    
    # Loss difference plot
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(loss_diff)
    axes[1, 1].set_title('Validation - Training Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()

def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size in different units.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'total_size_mb': size_all_mb,
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

def get_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input size for summary
        
    Returns:
        Model summary string
    """
    try:
        from torchsummary import summary
        summary_str = str(summary(model, input_size, device='cpu'))
    except ImportError:
        # Fallback summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary_str = f"""
Model Summary:
==============
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}

Architecture:
{model}
        """
    
    return summary_str

def save_model_artifacts(
    model: torch.nn.Module,
    config: Any,
    training_history: Dict[str, List],
    metrics: Dict[str, float],
    save_dir: str
):
    """
    Save all model artifacts.
    
    Args:
        model: Trained model
        config: Configuration object
        training_history: Training history
        metrics: Evaluation metrics
        save_dir: Directory to save artifacts
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), save_dir / 'model.pth')
    
    # Save configuration
    save_config(config, save_dir / 'config.json')
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save metrics
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model summary
    model_summary = get_model_summary(model)
    with open(save_dir / 'model_summary.txt', 'w') as f:
        f.write(model_summary)
    
    # Plot training history
    plot_training_history(training_history, save_dir / 'training_history.png')
    
    logger.info(f"Model artifacts saved to {save_dir}")

def load_model_artifacts(save_dir: str) -> Dict[str, Any]:
    """
    Load model artifacts.
    
    Args:
        save_dir: Directory containing artifacts
        
    Returns:
        Dictionary containing loaded artifacts
    """
    save_dir = Path(save_dir)
    
    artifacts = {}
    
    # Load model
    if (save_dir / 'model.pth').exists():
        artifacts['model_state_dict'] = torch.load(save_dir / 'model.pth')
    
    # Load configuration
    if (save_dir / 'config.json').exists():
        artifacts['config'] = load_config(save_dir / 'config.json')
    
    # Load training history
    if (save_dir / 'training_history.json').exists():
        with open(save_dir / 'training_history.json', 'r') as f:
            artifacts['training_history'] = json.load(f)
    
    # Load metrics
    if (save_dir / 'metrics.json').exists():
        with open(save_dir / 'metrics.json', 'r') as f:
            artifacts['metrics'] = json.load(f)
    
    return artifacts

def calculate_feature_importance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculate feature importance using permutation importance.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        feature_names: Names of features
        
    Returns:
        Dictionary of feature importance scores
    """
    model.eval()
    
    # Get baseline predictions
    baseline_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            outputs = model(images, metadata)
            baseline_predictions.extend(outputs['predictions'].cpu().numpy())
    
    baseline_predictions = np.array(baseline_predictions)
    baseline_mae = np.mean(np.abs(baseline_predictions - np.array([batch['target'] for batch in dataloader]).flatten()))
    
    # Calculate permutation importance
    feature_importance = {}
    
    for i, feature_name in enumerate(feature_names):
        permuted_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                metadata = batch['metadata'].to(device).clone()
                
                # Permute the feature
                metadata[:, i] = torch.randn_like(metadata[:, i])
                
                outputs = model(images, metadata)
                permuted_predictions.extend(outputs['predictions'].cpu().numpy())
        
        permuted_predictions = np.array(permuted_predictions)
        permuted_mae = np.mean(np.abs(permuted_predictions - np.array([batch['target'] for batch in dataloader]).flatten()))
        
        # Importance is the increase in MAE
        feature_importance[feature_name] = permuted_mae - baseline_mae
    
    return feature_importance
