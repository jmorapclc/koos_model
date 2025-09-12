#!/usr/bin/env python3
# model/config/config.py
"""
Configuration file for KOOS-PS Prediction CNN Model

This module contains all hyperparameters, paths, and configuration settings
for the medical X-ray image classification model.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Paths
    image_dir: str = "data/img_repo"
    csv_file: str = "data/HALS_Dataset_v1.csv"
    output_dir: str = "model/outputs"
    
    # Image preprocessing
    image_size: int = 224
    input_channels: int = 3
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    
    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]  # ImageNet normalization
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]

@dataclass
class ModelConfig:
    """Configuration for the CNN model architecture."""
    
    # Backbone architecture
    backbone: str = "densenet121"  # densenet121, resnet50, efficientnet_b0
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Model architecture
    num_metadata_features: int = 7  # Sex, Age, BMI, Side, Type_of_TKA, Patellar_Replacement, Preoperative_KOOS-PS
    hidden_dim: int = 512
    dropout_rate: float = 0.3
    
    # Output
    num_classes: int = 1  # Regression task for KOOS-PS score
    output_activation: str = "linear"  # linear, sigmoid, tanh
    
    # Advanced architecture options
    use_attention: bool = True
    use_metadata_fusion: bool = True
    use_skip_connections: bool = True

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Basic augmentations
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.1
    rotation_degrees: float = 15.0
    translation_ratio: float = 0.1
    scale_ratio: float = 0.1
    
    # Advanced augmentations
    use_elastic_transform: bool = True
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0
    
    use_color_jitter: bool = True
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    saturation_factor: float = 0.2
    hue_factor: float = 0.1
    
    use_gaussian_noise: bool = True
    noise_std: float = 0.01
    
    use_histogram_equalization: bool = True
    use_clahe: bool = True  # Contrast Limited Adaptive Histogram Equalization
    
    # Medical imaging specific
    use_lung_segmentation: bool = False  # Requires additional preprocessing
    use_anatomical_crop: bool = True

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine, step, plateau, onecycle
    
    # Optimizer
    optimizer: str = "adamw"  # adam, adamw, sgd
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Loss function
    loss_function: str = "mse"  # mse, mae, huber, focal
    loss_weights: Optional[Dict[str, float]] = None
    
    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    lr_scheduler_params: Dict[str, Any] = None
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        if self.lr_scheduler_params is None:
            self.lr_scheduler_params = {
                "T_max": self.num_epochs,
                "eta_min": self.learning_rate * 0.01
            }

@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    
    # Regression metrics
    primary_metric: str = "mae"  # mae, mse, rmse, r2
    secondary_metrics: List[str] = None
    
    # Additional metrics
    use_confidence_intervals: bool = True
    confidence_level: float = 0.95
    
    # Visualization
    save_predictions: bool = True
    save_attention_maps: bool = True
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["mse", "rmse", "r2", "mape"]

@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "model/outputs/training.log"
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "model/outputs/tensorboard"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "koos-ps-prediction"
    wandb_entity: str = None
    
    # Model checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "model/outputs/checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5  # Save every N epochs

@dataclass
class SystemConfig:
    """Configuration for system and hardware."""
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    
    # Memory
    max_memory_usage: float = 0.9
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False

class Config:
    """Main configuration class that combines all configs."""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.augmentation = AugmentationConfig()
        self.training = TrainingConfig()
        self.metrics = MetricsConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories."""
        # Create base output directory
        os.makedirs(self.data.output_dir, exist_ok=True)
        
        # Note: Specific iteration directories will be created during training
        # This ensures we don't create empty timestamped directories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "augmentation": self.augmentation.__dict__,
            "training": self.training.__dict__,
            "metrics": self.metrics.__dict__,
            "logging": self.logging.__dict__,
            "system": self.system.__dict__
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, params in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in params.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

# Default configuration instance
config = Config()
