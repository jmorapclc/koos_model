"""
Experiment management utilities for tracking training iterations.

This module provides functionality to create timestamped output directories
and manage experiment metadata for hyperparameter tuning and model iteration.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manages experiment directories and metadata for training iterations.
    """
    
    def __init__(self, base_output_dir: str):
        """
        Initialize experiment manager.
        
        Args:
            base_output_dir: Base directory for all experiments
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create current experiment directory
        self.experiment_id = self._generate_experiment_id()
        self.experiment_dir = self.base_output_dir / self.experiment_id
        
        # Create experiment directory structure
        self._create_experiment_structure()
        
        # Track experiment metadata
        self.start_time = time.time()
        self.metadata = {
            'experiment_id': self.experiment_id,
            'start_time': datetime.now().isoformat(),
            'base_output_dir': str(self.base_output_dir),
            'experiment_dir': str(self.experiment_dir)
        }
        
        logger.info(f"Created experiment directory: {self.experiment_dir}")
    
    def _generate_experiment_id(self) -> str:
        """
        Generate unique experiment ID based on timestamp.
        
        Returns:
            Experiment ID string (YYYYMMDD_HHMMSS)
        """
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")
    
    def _create_experiment_structure(self):
        """Create the complete experiment directory structure."""
        # Define directory structure
        structure = {
            'model_artifacts': {},
            'evaluation_plots': {},
            'results': {},
            'tensorboard': {},
            'logs': {}
        }
        
        # Create all directories
        for dir_name in structure:
            (self.experiment_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created experiment structure in {self.experiment_dir}")
    
    def get_experiment_dir(self) -> Path:
        """Get the current experiment directory."""
        return self.experiment_dir
    
    def get_model_artifacts_dir(self) -> Path:
        """Get the model artifacts directory."""
        return self.experiment_dir / 'model_artifacts'
    
    def get_evaluation_plots_dir(self) -> Path:
        """Get the evaluation plots directory."""
        return self.experiment_dir / 'evaluation_plots'
    
    def get_results_dir(self) -> Path:
        """Get the results directory."""
        return self.experiment_dir / 'results'
    
    def get_tensorboard_dir(self) -> Path:
        """Get the tensorboard directory."""
        return self.experiment_dir / 'tensorboard'
    
    def get_logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.experiment_dir / 'logs'
    
    def update_metadata(self, **kwargs):
        """
        Update experiment metadata.
        
        Args:
            **kwargs: Metadata fields to update
        """
        self.metadata.update(kwargs)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for the experiment.
        
        Returns:
            Dictionary containing system information
        """
        system_info = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'device_count': 0,
            'gpu_info': []
        }
        
        if torch.cuda.is_available():
            system_info['device_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i)
                }
                system_info['gpu_info'].append(gpu_info)
        
        return system_info
    
    def get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': size_all_mb,
            'model_size_gb': size_all_mb / 1024,
            'architecture': str(type(model).__name__),
            'backbone': getattr(model, 'backbone', None),
            'has_attention': hasattr(model, 'attention') and model.attention is not None,
            'has_metadata_fusion': hasattr(model, 'metadata_fusion') and model.metadata_fusion is not None
        }
        
        return model_info
    
    def create_model_summary(
        self, 
        model: torch.nn.Module, 
        config: Any,
        training_time: Optional[float] = None
    ) -> str:
        """
        Create comprehensive model summary.
        
        Args:
            model: Trained model
            config: Configuration object
            training_time: Training time in seconds
            
        Returns:
            Model summary string
        """
        # Get system and model information
        system_info = self.get_system_info()
        model_info = self.get_model_info(model)
        
        # Calculate training time
        if training_time is None:
            training_time = time.time() - self.start_time
        
        # Format training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        training_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Create summary
        summary = f"""
KOOS-PS Prediction Model - Training Summary
==========================================

Experiment Information:
-----------------------
Experiment ID: {self.metadata['experiment_id']}
Start Time: {self.metadata['start_time']}
Training Duration: {training_time_str} ({training_time:.2f} seconds)
Experiment Directory: {self.experiment_dir}

System Information:
------------------
Python Version: {system_info['python_version']}
PyTorch Version: {system_info['pytorch_version']}
CUDA Available: {system_info['cuda_available']}
MPS Available: {system_info['mps_available']}
Device Count: {system_info['device_count']}

GPU Information:
"""
        
        if system_info['gpu_info']:
            for gpu in system_info['gpu_info']:
                summary += f"""
GPU {gpu['device_id']}: {gpu['name']}
  Total Memory: {gpu['memory_total'] / 1024**3:.2f} GB
  Allocated Memory: {gpu['memory_allocated'] / 1024**3:.2f} GB
  Reserved Memory: {gpu['memory_reserved'] / 1024**3:.2f} GB
"""
        else:
            summary += "No GPU devices available\n"
        
        summary += f"""
Model Architecture:
------------------
Architecture: {model_info['architecture']}
Backbone: {model_info['backbone']}
Has Attention: {model_info['has_attention']}
Has Metadata Fusion: {model_info['has_metadata_fusion']}

Model Parameters:
----------------
Total Parameters: {model_info['total_parameters']:,}
Trainable Parameters: {model_info['trainable_parameters']:,}
Non-trainable Parameters: {model_info['non_trainable_parameters']:,}
Model Size: {model_info['model_size_mb']:.2f} MB ({model_info['model_size_gb']:.4f} GB)

Configuration:
--------------
Image Size: {config.data.image_size}
Batch Size: {config.data.batch_size}
Learning Rate: {config.training.learning_rate}
Optimizer: {config.training.optimizer}
Scheduler: {config.training.scheduler}
Epochs: {config.training.num_epochs}
Mixed Precision: {config.system.mixed_precision}

Data Configuration:
------------------
Train Ratio: {config.data.train_ratio}
Validation Ratio: {config.data.val_ratio}
Test Ratio: {config.data.test_ratio}
Number of Workers: {config.data.num_workers}
Pin Memory: {config.data.pin_memory}

Augmentation Configuration:
--------------------------
Horizontal Flip: {config.augmentation.horizontal_flip_prob}
Vertical Flip: {config.augmentation.vertical_flip_prob}
Rotation Degrees: {config.augmentation.rotation_degrees}
Elastic Transform: {config.augmentation.use_elastic_transform}
Color Jitter: {config.augmentation.use_color_jitter}
Gaussian Noise: {config.augmentation.use_gaussian_noise}
Histogram Equalization: {config.augmentation.use_histogram_equalization}
CLAHE: {config.augmentation.use_clahe}

Model Configuration:
-------------------
Backbone: {config.model.backbone}
Pretrained: {config.model.pretrained}
Freeze Backbone: {config.model.freeze_backbone}
Hidden Dimension: {config.model.hidden_dim}
Dropout Rate: {config.model.dropout_rate}
Use Attention: {config.model.use_attention}
Use Metadata Fusion: {config.model.use_metadata_fusion}
Use Skip Connections: {config.model.use_skip_connections}

Training Configuration:
----------------------
Number of Epochs: {config.training.num_epochs}
Learning Rate: {config.training.learning_rate}
Weight Decay: {config.training.weight_decay}
Optimizer: {config.training.optimizer}
Scheduler: {config.training.scheduler}
Early Stopping Patience: {config.training.early_stopping_patience}
Max Grad Norm: {config.training.max_grad_norm}
Loss Function: {config.training.loss_function}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return summary
    
    def save_experiment_metadata(self, config: Any, model: torch.nn.Module, training_time: float):
        """
        Save experiment metadata to JSON file.
        
        Args:
            config: Configuration object
            model: Trained model
            training_time: Training time in seconds
        """
        # Update metadata with final information
        self.update_metadata(
            end_time=datetime.now().isoformat(),
            training_time_seconds=training_time,
            training_time_formatted=f"{int(training_time//3600):02d}:{int((training_time%3600)//60):02d}:{int(training_time%60):02d}",
            config=config.to_dict(),
            system_info=self.get_system_info(),
            model_info=self.get_model_info(model)
        )
        
        # Save metadata
        metadata_file = self.experiment_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, '__dict__'):
                    # Handle objects with __dict__ (like Sequential, etc.)
                    return str(obj)
                return obj
            
            metadata_serializable = convert_numpy_types(self.metadata)
            json.dump(metadata_serializable, f, indent=2)
        
        logger.info(f"Experiment metadata saved to {metadata_file}")
    
    def list_experiments(self) -> list:
        """
        List all experiments in the base output directory.
        
        Returns:
            List of experiment directories
        """
        experiments = []
        for item in self.base_output_dir.iterdir():
            if item.is_dir() and len(item.name) == 15 and item.name.count('_') == 1:
                # Check if it's a timestamp directory (YYYYMMDD_HHMMSS)
                try:
                    datetime.strptime(item.name, '%Y%m%d_%H%M%S')
                    experiments.append(item)
                except ValueError:
                    continue
        
        return sorted(experiments, key=lambda x: x.name, reverse=True)
    
    def get_latest_experiment(self) -> Optional[Path]:
        """
        Get the latest experiment directory.
        
        Returns:
            Path to latest experiment or None
        """
        experiments = self.list_experiments()
        return experiments[0] if experiments else None
