#!/usr/bin/env python3
# model/train.py
"""
Main training script for KOOS-PS prediction model.

This script orchestrates the entire training pipeline including data loading,
model training, evaluation, and saving results.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Optional

# Add model directory to path
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from data.dataset import KOOSDataModule
from data.augmentations import create_augmentation_pipelines
from models.cnn_model import ModelFactory
from training.trainer import ModelTrainer
from metrics.evaluation import ModelEvaluator
from utils.helpers import (
    set_seed, get_device, get_device_info, optimize_for_device, 
    save_model_artifacts, plot_training_history, calculate_model_size, load_config
)
from utils.experiment_manager import ExperimentManager

def setup_logging(config: Config) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path(config.logging.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logging.log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    
    return logger

def load_data(config: Config, device_optimizations: dict = None) -> KOOSDataModule:
    """
    Load and prepare data with device-specific optimizations.
    
    Args:
        config: Configuration object
        device_optimizations: Device-specific optimization settings
        
    Returns:
        Data module
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")
    
    # Check if HDF5 binary format exists, otherwise fall back to individual files
    hdf5_file = None
    csv_file = None
    image_dir = None
    
    if os.path.exists(config.data.hdf5_file):
        hdf5_file = config.data.hdf5_file
        logger.info(f"Using binary PyTorch optimized format: {hdf5_file}")
    else:
        csv_file = config.data.csv_file
        image_dir = config.data.image_dir
        logger.info(f"Using individual files: CSV={csv_file}, ImageDir={image_dir}")
    
    # Create augmentation pipelines
    augmentation_pipelines = create_augmentation_pipelines(config)
    
    # Create data module
    data_module = KOOSDataModule(
        csv_file=csv_file,
        image_dir=image_dir,
        hdf5_file=hdf5_file,
        config=config,
        augmentations=augmentation_pipelines
    )
    
    logger.info("Data loading complete")
    return data_module

def create_model(config: Config) -> torch.nn.Module:
    """
    Create and initialize model.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized model
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating model...")
    
    # Create model
    model = ModelFactory.create_model(config)
    
    # Print model summary
    model_summary = ModelFactory.get_model_summary(model)
    logger.info(f"Model created:\n{model_summary}")
    
    # Calculate model size
    size_info = calculate_model_size(model)
    logger.info(f"Model size: {size_info['total_size_mb']:.2f} MB")
    logger.info(f"Total parameters: {size_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {size_info['trainable_parameters']:,}")
    
    return model

def train_model(
    model: torch.nn.Module,
    data_module: KOOSDataModule,
    config: Config,
    experiment_manager: ExperimentManager,
    device_optimizations: dict = None,
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train the model with device-specific optimizations.
    
    Args:
        model: Model to train
        data_module: Data module
        config: Configuration object
        experiment_manager: Experiment manager for output directories
        device_optimizations: Device-specific optimization settings
        resume_from_checkpoint: Path to checkpoint file to resume from
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Get data loaders with device optimizations
    train_loader, val_loader, test_loader = data_module.get_dataloaders(device_optimizations)
    
    # Create trainer with experiment manager
    trainer = ModelTrainer(config, experiment_manager)
    
    # Setup training (with optional checkpoint resume)
    trainer.setup_training(model, train_loader, val_loader, resume_from_checkpoint)
    
    # Train model
    training_results = trainer.train(train_loader, val_loader, test_loader)
    
    # Cleanup
    trainer.cleanup()
    
    logger.info("Model training complete")
    return training_results

def evaluate_model(
    model: torch.nn.Module,
    data_module: KOOSDataModule,
    config: Config,
    experiment_manager: ExperimentManager,
    device_optimizations: dict = None
) -> Dict[str, Any]:
    """
    Evaluate the trained model with device-specific optimizations.
    
    Args:
        model: Trained model
        data_module: Data module
        config: Configuration object
        experiment_manager: Experiment manager for output directories
        device_optimizations: Device-specific optimization settings
        
    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model...")
    
    # Get data loaders with device optimizations
    train_loader, val_loader, test_loader = data_module.get_dataloaders(device_optimizations)
    
    # Create evaluator with experiment manager
    evaluator = ModelEvaluator(config, experiment_manager)
    
    # Evaluate on test set
    device = get_device()
    test_results = evaluator.evaluate_model(model, test_loader, device)
    
    # Create evaluation plots
    plots_dir = experiment_manager.get_evaluation_plots_dir()
    evaluation_plots = evaluator.create_evaluation_plots(
        test_results, 
        save_path=str(plots_dir)
    )
    
    logger.info("Model evaluation complete")
    return test_results

def save_results(
    model: torch.nn.Module,
    config: Config,
    training_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    experiment_manager: ExperimentManager
):
    """
    Save all training and evaluation results.
    
    Args:
        model: Trained model
        config: Configuration object
        training_results: Training results
        evaluation_results: Evaluation results
        experiment_manager: Experiment manager for output directories
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving results...")
    
    # Prepare metrics for saving
    metrics = evaluation_results['metrics']
    
    # Save model artifacts
    artifacts_dir = experiment_manager.get_model_artifacts_dir()
    save_model_artifacts(
        model=model,
        config=config,
        training_history=training_results['training_history'],
        metrics=metrics,
        save_dir=str(artifacts_dir)
    )
    
    # Create comprehensive model summary
    training_time = training_results.get('total_time', 0)
    model_summary = experiment_manager.create_model_summary(
        model, config, training_time
    )
    
    # Save model summary
    with open(artifacts_dir / 'model_summary.txt', 'w') as f:
        f.write(model_summary)
    
    # Save detailed results
    results_dir = experiment_manager.get_results_dir()
    
    # Save evaluation results
    import json
    with open(results_dir / "evaluation_results.json", 'w') as f:
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
            return obj
        
        eval_results_serializable = convert_numpy_types(evaluation_results)
        json.dump(eval_results_serializable, f, indent=2)
    
    # Save training results
    with open(results_dir / "training_results.json", 'w') as f:
        training_results_serializable = convert_numpy_types(training_results)
        json.dump(training_results_serializable, f, indent=2)
    
    # Save experiment metadata
    experiment_manager.save_experiment_metadata(
        config, model, training_time
    )
    
    # Copy HDF5 file to experiment directory if it exists
    if os.path.exists(config.data.hdf5_file):
        hdf5_dest = results_dir / os.path.basename(config.data.hdf5_file)
        try:
            shutil.copy2(config.data.hdf5_file, hdf5_dest)
            logger.info(f"Copied HDF5 file to experiment directory: {hdf5_dest}")
        except Exception as e:
            logger.warning(f"Failed to copy HDF5 file: {e}")
    
    # Create highly compressed archive of entire experiment directory
    try:
        import zipfile
        experiment_dir = experiment_manager.get_experiment_dir()
        archive_path = experiment_dir.parent / f"{experiment_manager.experiment_id}.zip"
        
        logger.info(f"Creating compressed archive: {archive_path}")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            # Walk through all files in the experiment directory
            for root, dirs, files in os.walk(experiment_dir):
                # Skip .DS_Store files and hidden directories
                files = [f for f in files if not f.endswith('.DS_Store')]
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    file_path = Path(root) / file
                    # Create archive path preserving experiment directory structure
                    arcname = file_path.relative_to(experiment_dir.parent)
                    zipf.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)
        
        archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
        logger.info(f"Created compressed archive: {archive_path} ({archive_size_mb:.2f} MB)")
    except Exception as e:
        logger.warning(f"Failed to create compressed archive: {e}")
    
    logger.info(f"Results saved to {experiment_manager.get_experiment_dir()}")

def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train KOOS-PS prediction model')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='model/outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load from file
        config_dict = load_config(args.config)
        config = Config()
        config.update_from_dict(config_dict)
    else:
        # Use default configuration
        config = Config()
    
    # Override with command line arguments
    if args.data_dir:
        # Use the configured image directory name from config, not hardcoded 'rx'
        image_dir_name = os.path.basename(config.data.image_dir)
        config.data.image_dir = os.path.join(args.data_dir, image_dir_name)
        config.data.csv_file = os.path.join(args.data_dir, 'HALS_Dataset_v1.csv')
    if args.output_dir:
        config.data.output_dir = args.output_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.system.device = args.device
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(config)
    
    try:
        # Detect device and get optimizations
        device = get_device()
        device_info = get_device_info()
        device_optimizations = optimize_for_device(device)
        
        logger.info("="*60)
        logger.info("DEVICE DETECTION AND OPTIMIZATION")
        logger.info("="*60)
        logger.info(f"Device Type: {device_info['device_type']}")
        logger.info(f"Device Name: {device_info['device_name']}")
        logger.info(f"CUDA Available: {device_info['cuda_available']}")
        logger.info(f"MPS Available: {device_info['mps_available']}")
        
        if device_info['device_type'] == 'cuda':
            logger.info(f"GPU Memory: {device_info['memory_total'] / 1024**3:.2f} GB")
            logger.info(f"Compute Capability: {device_info['compute_capability']}")
        elif device_info['device_type'] == 'mps':
            logger.info("Apple Silicon M-series GPU detected")
        
        logger.info("Device Optimizations Applied:")
        for key, value in device_optimizations.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*60)
        
        # Handle resume from checkpoint - determine experiment directory
        resume_checkpoint = None
        experiment_dir_from_checkpoint = None
        
        if args.resume:
            resume_checkpoint = args.resume
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            # Extract experiment directory from checkpoint path
            checkpoint_path = Path(resume_checkpoint)
            # Checkpoint is typically in: outputs/EXPERIMENT_ID/model_artifacts/checkpoint.pth
            if 'model_artifacts' in checkpoint_path.parts:
                # Find the experiment directory (parent of model_artifacts)
                parts = checkpoint_path.parts
                artifacts_idx = parts.index('model_artifacts')
                if artifacts_idx > 0:
                    # Reconstruct path up to experiment directory
                    base_dir = Path(*parts[:parts.index('outputs') + 1])
                    exp_id = parts[parts.index('outputs') + 1]
                    experiment_dir_from_checkpoint = base_dir / exp_id
                    logger.info(f"Found experiment directory from checkpoint: {experiment_dir_from_checkpoint}")
        else:
            # Try to find the latest checkpoint in any existing experiment
            output_dir = Path(config.data.output_dir)
            if output_dir.exists():
                # Find all experiment directories
                experiment_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
                for exp_dir in experiment_dirs:
                    artifacts_dir = exp_dir / 'model_artifacts'
                    if artifacts_dir.exists():
                        # Check for best_checkpoint first, then latest epoch checkpoint
                        best_checkpoint = artifacts_dir / 'best_checkpoint.pth'
                        if best_checkpoint.exists():
                            resume_checkpoint = str(best_checkpoint)
                            experiment_dir_from_checkpoint = exp_dir
                            logger.info(f"Found best checkpoint, resuming from: {resume_checkpoint}")
                            break
                        else:
                            # Find latest epoch checkpoint
                            epoch_checkpoints = sorted(artifacts_dir.glob('checkpoint_epoch_*.pth'))
                            if epoch_checkpoints:
                                resume_checkpoint = str(epoch_checkpoints[-1])
                                experiment_dir_from_checkpoint = exp_dir
                                logger.info(f"Found latest checkpoint, resuming from: {resume_checkpoint}")
                                break
        
        # Create experiment manager
        # If resuming, try to use the same experiment directory
        if experiment_dir_from_checkpoint and experiment_dir_from_checkpoint.exists():
            # Use existing experiment directory
            experiment_manager = ExperimentManager(config.data.output_dir)
            # Override the experiment directory to use the existing one
            experiment_manager.experiment_dir = experiment_dir_from_checkpoint
            experiment_manager.experiment_id = experiment_dir_from_checkpoint.name
            logger.info(f"Resuming experiment: {experiment_manager.experiment_id}")
        else:
            # Create new experiment
            experiment_manager = ExperimentManager(config.data.output_dir)
            logger.info(f"Created new experiment: {experiment_manager.experiment_id}")
        
        # Load data with device optimizations
        data_module = load_data(config, device_optimizations)
        
        # Create model
        model = create_model(config)
        
        # Train model with device optimizations
        training_results = train_model(model, data_module, config, experiment_manager, device_optimizations, resume_checkpoint)
        
        # Evaluate model with device optimizations
        evaluation_results = evaluate_model(model, data_module, config, experiment_manager, device_optimizations)
        
        # Save results
        save_results(model, config, training_results, evaluation_results, experiment_manager)
        
        # Print final metrics
        logger.info("Training completed successfully!")
        logger.info(f"Experiment ID: {experiment_manager.experiment_id}")
        logger.info(f"Experiment directory: {experiment_manager.get_experiment_dir()}")
        logger.info(f"Device used: {device_info['device_name']}")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"Test MAE: {evaluation_results['metrics']['mae']:.4f}")
        logger.info(f"Test R²: {evaluation_results['metrics']['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
