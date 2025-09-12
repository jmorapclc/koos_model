#!/usr/bin/env python3
"""
Main training script for KOOS-PS prediction model.

This script orchestrates the entire training pipeline including data loading,
model training, evaluation, and saving results.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any

# Add model directory to path
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from data.dataset import KOOSDataModule
from data.augmentations import create_augmentation_pipelines
from models.cnn_model import ModelFactory
from training.trainer import ModelTrainer
from metrics.evaluation import ModelEvaluator
from utils.helpers import (
    set_seed, get_device, save_model_artifacts, 
    plot_training_history, calculate_model_size, load_config
)

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

def load_data(config: Config) -> KOOSDataModule:
    """
    Load and prepare data.
    
    Args:
        config: Configuration object
        
    Returns:
        Data module
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")
    
    # Create augmentation pipelines
    augmentation_pipelines = create_augmentation_pipelines(config)
    
    # Create data module
    data_module = KOOSDataModule(
        csv_file=config.data.csv_file,
        image_dir=config.data.image_dir,
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
    config: Config
) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        model: Model to train
        data_module: Data module
        config: Configuration object
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Setup training
    trainer.setup_training(model, train_loader, val_loader)
    
    # Train model
    training_results = trainer.train(train_loader, val_loader, test_loader)
    
    # Cleanup
    trainer.cleanup()
    
    logger.info("Model training complete")
    return training_results

def evaluate_model(
    model: torch.nn.Module,
    data_module: KOOSDataModule,
    config: Config
) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        data_module: Data module
        config: Configuration object
        
    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model...")
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate on test set
    device = get_device()
    test_results = evaluator.evaluate_model(model, test_loader, device)
    
    # Create evaluation plots
    plots_dir = Path(config.data.output_dir) / "evaluation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
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
    evaluation_results: Dict[str, Any]
):
    """
    Save all training and evaluation results.
    
    Args:
        model: Trained model
        config: Configuration object
        training_results: Training results
        evaluation_results: Evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving results...")
    
    # Prepare metrics for saving
    metrics = evaluation_results['metrics']
    
    # Save model artifacts
    artifacts_dir = Path(config.data.output_dir) / "model_artifacts"
    save_model_artifacts(
        model=model,
        config=config,
        training_history=training_results['training_history'],
        metrics=metrics,
        save_dir=str(artifacts_dir)
    )
    
    # Save detailed results
    results_dir = Path(config.data.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation results
    import json
    with open(results_dir / "evaluation_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        eval_results_serializable = {}
        for key, value in evaluation_results.items():
            if isinstance(value, np.ndarray):
                eval_results_serializable[key] = value.tolist()
            else:
                eval_results_serializable[key] = value
        json.dump(eval_results_serializable, f, indent=2)
    
    # Save training results
    with open(results_dir / "training_results.json", 'w') as f:
        training_results_serializable = {}
        for key, value in training_results.items():
            if isinstance(value, np.ndarray):
                training_results_serializable[key] = value.tolist()
            else:
                training_results_serializable[key] = value
        json.dump(training_results_serializable, f, indent=2)
    
    logger.info(f"Results saved to {config.data.output_dir}")

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
        config.data.image_dir = os.path.join(args.data_dir, 'rx')
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
        # Load data
        data_module = load_data(config)
        
        # Create model
        model = create_model(config)
        
        # Train model
        training_results = train_model(model, data_module, config)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, data_module, config)
        
        # Save results
        save_results(model, config, training_results, evaluation_results)
        
        # Print final metrics
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"Test MAE: {evaluation_results['metrics']['mae']:.4f}")
        logger.info(f"Test R²: {evaluation_results['metrics']['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
