#!/usr/bin/env python3
# model/training/trainer.py
"""
Training module for KOOS-PS prediction model.

This module provides comprehensive training functionality including
training loops, validation, checkpointing, and monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from typing import Dict, Any, Optional, Tuple, List
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(
        self, 
        patience: int = 15, 
        min_delta: float = 1e-4,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            return True
        return False

class LearningRateScheduler:
    """
    Learning rate scheduler with multiple strategies.
    """
    
    def __init__(self, optimizer: optim.Optimizer, config: Any):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            config: Configuration object
        """
        self.optimizer = optimizer
        self.scheduler_type = config.training.scheduler
        self.scheduler_params = config.training.lr_scheduler_params
        
        if self.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.scheduler_params
            )
        elif self.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )
        elif self.scheduler_type == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=config.training.learning_rate * 10,
                epochs=config.training.num_epochs,
                steps_per_epoch=1  # Will be updated during training
            )
        else:
            self.scheduler = None
    
    def step(self, val_loss: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler is None:
            return
        
        if self.scheduler_type == "plateau" and val_loss is not None:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

class ModelTrainer:
    """
    Comprehensive model trainer with advanced features.
    """
    
    def __init__(self, config: Any, experiment_manager=None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            experiment_manager: Experiment manager for output directories
        """
        self.config = config
        self.experiment_manager = experiment_manager
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        self.writer = None
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.system.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.system.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        # Use experiment manager directories if available
        if self.experiment_manager:
            tensorboard_dir = self.experiment_manager.get_tensorboard_dir()
            log_file = self.experiment_manager.get_logs_dir() / 'training.log'
        else:
            tensorboard_dir = self.config.logging.tensorboard_dir
            log_file = self.config.logging.log_file
        
        # TensorBoard
        if self.config.logging.use_tensorboard:
            self.writer = SummaryWriter(str(tensorboard_dir))
        
        # Weights & Biases
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                config=self.config.to_dict(),
                name=self.experiment_manager.experiment_id if self.experiment_manager else None
            )
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = self.config.training.optimizer.lower()
        
        if optimizer_type == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps
            )
        elif optimizer_type == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps
            )
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_type = self.config.training.loss_function.lower()
        
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "huber":
            criterion = nn.SmoothL1Loss()
        elif loss_type == "focal":
            # Custom focal loss for regression
            criterion = FocalLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
        
        return criterion
    
    def _create_early_stopping(self) -> EarlyStopping:
        """Create early stopping utility."""
        return EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=self.config.training.min_delta
        )
    
    def setup_training(
        self, 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Setup training components.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            resume_from_checkpoint: Path to checkpoint file to resume from
        """
        self.model = model.to(self.device)
        self.optimizer = self._create_optimizer(self.model)
        self.criterion = self._create_criterion()
        self.early_stopping = self._create_early_stopping()
        
        # Create scheduler
        self.scheduler = LearningRateScheduler(self.optimizer, self.config)
        
        # Mixed precision scaler
        if self.config.system.mixed_precision:
            if self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Mixed precision training enabled (CUDA)")
            elif self.device.type == 'mps':
                # MPS supports mixed precision but doesn't need GradScaler
                self.scaler = None
                logger.info("Mixed precision training enabled (MPS)")
            else:
                self.scaler = None
                logger.info("Mixed precision not supported on CPU")
        else:
            self.scaler = None
        
        # Load checkpoint if provided
        self.start_epoch = 0
        if resume_from_checkpoint:
            self.start_epoch = self.load_checkpoint(resume_from_checkpoint)
            logger.info(f"Resuming training from epoch {self.start_epoch + 1}")
        
        logger.info(f"Training setup complete. Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                # CUDA mixed precision with GradScaler
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, metadata)
                    predictions = outputs['predictions']
                    loss = self.criterion(predictions, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif self.device.type == 'mps' and self.config.system.mixed_precision:
                # MPS mixed precision (no GradScaler needed)
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = self.model(images, metadata)
                    predictions = outputs['predictions']
                    loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
            else:
                # Standard precision training
                outputs = self.model(images, metadata)
                predictions = outputs['predictions']
                loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            mae = torch.mean(torch.abs(predictions - targets)).item()
            total_mae += mae
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    # CUDA mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, metadata)
                        predictions = outputs['predictions']
                        loss = self.criterion(predictions, targets)
                elif self.device.type == 'mps' and self.config.system.mixed_precision:
                    # MPS mixed precision
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        outputs = self.model(images, metadata)
                        predictions = outputs['predictions']
                        loss = self.criterion(predictions, targets)
                else:
                    # Standard precision
                    outputs = self.model(images, metadata)
                    predictions = outputs['predictions']
                    loss = self.criterion(predictions, targets)
                
                # Update metrics
                total_loss += loss.item()
                mae = torch.mean(torch.abs(predictions - targets)).item()
                total_mae += mae
                num_batches += 1
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return {
            'loss': avg_loss,
            'mae': avg_mae
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training results
        """
        logger.info("Starting training...")
        
        start_time = time.time()
        
        # Start from the resume epoch if checkpoint was loaded
        start_epoch = getattr(self, 'start_epoch', 0)
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_mae'].append(train_metrics['mae'])
            self.training_history['val_mae'].append(val_metrics['mae'])
            self.training_history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint (respect save_frequency)
            if self.config.logging.save_checkpoints:
                if (epoch + 1) % self.config.logging.save_frequency == 0 or epoch == 0:
                    self._save_checkpoint(epoch, val_metrics['loss'])
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Update best validation loss
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_best_model()
            
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train MAE: {train_metrics['mae']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final cleanup: remove all intermediate checkpoints, keep only best
        if self.config.logging.save_best_only:
            if self.experiment_manager:
                checkpoint_dir = Path(self.experiment_manager.get_model_artifacts_dir())
            else:
                checkpoint_dir = Path(self.config.logging.checkpoint_dir)
            self._cleanup_old_checkpoints(checkpoint_dir, keep_last_n=0)
            logger.info("Cleaned up intermediate checkpoints, kept only best model and checkpoint")
        
        # Load best model
        self._load_best_model()
        
        # Test evaluation if test loader provided
        test_results = None
        if test_loader is not None:
            test_results = self.validate_epoch(test_loader)
            logger.info(f"Test Loss: {test_results['loss']:.4f}, Test MAE: {test_results['mae']:.4f}")
        
        return {
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'test_results': test_results,
            'total_time': total_time
        }
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to various backends."""
        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('MAE/Train', train_metrics['mae'], epoch)
            self.writer.add_scalar('MAE/Validation', val_metrics['mae'], epoch)
            self.writer.add_scalar('Learning_Rate', train_metrics['learning_rate'], epoch)
        
        # Weights & Biases
        if self.config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'learning_rate': train_metrics['learning_rate']
            })
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint with automatic cleanup of old checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'training_history': self.training_history
        }
        
        # Save scheduler state if available
        if self.scheduler.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.scheduler.state_dict()
        
        # Save scaler state if available
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Use experiment manager directory if available
        if self.experiment_manager:
            checkpoint_dir = Path(self.experiment_manager.get_model_artifacts_dir())
        else:
            checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best so far
        if val_loss < self.best_val_loss:
            best_checkpoint_path = checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_checkpoint_path)
        
        # Cleanup old checkpoints if save_best_only is enabled
        if self.config.logging.save_best_only:
            self._cleanup_old_checkpoints(checkpoint_dir, keep_last_n=1)
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path, keep_last_n: int = 1):
        """
        Remove old checkpoint files, keeping only essential ones.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            keep_last_n: Number of recent checkpoints to keep (in addition to best)
        """
        try:
            # Find all epoch checkpoint files
            checkpoint_files = sorted(
                checkpoint_dir.glob('checkpoint_epoch_*.pth'),
                key=lambda x: int(x.stem.split('_')[-1])
            )
            
            # Keep only the last N checkpoints (excluding best)
            if len(checkpoint_files) > keep_last_n:
                files_to_remove = checkpoint_files[:-keep_last_n]
                for file in files_to_remove:
                    try:
                        file.unlink()
                        logger.debug(f"Removed old checkpoint: {file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {file.name}: {e}")
        except Exception as e:
            logger.warning(f"Error during checkpoint cleanup: {e}")
    
    def _save_best_model(self):
        """Save the best model."""
        # Use experiment manager directory if available
        if self.experiment_manager:
            model_dir = self.experiment_manager.get_model_artifacts_dir()
        else:
            model_dir = self.config.logging.checkpoint_dir
        
        best_model_path = os.path.join(
            model_dir,
            'best_model.pth'
        )
        torch.save(self.model.state_dict(), best_model_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number to resume from
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        # Load best validation loss
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']
        
        # Get epoch number (0-indexed, so we resume from epoch + 1)
        start_epoch = checkpoint.get('epoch', 0)
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and self.scheduler.scheduler is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch + 1}")
        logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")
        
        return start_epoch
    
    def _load_best_model(self):
        """Load the best model."""
        # Use experiment manager directory if available
        if self.experiment_manager:
            model_dir = self.experiment_manager.get_model_artifacts_dir()
        else:
            model_dir = self.config.logging.checkpoint_dir
        
        best_model_path = os.path.join(
            model_dir,
            'best_model.pth'
        )
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            logger.info("Loaded best model")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer is not None:
            self.writer.close()
        
        if self.config.logging.use_wandb:
            wandb.finish()

class FocalLoss(nn.Module):
    """
    Focal Loss for regression tasks.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Focal loss value
        """
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse_loss
        return focal_loss.mean()
