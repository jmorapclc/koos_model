#!/usr/bin/env python3
"""
Test script to validate model setup and configuration.

This script tests the model components without training to ensure
everything is properly configured and working.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add model directory to path
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from data.dataset import KOOSDataset
from data.augmentations import create_augmentation_pipelines
from models.cnn_model import ModelFactory
from utils.helpers import set_seed, get_device, calculate_model_size
from utils.experiment_manager import ExperimentManager

def test_configuration():
    """Test configuration loading and validation."""
    print("Testing configuration...")
    
    try:
        config = Config()
        print(f"✓ Configuration loaded successfully")
        print(f"  - Image size: {config.data.image_size}")
        print(f"  - Batch size: {config.data.batch_size}")
        print(f"  - Model backbone: {config.model.backbone}")
        print(f"  - Learning rate: {config.training.learning_rate}")
        return config
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return None

def test_data_loading(config):
    """Test data loading and preprocessing."""
    print("\nTesting data loading...")
    
    try:
        # Test dataset creation
        dataset = KOOSDataset(
            csv_file=config.data.csv_file,
            image_dir=config.data.image_dir,
            image_size=config.data.image_size,
            augmentations=None,
            is_training=True
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Number of samples: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test data loading
            sample = dataset[0]
            print(f"  - Image shape: {sample['image'].shape}")
            print(f"  - Metadata shape: {sample['metadata'].shape}")
            print(f"  - Target: {sample['target']}")
            print(f"  - HALS MRN: {sample['hals_mrn']}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_augmentations(config):
    """Test augmentation pipelines."""
    print("\nTesting augmentations...")
    
    try:
        augmentation_pipelines = create_augmentation_pipelines(config)
        
        print(f"✓ Augmentation pipelines created successfully")
        print(f"  - Train augmentations: {len(augmentation_pipelines['train'].transforms)} transforms")
        print(f"  - Val augmentations: {len(augmentation_pipelines['val'].transforms)} transforms")
        print(f"  - Test augmentations: {len(augmentation_pipelines['test'].transforms)} transforms")
        
        return True
    except Exception as e:
        print(f"✗ Augmentations failed: {e}")
        return False

def test_model_creation(config):
    """Test model creation and architecture."""
    print("\nTesting model creation...")
    
    try:
        # Create model
        model = ModelFactory.create_model(config)
        
        print(f"✓ Model created successfully")
        
        # Calculate model size
        size_info = calculate_model_size(model)
        print(f"  - Total parameters: {size_info['total_parameters']:,}")
        print(f"  - Trainable parameters: {size_info['trainable_parameters']:,}")
        print(f"  - Model size: {size_info['total_size_mb']:.2f} MB")
        
        # Test forward pass
        device = get_device()
        model = model.to(device)
        
        # Create dummy inputs
        batch_size = 2
        images = torch.randn(batch_size, 3, config.data.image_size, config.data.image_size).to(device)
        metadata = torch.randn(batch_size, config.model.num_metadata_features).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(images, metadata)
        
        print(f"  - Output shape: {outputs['predictions'].shape}")
        print(f"  - Features shape: {outputs['features'].shape}")
        print(f"  - Fused features shape: {outputs['fused_features'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_training_components(config):
    """Test training components."""
    print("\nTesting training components...")
    
    try:
        from training.trainer import ModelTrainer, EarlyStopping, LearningRateScheduler
        from torch import optim
        
        # Test early stopping
        early_stopping = EarlyStopping(patience=5)
        print(f"✓ Early stopping created")
        
        # Test learning rate scheduler
        dummy_optimizer = optim.Adam([torch.tensor(1.0, requires_grad=True)])
        scheduler = LearningRateScheduler(dummy_optimizer, config)
        print(f"✓ Learning rate scheduler created")
        
        # Test trainer
        trainer = ModelTrainer(config)
        print(f"✓ Trainer created")
        
        return True
    except Exception as e:
        print(f"✗ Training components failed: {e}")
        return False

def test_evaluation_components(config):
    """Test evaluation components."""
    print("\nTesting evaluation components...")
    
    try:
        from metrics.evaluation import RegressionMetrics, ModelEvaluator
        
        # Test metrics calculator
        metrics_calc = RegressionMetrics(config)
        print(f"✓ Metrics calculator created")
        
        # Test evaluator
        evaluator = ModelEvaluator(config)
        print(f"✓ Model evaluator created")
        
        # Test metrics calculation
        predictions = np.random.randn(100) * 10 + 50
        targets = np.random.randn(100) * 10 + 50
        metadata = np.random.randn(100, 7)
        
        metrics = metrics_calc.calculate_metrics(predictions, targets, metadata)
        print(f"  - Calculated {len(metrics)} metrics")
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - R²: {metrics['r2']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation components failed: {e}")
        return False

def test_experiment_manager(config):
    """Test experiment manager."""
    print("\nTesting experiment manager...")
    
    try:
        # Create experiment manager
        experiment_manager = ExperimentManager("test_outputs")
        print(f"✓ Experiment manager created")
        print(f"  - Experiment ID: {experiment_manager.experiment_id}")
        print(f"  - Experiment directory: {experiment_manager.get_experiment_dir()}")
        
        # Test directory structure
        assert experiment_manager.get_model_artifacts_dir().exists()
        assert experiment_manager.get_evaluation_plots_dir().exists()
        assert experiment_manager.get_results_dir().exists()
        assert experiment_manager.get_tensorboard_dir().exists()
        assert experiment_manager.get_logs_dir().exists()
        print(f"✓ Directory structure created")
        
        # Test system info
        system_info = experiment_manager.get_system_info()
        print(f"  - System info collected: {len(system_info)} fields")
        
        # Cleanup test directory
        import shutil
        shutil.rmtree("test_outputs", ignore_errors=True)
        print(f"✓ Test cleanup completed")
        
        return True
    except Exception as e:
        print(f"✗ Experiment manager failed: {e}")
        return False

def test_device_detection():
    """Test device detection and optimization."""
    print("\nTesting device detection...")
    
    try:
        from utils.helpers import get_device, get_device_info, optimize_for_device
        
        # Test device detection
        device = get_device()
        print(f"✓ Device detected: {device}")
        
        # Test device info
        device_info = get_device_info()
        print(f"  - Device type: {device_info['device_type']}")
        print(f"  - Device name: {device_info['device_name']}")
        print(f"  - CUDA available: {device_info['cuda_available']}")
        print(f"  - MPS available: {device_info['mps_available']}")
        
        # Test optimizations
        optimizations = optimize_for_device(device)
        print(f"  - Optimizations: {len(optimizations)} settings")
        
        # Test basic tensor operations
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✓ Tensor operations successful on {device}")
        
        # Test mixed precision if available
        if device_info['device_type'] == 'cuda' and torch.cuda.is_available():
            try:
                with torch.cuda.amp.autocast():
                    test_tensor = torch.randn(10, 10).to(device)
                    result = torch.matmul(test_tensor, test_tensor.T)
                print(f"✓ CUDA mixed precision test successful")
            except Exception as e:
                print(f"✗ CUDA mixed precision test failed: {e}")
                
        elif device_info['device_type'] == 'mps' and torch.backends.mps.is_available():
            try:
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    test_tensor = torch.randn(10, 10).to(device)
                    result = torch.matmul(test_tensor, test_tensor.T)
                print(f"✓ MPS mixed precision test successful")
            except Exception as e:
                print(f"✗ MPS mixed precision test failed: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("KOOS-PS Prediction Model Setup Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Test configuration
    config = test_configuration()
    if config is None:
        print("\n❌ Configuration test failed. Exiting.")
        return
    
    # Test data loading
    data_ok = test_data_loading(config)
    
    # Test augmentations
    aug_ok = test_augmentations(config)
    
    # Test model creation
    model_ok = test_model_creation(config)
    
    # Test training components
    training_ok = test_training_components(config)
    
    # Test evaluation components
    eval_ok = test_evaluation_components(config)
    
    # Test experiment manager
    exp_ok = test_experiment_manager(config)
    
    # Test device detection
    device_ok = test_device_detection()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    tests = [
        ("Configuration", config is not None),
        ("Data Loading", data_ok),
        ("Augmentations", aug_ok),
        ("Model Creation", model_ok),
        ("Training Components", training_ok),
        ("Evaluation Components", eval_ok),
        ("Experiment Manager", exp_ok),
        ("Device Detection", device_ok)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Model setup is ready for training.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
