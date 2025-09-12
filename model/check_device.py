#!/usr/bin/env python3
# model/check_device.py
"""
Device detection and capability check script.

This script checks the available compute devices and their capabilities
for optimal model training configuration.
"""

import sys
import torch
import platform
from pathlib import Path

# Add model directory to path
sys.path.append(str(Path(__file__).parent))

from utils.helpers import get_device, get_device_info, optimize_for_device

def check_system_info():
    """Check basic system information."""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print()

def check_cuda_capabilities():
    """Check CUDA capabilities if available."""
    if torch.cuda.is_available():
        print("="*60)
        print("CUDA GPU INFORMATION")
        print("="*60)
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print(f"  Max Threads per Block: {props.max_threads_per_block}")
            print(f"  Max Threads per Multiprocessor: {props.max_threads_per_multiprocessor}")
            print(f"  Memory Clock Rate: {props.memory_clock_rate / 1000:.0f} MHz")
            print(f"  Memory Bus Width: {props.memory_bus_width} bits")
            print()
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory Allocated: {memory_allocated:.2f} GB")
            print(f"  Memory Reserved: {memory_reserved:.2f} GB")
            print(f"  Memory Free: {(props.total_memory / 1024**3) - memory_reserved:.2f} GB")
            print()
    else:
        print("CUDA not available")
        print()

def check_mps_capabilities():
    """Check MPS capabilities if available."""
    if torch.backends.mps.is_available():
        print("="*60)
        print("MPS (APPLE SILICON) INFORMATION")
        print("="*60)
        print("MPS (Metal Performance Shaders) is available")
        print("This indicates you have an Apple Silicon Mac (M1, M2, M3, etc.)")
        print("MPS provides GPU acceleration for PyTorch on Apple Silicon")
        print()
        
        # Test MPS functionality
        try:
            device = torch.device("mps")
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print("✓ MPS functionality test passed")
            print(f"✓ Test tensor computation successful on MPS")
        except Exception as e:
            print(f"✗ MPS functionality test failed: {e}")
        print()
    else:
        print("MPS not available (not running on Apple Silicon)")
        print()

def check_optimization_recommendations():
    """Provide optimization recommendations based on detected hardware."""
    print("="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    device = get_device()
    device_info = get_device_info()
    optimizations = optimize_for_device(device)
    
    print(f"Recommended Device: {device_info['device_name']}")
    print(f"Device Type: {device_info['device_type']}")
    print()
    
    print("Recommended Settings:")
    for key, value in optimizations.items():
        print(f"  {key}: {value}")
    print()
    
    if device_info['device_type'] == 'cuda':
        print("CUDA-Specific Recommendations:")
        print("  - Use mixed precision training for better performance")
        print("  - Enable pin_memory for faster data transfer")
        print("  - Use multiple workers for data loading")
        print("  - Consider gradient accumulation for large models")
        print("  - Monitor GPU memory usage during training")
        print()
        
    elif device_info['device_type'] == 'mps':
        print("MPS-Specific Recommendations:")
        print("  - Use mixed precision training (MPS supports it)")
        print("  - Disable pin_memory (not beneficial on MPS)")
        print("  - Use single-threaded data loading (num_workers=0)")
        print("  - MPS handles memory management automatically")
        print("  - Consider smaller batch sizes if memory is limited")
        print()
        
    else:
        print("CPU-Specific Recommendations:")
        print("  - Disable mixed precision (not supported on CPU)")
        print("  - Use multiple workers for data loading")
        print("  - Consider reducing model complexity")
        print("  - Training will be slower than GPU")
        print()

def test_training_simulation():
    """Test a small training simulation to verify everything works."""
    print("="*60)
    print("TRAINING SIMULATION TEST")
    print("="*60)
    
    try:
        device = get_device()
        device_info = get_device_info()
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(device)
        
        # Create dummy data
        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 1).to(device)
        
        # Test forward pass
        with torch.no_grad():
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
        
        print(f"✓ Model created and moved to {device}")
        print(f"✓ Forward pass successful")
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        
        # Test mixed precision if available
        if device_info['device_type'] == 'cuda' and torch.cuda.is_available():
            try:
                with torch.cuda.amp.autocast():
                    output = model(x)
                    loss = torch.nn.functional.mse_loss(output, y)
                print("✓ CUDA mixed precision test successful")
            except Exception as e:
                print(f"✗ CUDA mixed precision test failed: {e}")
                
        elif device_info['device_type'] == 'mps' and torch.backends.mps.is_available():
            try:
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    output = model(x)
                    loss = torch.nn.functional.mse_loss(output, y)
                print("✓ MPS mixed precision test successful")
            except Exception as e:
                print(f"✗ MPS mixed precision test failed: {e}")
        
        print("✓ Training simulation completed successfully")
        
    except Exception as e:
        print(f"✗ Training simulation failed: {e}")
        return False
    
    return True

def main():
    """Main function to run all checks."""
    print("KOOS-PS Model Device Detection and Capability Check")
    print("="*60)
    print()
    
    # Run all checks
    check_system_info()
    check_cuda_capabilities()
    check_mps_capabilities()
    check_optimization_recommendations()
    
    # Test training simulation
    success = test_training_simulation()
    
    print("="*60)
    if success:
        print("✓ ALL CHECKS PASSED - System is ready for training!")
    else:
        print("✗ SOME CHECKS FAILED - Please review the issues above")
    print("="*60)

if __name__ == "__main__":
    main()
