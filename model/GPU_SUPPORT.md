# GPU Support and Device Optimization

## Overview

The KOOS-PS prediction model is designed to automatically detect and optimize for different compute devices, including NVIDIA CUDA GPUs and Apple Silicon M-series chips. The system automatically configures data loading, mixed precision training, and other optimizations based on the detected hardware.

## Supported Devices

### 1. **NVIDIA CUDA GPUs**
- **Requirements**: CUDA-compatible GPU with compute capability 3.5+
- **PyTorch Support**: Full CUDA support with cuDNN
- **Mixed Precision**: Supported with automatic gradient scaling
- **Memory Management**: Automatic GPU memory optimization

### 2. **Apple Silicon M-series (M1, M2, M3, etc.)**
- **Requirements**: Apple Silicon Mac with macOS 12.3+
- **PyTorch Support**: MPS (Metal Performance Shaders) backend
- **Mixed Precision**: Supported without gradient scaling
- **Memory Management**: Automatic memory management by MPS

### 3. **CPU Fallback**
- **Requirements**: Any CPU with sufficient RAM
- **PyTorch Support**: Standard CPU operations
- **Mixed Precision**: Not supported (disabled automatically)
- **Memory Management**: Standard CPU memory management

## Automatic Device Detection

The system automatically detects the best available device and applies appropriate optimizations:

```python
# Device detection happens automatically in train.py
device = get_device()  # Automatically selects best device
device_info = get_device_info()  # Get detailed device information
optimizations = optimize_for_device(device)  # Get device-specific settings
```

### Detection Priority
1. **CUDA GPU** (if available and compatible)
2. **MPS (Apple Silicon)** (if available)
3. **CPU** (fallback)

## Device-Specific Optimizations

### NVIDIA CUDA GPUs

**Data Loading Optimizations:**
- `pin_memory: True` - Faster CPU to GPU transfer
- `num_workers: min(8, gpu_count * 2)` - Multiple data loading workers
- `persistent_workers: True` - Keep workers alive between epochs
- `prefetch_factor: 2` - Prefetch batches for faster loading

**Training Optimizations:**
- Mixed precision training with `torch.cuda.amp.autocast()`
- Automatic gradient scaling with `GradScaler`
- cuDNN benchmark mode for optimal performance
- GPU memory cache clearing

**Memory Management:**
- Automatic GPU memory monitoring
- Memory allocation tracking
- Cache clearing between operations

### Apple Silicon M-series

**Data Loading Optimizations:**
- `pin_memory: False` - Not beneficial on MPS
- `num_workers: 0` - Single-threaded data loading (MPS preference)
- `persistent_workers: False` - Not needed for single-threaded loading

**Training Optimizations:**
- Mixed precision training with `torch.autocast(device_type='mps')`
- No gradient scaling needed (MPS handles it automatically)
- MPS-specific optimizations enabled

**Memory Management:**
- Automatic memory management by MPS
- No manual memory clearing needed
- Unified memory architecture utilization

### CPU Fallback

**Data Loading Optimizations:**
- `pin_memory: False` - Not beneficial on CPU
- `num_workers: min(4, cpu_count)` - Multiple workers for data loading
- `persistent_workers: False` - Not needed for CPU

**Training Optimizations:**
- Standard precision training (no mixed precision)
- No special optimizations needed

## Mixed Precision Training

### CUDA Mixed Precision
```python
# Automatic mixed precision with gradient scaling
with torch.cuda.amp.autocast():
    outputs = model(images, metadata)
    loss = criterion(outputs, targets)

# Gradient scaling for stability
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### MPS Mixed Precision
```python
# MPS mixed precision (no gradient scaling needed)
with torch.autocast(device_type='mps', dtype=torch.float16):
    outputs = model(images, metadata)
    loss = criterion(outputs, targets)

# Standard backward pass
loss.backward()
optimizer.step()
```

## Device Information and Monitoring

The system provides comprehensive device information:

```python
device_info = get_device_info()
# Returns:
{
    'device_type': 'cuda',  # or 'mps' or 'cpu'
    'device_name': 'NVIDIA GeForce RTX 4090',
    'memory_total': 24576000000,  # bytes
    'memory_allocated': 1234567890,  # bytes
    'memory_reserved': 2345678901,  # bytes
    'compute_capability': (8, 9),  # for CUDA
    'cuda_available': True,
    'mps_available': False
}
```

## Performance Considerations

### NVIDIA CUDA GPUs
- **Best Performance**: RTX 3080, 4080, 4090, A100, H100
- **Memory Requirements**: 8GB+ recommended for full model
- **Batch Size**: Can use larger batch sizes (32-64+)
- **Data Loading**: Multiple workers provide significant speedup

### Apple Silicon M-series
- **Best Performance**: M2 Pro/Max, M3 Pro/Max, M4 Pro/Max
- **Memory Requirements**: 16GB+ unified memory recommended
- **Batch Size**: Moderate batch sizes (16-32) work well
- **Data Loading**: Single-threaded loading is optimal

### CPU Fallback
- **Performance**: Significantly slower than GPU
- **Memory Requirements**: 32GB+ RAM recommended
- **Batch Size**: Smaller batch sizes (8-16) recommended
- **Data Loading**: Multiple workers help with I/O

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python model/train.py --batch_size 16

# Or reduce model complexity in config
```

#### MPS Issues
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# If MPS is not available, check macOS version (12.3+ required)
```

#### Mixed Precision Errors
- **CUDA**: Ensure CUDA 11.0+ and compatible GPU
- **MPS**: Ensure macOS 12.3+ and Apple Silicon
- **CPU**: Mixed precision is automatically disabled

### Device Check Script

Use the provided device check script to diagnose issues:

```bash
python model/check_device.py
```

This script will:
- Check system information
- Test CUDA capabilities
- Test MPS capabilities
- Provide optimization recommendations
- Run training simulation tests

## Configuration Examples

### CUDA Configuration
```python
# Optimal for NVIDIA GPU
config.system.device = "cuda"
config.system.mixed_precision = True
config.data.batch_size = 32
config.data.num_workers = 8
config.data.pin_memory = True
```

### MPS Configuration
```python
# Optimal for Apple Silicon
config.system.device = "mps"
config.system.mixed_precision = True
config.data.batch_size = 16
config.data.num_workers = 0
config.data.pin_memory = False
```

### CPU Configuration
```python
# Fallback for CPU
config.system.device = "cpu"
config.system.mixed_precision = False
config.data.batch_size = 8
config.data.num_workers = 4
config.data.pin_memory = False
```

## Best Practices

### 1. **Device Selection**
- Let the system auto-detect the best device
- Use `python model/check_device.py` to verify capabilities
- Monitor GPU memory usage during training

### 2. **Batch Size Tuning**
- Start with default batch size
- Increase if you have more GPU memory
- Decrease if you get out-of-memory errors

### 3. **Mixed Precision**
- Always enable for CUDA and MPS
- Monitor for numerical instability
- Disable if you encounter issues

### 4. **Data Loading**
- Use recommended worker counts for your device
- Monitor data loading bottlenecks
- Adjust prefetch factor if needed

### 5. **Memory Management**
- Monitor memory usage during training
- Use gradient accumulation for large models
- Clear cache between experiments if needed

## Performance Benchmarks

### Expected Training Times (100 epochs, 1000 samples)

| Device | Batch Size | Time per Epoch | Total Time |
|--------|------------|----------------|------------|
| RTX 4090 | 32 | ~2 min | ~3.3 hours |
| RTX 3080 | 32 | ~3 min | ~5 hours |
| M2 Max | 16 | ~4 min | ~6.7 hours |
| M1 Pro | 16 | ~6 min | ~10 hours |
| CPU (i7) | 8 | ~15 min | ~25 hours |

*Times are approximate and depend on model complexity and data size.*

## Conclusion

The KOOS-PS prediction model is fully optimized for both NVIDIA CUDA GPUs and Apple Silicon M-series chips. The automatic device detection and optimization system ensures optimal performance regardless of your hardware configuration. Use the provided tools to verify your setup and monitor performance during training.
