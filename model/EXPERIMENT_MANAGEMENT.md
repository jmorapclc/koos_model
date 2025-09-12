# Experiment Management System

## Overview

The KOOS-PS prediction model now includes a comprehensive experiment management system that automatically creates timestamped directories for each training iteration. This allows you to track, compare, and manage multiple experimental runs with different hyperparameters and model architectures.

## Directory Structure

Each training run creates a timestamped directory with the following structure:

```
model/outputs/
20250911_203021/                    # Timestamp: YYYYMMDD_HHMMSS
├── model_artifacts/
│   ├── model.pth                   # Trained model weights
│   ├── best_model.pth              # Best model weights
│   ├── config.json                 # Configuration used
│   ├── training_history.json       # Training metrics
│   ├── metrics.json                # Evaluation metrics
│   └── model_summary.txt           # Detailed model summary
├── evaluation_plots/
│   ├── scatter.png                 # Prediction vs target
│   ├── residuals.png               # Residuals analysis
│   ├── bland_altman.png            # Bland-Altman plot
│   ├── error_distribution.png      # Error distribution
│   ├── subgroup.png                # Subgroup analysis
│   └── metrics_summary.png         # Metrics summary
├── results/
│   ├── evaluation_results.json     # Detailed evaluation results
│   ├── training_results.json       # Training results
│   └── experiment_metadata.json    # Experiment metadata
├── tensorboard/                    # TensorBoard logs
├── logs/
│   └── training.log                # Training logs
└── predictions.csv                 # Detailed predictions
```

## Key Features

### 1. **Automatic Timestamping**
- Each experiment gets a unique timestamp (YYYYMMDD_HHMMSS)
- No manual directory management required
- Chronological ordering of experiments

### 2. **Comprehensive Model Summary**
The `model_summary.txt` file includes:
- **Experiment Information**: ID, start time, training duration
- **System Information**: Python version, PyTorch version, GPU details
- **Model Architecture**: Parameters, size, backbone, features
- **Configuration**: All hyperparameters and settings
- **Training Details**: Epochs, learning rate, optimizer, scheduler

### 3. **Experiment Metadata**
Each experiment saves detailed metadata including:
- System specifications
- Model architecture details
- Training configuration
- Performance metrics
- GPU information and memory usage

### 4. **Easy Comparison**
- List all experiments with key metrics
- Compare experiments by specific metrics
- View detailed information for any experiment

## Usage

### Basic Training
```bash
# Train with default settings
python model/train.py

# Train with custom parameters
python model/train.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

### List Experiments
```bash
# List all experiments
python model/list_experiments.py

# Compare by test MAE
python model/list_experiments.py --compare test_mae

# Show details for specific experiment
python model/list_experiments.py --details 20250911_203021

# Show details for latest experiment
python model/list_experiments.py --latest
```

### Example Output
```
TRAINING EXPERIMENTS SUMMARY
========================================================================================================================
experiment_id    start_time              training_time    best_val_loss    test_mae    test_r2    total_parameters    gpu_used
20250911_203021  2025-09-11T20:30:21    00:15:32         8.2345           7.8923      0.7845     8,234,567           NVIDIA RTX 4090
20250911_184512  2025-09-11T18:45:12    00:12:45         9.1234           8.4567      0.7234     8,234,567           NVIDIA RTX 4090
20250911_162301  2025-09-11T16:23:01    00:18:21         7.8901           7.2345      0.8123     8,234,567           NVIDIA RTX 4090
========================================================================================================================
Total experiments: 3
```

## Experiment Management Features

### 1. **Automatic Directory Creation**
- Creates complete directory structure for each experiment
- No manual setup required
- Consistent organization across all runs

### 2. **Comprehensive Logging**
- Training logs saved to experiment-specific directory
- TensorBoard logs organized by experiment
- All outputs isolated per experiment

### 3. **Model Artifacts**
- Model weights and checkpoints
- Configuration files
- Training history and metrics
- Detailed model summary with system info

### 4. **Evaluation Results**
- Comprehensive evaluation plots
- Detailed metrics and statistics
- Predictions with metadata
- Subgroup analysis results

## Model Summary Details

The `model_summary.txt` file provides comprehensive information:

```
KOOS-PS Prediction Model - Training Summary
==========================================

Experiment Information:
-----------------------
Experiment ID: 20250911_203021
Start Time: 2025-09-11T20:30:21
Training Duration: 00:15:32 (932.45 seconds)
Experiment Directory: /path/to/model/outputs/20250911_203021

System Information:
------------------
Python Version: 3.12.1
PyTorch Version: 2.5.1
CUDA Available: True
MPS Available: False
Device Count: 1

GPU Information:
GPU 0: NVIDIA GeForce RTX 4090
  Total Memory: 24.00 GB
  Allocated Memory: 8.45 GB
  Reserved Memory: 9.12 GB

Model Architecture:
------------------
Architecture: KOOSPredictionModel
Backbone: densenet121
Has Attention: True
Has Metadata Fusion: True

Model Parameters:
----------------
Total Parameters: 8,234,567
Trainable Parameters: 8,234,567
Non-trainable Parameters: 0
Model Size: 31.45 MB (0.0307 GB)

Configuration:
--------------
Image Size: 224
Batch Size: 32
Learning Rate: 0.0001
Optimizer: adamw
Scheduler: cosine
Epochs: 100
Mixed Precision: True

[... detailed configuration continues ...]
```

## Best Practices

### 1. **Experiment Naming**
- Use descriptive comments in configuration
- Document changes between experiments
- Keep track of what each experiment tested

### 2. **Hyperparameter Tracking**
- All hyperparameters are automatically saved
- Use configuration files for reproducibility
- Document reasoning for parameter choices

### 3. **Model Comparison**
- Use the comparison tools to identify best models
- Track performance trends over time
- Document successful configurations

### 4. **Storage Management**
- Regularly clean up old experiments
- Keep only the most relevant runs
- Archive important experiments

## Advanced Usage

### Custom Experiment Names
```python
from model.utils.experiment_manager import ExperimentManager

# Create experiment with custom base directory
experiment_manager = ExperimentManager("custom_experiments")
```

### Batch Experiment Comparison
```python
# Compare multiple experiments programmatically
experiments = experiment_manager.list_experiments()
best_mae = min(exp for exp in experiments if isinstance(exp['test_mae'], float))
print(f"Best MAE: {best_mae['test_mae']:.4f} in {best_mae['experiment_id']}")
```

### Experiment Metadata Access
```python
# Load experiment metadata
import json
with open("model/outputs/20250911_203021/experiment_metadata.json", 'r') as f:
    metadata = json.load(f)
    
print(f"GPU used: {metadata['system_info']['gpu_info'][0]['name']}")
print(f"Training time: {metadata['training_time_formatted']}")
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Ensure write permissions to output directory
   - Check disk space availability

2. **Missing Experiments**
   - Verify base output directory path
   - Check if experiments completed successfully

3. **Corrupted Metadata**
   - Some experiments may have incomplete metadata
   - Use `--latest` flag to see most recent complete experiment

### Recovery

If an experiment fails partway through:
- Check the logs in the experiment directory
- Partial results may still be available
- Restart with same configuration to continue

## Integration with Existing Workflow

The experiment management system integrates seamlessly with:
- **TensorBoard**: Each experiment has its own TensorBoard logs
- **Weights & Biases**: Experiments are tagged with timestamps
- **Configuration System**: All settings are automatically saved
- **Evaluation Pipeline**: Results are organized by experiment

This system makes it easy to iterate on hyperparameters, compare different architectures, and track the progress of your model development over time.
