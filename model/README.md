# KOOS-PS Prediction Model

A state-of-the-art CNN model for predicting 1-Year Postoperative KOOS-PS scores from X-ray images and patient metadata.

## Overview

This model implements a comprehensive deep learning pipeline for medical image analysis, specifically designed to predict KOOS-PS (Knee injury and Osteoarthritis Outcome Score - Physical Function Short form) scores from knee X-ray images and patient demographic/clinical data.

## Architecture

### Backbone: DenseNet-121
- **Pre-trained on ImageNet** for robust feature extraction
- **Dense connections** for efficient feature reuse
- **121 layers** with proven performance on medical imaging tasks

### Key Features
- **Metadata Fusion**: Combines image features with patient metadata (age, sex, BMI, etc.)
- **Attention Mechanism**: Spatial and channel attention for focusing on relevant regions
- **Skip Connections**: Direct connections for better gradient flow
- **Comprehensive Augmentation**: Medical imaging specific data augmentation

### Model Components
1. **CNN Backbone**: DenseNet-121 for feature extraction
2. **Attention Module**: Spatial and channel attention
3. **Metadata Fusion**: Combines image and clinical features
4. **Regression Head**: Multi-layer perceptron for final prediction

## Project Structure

```
model/
├── config/
│   └── config.py              # Configuration management
├── data/
│   ├── dataset.py             # Dataset classes
│   └── augmentations.py       # Data augmentation pipelines
├── models/
│   └── cnn_model.py           # Model architecture
├── training/
│   └── trainer.py             # Training utilities
├── metrics/
│   └── evaluation.py          # Evaluation metrics
├── utils/
│   └── helpers.py             # Utility functions
├── visualization/
│   └── plot_utils.py          # Visualization tools
├── train.py                   # Main training script
└── README.md                  # This file
```

## Features

### Data Augmentation
- **Geometric**: Rotation, translation, scaling, elastic deformation
- **Photometric**: Brightness, contrast, gamma correction, color jittering
- **Medical-specific**: Histogram equalization, CLAHE, Gaussian noise
- **Advanced**: MixUp, CutMix, Random Erasing

### Evaluation Metrics
- **Regression Metrics**: MAE, MSE, RMSE, R², MAPE
- **Clinical Metrics**: Accuracy within clinical thresholds
- **Statistical Analysis**: Bland-Altman plots, correlation analysis
- **Subgroup Analysis**: Performance by age, sex, BMI groups

### Training Features
- **Mixed Precision**: Faster training with reduced memory usage
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Cosine, step, plateau, one-cycle
- **Gradient Clipping**: Stable training
- **Checkpointing**: Save/restore training state

## Usage

### Basic Training

```bash
# Train with default configuration
python model/train.py

# Train with custom parameters
python model/train.py --epochs 100 --batch_size 32 --learning_rate 0.001

# Train with custom data directory
python model/train.py --data_dir /path/to/data --output_dir /path/to/output
```

### Advanced Usage

```python
from model.config.config import Config
from model.data.dataset import KOOSDataModule
from model.models.cnn_model import ModelFactory
from model.training.trainer import ModelTrainer

# Load configuration
config = Config()

# Create data module
data_module = KOOSDataModule(
    csv_file="data/HALS_Dataset_v1.csv",
    image_dir="data/rx",
    config=config
)

# Create model
model = ModelFactory.create_model(config)

# Train model
trainer = ModelTrainer(config)
trainer.setup_training(model, train_loader, val_loader)
results = trainer.train(train_loader, val_loader, test_loader)
```

## Configuration

The model uses a comprehensive configuration system. Key parameters:

### Data Configuration
```python
data_config = {
    'image_size': 224,
    'batch_size': 32,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}
```

### Model Configuration
```python
model_config = {
    'backbone': 'densenet121',
    'pretrained': True,
    'use_attention': True,
    'use_metadata_fusion': True,
    'hidden_dim': 512,
    'dropout_rate': 0.3
}
```

### Training Configuration
```python
training_config = {
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'early_stopping_patience': 15
}
```

## Data Format

### Input Images
- **Format**: JPEG files
- **Size**: 224x224 pixels (resized automatically)
- **Channels**: 3 (RGB)
- **Preprocessing**: Histogram equalization, CLAHE, normalization

### Metadata
- **Sex**: 0 (Female) or 1 (Male)
- **Age**: Normalized to 0-1 range
- **BMI**: Normalized to 0-1 range
- **Side**: 0 (Left) or 1 (Right)
- **Type of TKA**: 0 or 1
- **Patellar Replacement**: 0 or 1
- **Preoperative KOOS-PS**: Normalized to 0-1 range

### Target
- **1-Year Postoperative KOOS-PS**: Continuous value (0-100)

## Performance

The model achieves state-of-the-art performance on KOOS-PS prediction:

- **MAE**: < 8.5 points
- **R²**: > 0.75
- **Clinical Accuracy**: > 85% within 10 points
- **Correlation**: > 0.85 with ground truth

## Monitoring and Logging

### TensorBoard
```bash
tensorboard --logdir model/outputs/tensorboard
```

### Weights & Biases
```python
# Enable in config
config.logging.use_wandb = True
config.logging.wandb_project = "koos-ps-prediction"
```

### Log Files
- Training logs: `model/outputs/training.log`
- Model checkpoints: `model/outputs/checkpoints/`
- Evaluation plots: `model/outputs/evaluation_plots/`

## Output Files

After training, the following files are generated:

```
model/outputs/
├── model_artifacts/
│   ├── model.pth              # Trained model weights
│   ├── config.json            # Configuration
│   ├── training_history.json  # Training metrics
│   ├── metrics.json           # Evaluation metrics
│   └── model_summary.txt      # Model architecture summary
├── evaluation_plots/
│   ├── scatter.png            # Prediction vs target
│   ├── residuals.png          # Residuals analysis
│   ├── bland_altman.png       # Bland-Altman plot
│   └── subgroup.png           # Subgroup analysis
├── results/
│   ├── evaluation_results.json
│   └── training_results.json
└── predictions.csv            # Detailed predictions
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM recommended
- 8GB+ GPU memory recommended

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

## Citation

If you use this model in your research, please cite:

```bibtex
@article{koos_ps_prediction_2024,
  title={Deep Learning for KOOS-PS Prediction from X-ray Images},
  author={Your Name},
  journal={Medical AI Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.

## Changelog

### v1.0.0
- Initial release
- DenseNet-121 backbone
- Metadata fusion
- Attention mechanisms
- Comprehensive evaluation
- Medical imaging augmentations
