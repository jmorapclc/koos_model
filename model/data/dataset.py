#!/usr/bin/env python3
# model/data/dataset.py
"""
Dataset classes for KOOS-PS prediction model.

This module contains PyTorch Dataset classes for loading and preprocessing
medical X-ray images with associated metadata.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Tuple, Dict, Any, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)

class KOOSDataset(Dataset):
    """
    PyTorch Dataset for KOOS-PS prediction.
    
    Loads X-ray images and associated metadata for regression task.
    """
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        image_size: int = 224,
        augmentations: Optional[A.Compose] = None,
        is_training: bool = True,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file with metadata
            image_dir: Directory containing images
            image_size: Target image size for resizing
            augmentations: Albumentations augmentation pipeline
            is_training: Whether this is training data
            normalize_mean: Mean values for normalization
            normalize_std: Std values for normalization
        """
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.image_size = image_size
        self.augmentations = augmentations
        self.is_training = is_training
        
        # Default ImageNet normalization
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        
        # Load metadata
        self.df = self._load_metadata()
        
        # Filter valid samples
        self.valid_samples = self._filter_valid_samples()
        
        logger.info(f"Loaded {len(self.valid_samples)} valid samples from {csv_file}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from CSV file."""
        try:
            # Load CSV with HALS_MRN as string to prevent float conversion
            df = pd.read_csv(self.csv_file, dtype={'HALS_MRN': str})
            logger.info(f"Loaded metadata with {len(df)} rows and columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {self.csv_file}: {e}")
            raise
    
    def _filter_valid_samples(self) -> List[Dict[str, Any]]:
        """Filter samples that have valid images and metadata."""
        valid_samples = []
        
        for idx, row in self.df.iterrows():
            # Check if image exists
            image_path = os.path.join(self.image_dir, f"{row['HALS_MRN']}.jpeg")
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Check if target value exists and is valid
            target = row.get('1-Year_Postop_KOOS-PS')
            if pd.isna(target) or target < 0 or target > 100:
                logger.warning(f"Invalid target value for {row['HALS_MRN']}: {target}")
                continue
            
            # Extract metadata features
            metadata = self._extract_metadata(row)
            
            valid_samples.append({
                'image_path': image_path,
                'target': float(target),
                'metadata': metadata,
                'hals_mrn': row['HALS_MRN']
            })
        
        return valid_samples
    
    def _extract_metadata(self, row: pd.Series) -> torch.Tensor:
        """
        Extract metadata features from CSV row.
        
        Args:
            row: Pandas Series containing row data
            
        Returns:
            Tensor of metadata features
        """
        features = []
        
        # Sex (0 or 1)
        features.append(float(row.get('Sex', 0)))
        
        # Age (normalized to 0-1)
        age = row.get('Age', 50)
        features.append(min(age / 100.0, 1.0))
        
        # BMI (normalized to 0-1)
        bmi = row.get('BMI', 25)
        features.append(min(bmi / 50.0, 1.0))
        
        # Side (0 or 1)
        features.append(float(row.get('Side', 0)))
        
        # Type of TKA (0 or 1)
        features.append(float(row.get('Type_of_TKA', 0)))
        
        # Patellar Replacement (0 or 1)
        features.append(float(row.get('Patellar_Replacement', 0)))
        
        # Preoperative KOOS-PS (normalized to 0-1)
        preop_koos = row.get('Preoperative_KOOS-PS', 50)
        features.append(min(preop_koos / 100.0, 1.0))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Apply medical imaging preprocessing
            image = self._preprocess_medical_image(image)
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def _preprocess_medical_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply medical imaging specific preprocessing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to grayscale and back to RGB for consistency
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Apply histogram equalization for better contrast
        if len(image.shape) == 3:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for i in range(image.shape[2]):
                image[:, :, i] = clahe.apply(image[:, :, i])
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        return image
    
    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing image, metadata, and target
        """
        sample = self.valid_samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Apply augmentations if provided
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        else:
            # Convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = torch.nn.functional.normalize(
                image, 
                mean=self.normalize_mean, 
                std=self.normalize_std
            )
        
        return {
            'image': image,
            'metadata': sample['metadata'],
            'target': torch.tensor(sample['target'], dtype=torch.float32),
            'hals_mrn': sample['hals_mrn']
        }

class KOOSDataModule:
    """
    Data module for handling train/val/test splits and data loaders.
    """
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        config: Any,
        augmentations: Optional[Dict[str, A.Compose]] = None
    ):
        """
        Initialize data module.
        
        Args:
            csv_file: Path to CSV file
            image_dir: Directory containing images
            config: Configuration object
            augmentations: Dictionary of augmentation pipelines
        """
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.config = config
        self.augmentations = augmentations or {}
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self._create_datasets()
    
    def _create_datasets(self):
        """Create train, validation, and test datasets."""
        # Load full dataset
        full_dataset = KOOSDataset(
            csv_file=self.csv_file,
            image_dir=self.image_dir,
            image_size=self.config.data.image_size,
            augmentations=None,  # Will be set per split
            is_training=True,
            normalize_mean=self.config.data.normalize_mean,
            normalize_std=self.config.data.normalize_std
        )
        
        # Split dataset
        train_size = int(self.config.data.train_ratio * len(full_dataset))
        val_size = int(self.config.data.val_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        # Create indices for splitting
        indices = list(range(len(full_dataset)))
        np.random.seed(self.config.data.random_seed)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # Set augmentations
        if 'train' in self.augmentations:
            self.train_dataset.dataset.augmentations = self.augmentations['train']
        if 'val' in self.augmentations:
            self.val_dataset.dataset.augmentations = self.augmentations['val']
        if 'test' in self.augmentations:
            self.test_dataset.dataset.augmentations = self.augmentations['test']
        
        logger.info(f"Created datasets - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def get_dataloaders(self, device_optimizations: dict = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for train, validation, and test sets with device-specific optimizations.
        
        Args:
            device_optimizations: Device-specific optimization settings
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Use device optimizations if provided, otherwise use config defaults
        if device_optimizations:
            num_workers = device_optimizations.get('num_workers', self.config.data.num_workers)
            pin_memory = device_optimizations.get('pin_memory', self.config.data.pin_memory)
            persistent_workers = device_optimizations.get('persistent_workers', False)
            prefetch_factor = device_optimizations.get('prefetch_factor', 2) if persistent_workers else None
        else:
            num_workers = self.config.data.num_workers
            pin_memory = self.config.data.pin_memory
            persistent_workers = False
            prefetch_factor = None
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader
