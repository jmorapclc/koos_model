#!/usr/bin/env python3
# model/data/augmentations.py
"""
Data augmentation pipelines for medical X-ray images.

This module contains comprehensive augmentation strategies specifically
designed for medical imaging data to improve model robustness and performance.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional
import numpy as np
import cv2
from skimage import exposure, filters
import logging

logger = logging.getLogger(__name__)

class MedicalImageAugmentations:
    """
    Medical image augmentation pipeline using Albumentations.
    
    Provides comprehensive augmentation strategies for X-ray images
    including geometric, photometric, and medical-specific augmentations.
    """
    
    def __init__(self, config: Any):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Configuration object containing augmentation parameters
        """
        self.config = config
        self.aug_config = config.augmentation
        
    def get_train_augmentations(self) -> A.Compose:
        """
        Get training augmentations.
        
        Returns:
            Albumentations compose object for training
        """
        transforms = []
        
        # Basic geometric augmentations
        if self.aug_config.horizontal_flip_prob > 0:
            transforms.append(
                A.HorizontalFlip(p=self.aug_config.horizontal_flip_prob)
            )
        
        if self.aug_config.vertical_flip_prob > 0:
            transforms.append(
                A.VerticalFlip(p=self.aug_config.vertical_flip_prob)
            )
        
        if self.aug_config.rotation_degrees > 0:
            transforms.append(
                A.Rotate(
                    limit=self.aug_config.rotation_degrees,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill_value=0
                )
            )
        
        # Translation and scaling
        if self.aug_config.translation_ratio > 0 or self.aug_config.scale_ratio > 0:
            transforms.append(
                A.Affine(
                    translate_percent=self.aug_config.translation_ratio,
                    scale=(1-self.aug_config.scale_ratio, 1+self.aug_config.scale_ratio),
                    p=0.5,
                    mode=cv2.BORDER_CONSTANT,
                    cval=0
                )
            )
        
        # Elastic transformation for medical images
        if self.aug_config.use_elastic_transform:
            transforms.append(
                A.ElasticTransform(
                    alpha=self.aug_config.elastic_alpha,
                    sigma=self.aug_config.elastic_sigma,
                    alpha_affine=50,
                    p=0.3,
                    mode=cv2.BORDER_CONSTANT,
                    cval=0
                )
            )
        
        # Photometric augmentations
        if self.aug_config.use_color_jitter:
            transforms.append(
                A.ColorJitter(
                    brightness=self.aug_config.brightness_factor,
                    contrast=self.aug_config.contrast_factor,
                    saturation=self.aug_config.saturation_factor,
                    hue=self.aug_config.hue_factor,
                    p=0.5
                )
            )
        
        # Gaussian noise
        if self.aug_config.use_gaussian_noise:
            transforms.append(
                A.GaussNoise(
                    var_limit=(0, self.aug_config.noise_std * 255),
                    p=0.3
                )
            )
        
        # Medical imaging specific augmentations
        transforms.append(
            A.Lambda(
                image=self._histogram_equalization,
                p=0.3 if self.aug_config.use_histogram_equalization else 0
            )
        )
        
        transforms.append(
            A.Lambda(
                image=self._clahe_enhancement,
                p=0.3 if self.aug_config.use_clahe else 0
            )
        )
        
        # Random brightness and contrast
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            )
        )
        
        # Random gamma correction
        transforms.append(
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            )
        )
        
        # Blur augmentation
        transforms.append(
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.2)
        )
        
        # Cutout for regularization
        transforms.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            )
        )
        
        # Resize to target size
        transforms.append(
            A.Resize(
                height=self.config.data.image_size,
                width=self.config.data.image_size,
                interpolation=cv2.INTER_LINEAR
            )
        )
        
        # Normalize and convert to tensor
        transforms.extend([
            A.Normalize(
                mean=self.config.data.normalize_mean,
                std=self.config.data.normalize_std
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def get_validation_augmentations(self) -> A.Compose:
        """
        Get validation augmentations (minimal augmentations).
        
        Returns:
            Albumentations compose object for validation
        """
        transforms = [
            A.Resize(
                height=self.config.data.image_size,
                width=self.config.data.image_size,
                interpolation=cv2.INTER_LINEAR
            ),
            A.Normalize(
                mean=self.config.data.normalize_mean,
                std=self.config.data.normalize_std
            ),
            ToTensorV2()
        ]
        
        return A.Compose(transforms)
    
    def get_test_augmentations(self) -> A.Compose:
        """
        Get test augmentations (no augmentations, just preprocessing).
        
        Returns:
            Albumentations compose object for test
        """
        return self.get_validation_augmentations()
    
    def _histogram_equalization(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image: Input image
            **kwargs: Additional arguments
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Apply to each channel separately
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = exposure.equalize_hist(image[:, :, i])
            return enhanced
        else:
            return exposure.equalize_hist(image)
    
    def _clahe_enhancement(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            **kwargs: Additional arguments
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def get_augmentation_pipelines(self) -> Dict[str, A.Compose]:
        """
        Get all augmentation pipelines.
        
        Returns:
            Dictionary containing train, val, and test augmentation pipelines
        """
        return {
            'train': self.get_train_augmentations(),
            'val': self.get_validation_augmentations(),
            'test': self.get_test_augmentations()
        }

class AdvancedAugmentations:
    """
    Advanced augmentation techniques for medical imaging.
    
    Includes more sophisticated augmentation strategies that can be
    applied selectively based on the specific medical imaging task.
    """
    
    @staticmethod
    def mixup_augmentation(images: np.ndarray, targets: np.ndarray, alpha: float = 0.2) -> tuple:
        """
        Apply MixUp augmentation.
        
        Args:
            images: Batch of images
            targets: Batch of targets
            alpha: MixUp parameter
            
        Returns:
            Mixed images and targets
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        return mixed_images, mixed_targets
    
    @staticmethod
    def cutmix_augmentation(images: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> tuple:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images
            targets: Batch of targets
            alpha: CutMix parameter
            
        Returns:
            Mixed images and targets
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.shape[0]
        index = np.random.permutation(batch_size)
        
        # Get bounding box
        h, w = images.shape[2], images.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        mixed_images = images.copy()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        return mixed_images, mixed_targets
    
    @staticmethod
    def random_erasing(images: np.ndarray, probability: float = 0.5) -> np.ndarray:
        """
        Apply random erasing augmentation.
        
        Args:
            images: Batch of images
            probability: Probability of applying random erasing
            
        Returns:
            Augmented images
        """
        if np.random.random() > probability:
            return images
        
        batch_size, channels, height, width = images.shape
        
        for i in range(batch_size):
            # Random erasing parameters
            area = height * width
            target_area = np.random.uniform(0.02, 0.4) * area
            aspect_ratio = np.random.uniform(0.3, 3.0)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if h < height and w < width:
                y = np.random.randint(0, height - h)
                x = np.random.randint(0, width - w)
                images[i, :, y:y+h, x:x+w] = np.random.uniform(0, 1, (channels, h, w))
        
        return images

def create_augmentation_pipelines(config: Any) -> Dict[str, A.Compose]:
    """
    Create augmentation pipelines based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary of augmentation pipelines
    """
    augmentation_manager = MedicalImageAugmentations(config)
    return augmentation_manager.get_augmentation_pipelines()
