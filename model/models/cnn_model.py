#!/usr/bin/env python3
# model/models/cnn_model.py
"""
CNN model architecture for KOOS-PS prediction.

This module contains the main CNN model architecture based on DenseNet-121
with additional components for metadata fusion and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AttentionModule(nn.Module):
    """
    Attention module for focusing on important image regions.
    
    Implements both spatial and channel attention mechanisms.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize attention module.
        
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for channel attention
        """
        super(AttentionModule, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention module.
        
        Args:
            x: Input feature map
            
        Returns:
            Attention-weighted feature map
        """
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

class MetadataFusion(nn.Module):
    """
    Module for fusing image features with metadata.
    
    Combines CNN features with patient metadata using various fusion strategies.
    """
    
    def __init__(
        self,
        image_features_dim: int,
        metadata_dim: int,
        hidden_dim: int = 512,
        fusion_type: str = "concat"
    ):
        """
        Initialize metadata fusion module.
        
        Args:
            image_features_dim: Dimension of image features
            metadata_dim: Dimension of metadata
            hidden_dim: Hidden dimension for fusion
            fusion_type: Type of fusion ("concat", "add", "mul", "attention")
        """
        super(MetadataFusion, self).__init__()
        
        self.fusion_type = fusion_type
        self.image_features_dim = image_features_dim
        self.metadata_dim = metadata_dim
        
        if fusion_type == "concat":
            self.fusion_layer = nn.Linear(
                image_features_dim + metadata_dim, 
                hidden_dim
            )
        elif fusion_type == "add":
            assert image_features_dim == metadata_dim, "Dimensions must match for addition"
            self.fusion_layer = nn.Linear(image_features_dim, hidden_dim)
        elif fusion_type == "mul":
            assert image_features_dim == metadata_dim, "Dimensions must match for multiplication"
            self.fusion_layer = nn.Linear(image_features_dim, hidden_dim)
        elif fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=image_features_dim,
                num_heads=8,
                batch_first=True
            )
            self.metadata_projection = nn.Linear(metadata_dim, image_features_dim)
            self.fusion_layer = nn.Linear(image_features_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        image_features: torch.Tensor, 
        metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of metadata fusion.
        
        Args:
            image_features: CNN features [batch_size, image_features_dim]
            metadata: Patient metadata [batch_size, metadata_dim]
            
        Returns:
            Fused features [batch_size, hidden_dim]
        """
        if self.fusion_type == "concat":
            fused = torch.cat([image_features, metadata], dim=1)
            fused = self.fusion_layer(fused)
            
        elif self.fusion_type == "add":
            # Project metadata to same dimension as image features
            metadata_proj = self.metadata_projection(metadata)
            fused = image_features + metadata_proj
            fused = self.fusion_layer(fused)
            
        elif self.fusion_type == "mul":
            # Project metadata to same dimension as image features
            metadata_proj = self.metadata_projection(metadata)
            fused = image_features * metadata_proj
            fused = self.fusion_layer(fused)
            
        elif self.fusion_type == "attention":
            # Project metadata to image feature dimension
            metadata_proj = self.metadata_projection(metadata)
            
            # Use attention mechanism
            # Reshape for attention: [batch_size, 1, image_features_dim]
            image_features_reshaped = image_features.unsqueeze(1)
            metadata_reshaped = metadata_proj.unsqueeze(1)
            
            # Apply attention
            attended_features, _ = self.attention(
                image_features_reshaped,
                metadata_reshaped,
                metadata_reshaped
            )
            
            fused = attended_features.squeeze(1)
            fused = self.fusion_layer(fused)
        
        # Apply dropout and normalization
        fused = self.dropout(fused)
        fused = self.norm(fused)
        
        return fused

class KOOSPredictionModel(nn.Module):
    """
    Main CNN model for KOOS-PS prediction.
    
    Based on DenseNet-121 backbone with metadata fusion and attention mechanisms.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the model.
        
        Args:
            config: Configuration object containing model parameters
        """
        super(KOOSPredictionModel, self).__init__()
        
        self.config = config
        self.model_config = config.model
        
        # Load backbone
        self.backbone = self._create_backbone()
        
        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()
        
        # Attention module (only for spatial attention on feature maps)
        if self.model_config.use_attention:
            # For DenseNet-121, features have 1024 channels
            self.attention = AttentionModule(self.feature_dim)
        else:
            self.attention = None
        
        # Get the flattened feature dimension for metadata fusion
        self.flattened_feature_dim = self._get_flattened_feature_dim()
        
        # Metadata fusion
        if self.model_config.use_metadata_fusion:
            self.metadata_fusion = MetadataFusion(
                image_features_dim=self.flattened_feature_dim,
                metadata_dim=self.model_config.num_metadata_features,
                hidden_dim=self.model_config.hidden_dim,
                fusion_type="concat"
            )
            fusion_output_dim = self.model_config.hidden_dim
        else:
            # When metadata fusion is disabled, use the actual feature dimension
            # This is the dimension after global average pooling
            fusion_output_dim = self.flattened_feature_dim
        
        # Skip connections
        if self.model_config.use_skip_connections:
            self.skip_connection = nn.Linear(
                self.model_config.num_metadata_features,
                fusion_output_dim
            )
            final_input_dim = fusion_output_dim + fusion_output_dim  # Concatenate
        else:
            final_input_dim = fusion_output_dim
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, self.model_config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.model_config.dropout_rate),
            
            nn.Linear(self.model_config.hidden_dim, self.model_config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.model_config.dropout_rate),
            
            nn.Linear(self.model_config.hidden_dim // 2, self.model_config.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.model_config.dropout_rate),
            
            nn.Linear(self.model_config.hidden_dim // 4, self.model_config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized KOOSPredictionModel with {self._count_parameters()} parameters")
    
    def _create_backbone(self) -> nn.Module:
        """
        Create the backbone CNN model.
        
        Returns:
            Backbone model
        """
        backbone_name = self.model_config.backbone.lower()
        
        if backbone_name == "densenet121":
            model = models.densenet121(pretrained=self.model_config.pretrained)
            # Remove the classifier
            model = nn.Sequential(*list(model.features.children()))
            
        elif backbone_name == "resnet50":
            model = models.resnet50(pretrained=self.model_config.pretrained)
            # Remove the classifier and avgpool
            model = nn.Sequential(*list(model.children())[:-2])
            
        elif backbone_name == "efficientnet_b0":
            try:
                import torchvision.models as efficientnet_models
                model = efficientnet_models.efficientnet_b0(pretrained=self.model_config.pretrained)
                # Remove the classifier
                model = nn.Sequential(*list(model.features.children()))
            except ImportError:
                logger.warning("EfficientNet not available, falling back to DenseNet-121")
                model = models.densenet121(pretrained=self.model_config.pretrained)
                model = nn.Sequential(*list(model.features.children()))
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Freeze backbone if specified
        if self.model_config.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")
        
        return model
    
    def _get_feature_dim(self) -> int:
        """
        Get the feature dimension from the backbone (number of channels).
        
        Returns:
            Number of channels in feature maps
        """
        # Create a dummy input to get feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # For DenseNet, features are [batch, channels, height, width]
            # We need the number of channels for attention
            if len(features.shape) == 4:  # [B, C, H, W]
                return features.size(1)  # Return number of channels
            else:  # [B, features]
                return features.size(1)
    
    def _get_flattened_feature_dim(self) -> int:
        """
        Get the flattened feature dimension from the backbone after global average pooling.
        
        Returns:
            Flattened feature dimension
        """
        # Create a dummy input to get feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # Apply global average pooling first
            features = F.adaptive_avg_pool2d(features, (1, 1))
            # Return the flattened size
            return features.view(features.size(0), -1).size(1)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self, 
        images: torch.Tensor, 
        metadata: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            metadata: Patient metadata [batch_size, metadata_dim]
            
        Returns:
            Dictionary containing predictions and attention maps
        """
        # Extract features using backbone
        features = self.backbone(images)  # [batch_size, feature_dim, h, w]
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)  # [batch_size, feature_dim]
        
        # Fuse with metadata
        if self.model_config.use_metadata_fusion and hasattr(self, 'metadata_fusion'):
            fused_features = self.metadata_fusion(features, metadata)
        else:
            # When metadata fusion is disabled, features are already the correct size
            fused_features = features
        
        # Skip connection
        if self.model_config.use_skip_connections and hasattr(self, 'skip_connection'):
            skip_features = self.skip_connection(metadata)
            fused_features = torch.cat([fused_features, skip_features], dim=1)
        
        # Final prediction
        predictions = self.classifier(fused_features)
        
        # Apply output activation
        if self.model_config.output_activation == "sigmoid":
            predictions = torch.sigmoid(predictions) * 100  # Scale to 0-100
        elif self.model_config.output_activation == "tanh":
            predictions = (torch.tanh(predictions) + 1) * 50  # Scale to 0-100
        # For "linear", no activation is applied
        
        return {
            'predictions': predictions.squeeze(-1),
            'features': features,
            'fused_features': fused_features
        }
    
    def get_attention_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps for visualization.
        
        Args:
            images: Input images
            
        Returns:
            Attention maps
        """
        if self.attention is None:
            return None
        
        with torch.no_grad():
            features = self.backbone(images)
            attention_maps = self.attention(features)
            return attention_maps

class ModelFactory:
    """
    Factory class for creating different model architectures.
    """
    
    @staticmethod
    def create_model(config: Any) -> nn.Module:
        """
        Create a model based on configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Model instance
        """
        model_type = getattr(config.model, 'type', 'koos_prediction')
        
        if model_type == 'koos_prediction':
            return KOOSPredictionModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)) -> str:
        """
        Get a summary of the model architecture.
        
        Args:
            model: Model instance
            input_size: Input size for summary
            
        Returns:
            Model summary string
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = f"""
Model Summary:
==============
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}

Architecture:
{model}
        """
        
        return summary
