"""
Visualization utilities for KOOS-PS prediction model.

This module provides comprehensive visualization tools for model analysis,
including attention maps, feature visualizations, and interactive plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """
    Visualize attention maps and model interpretability.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize attention visualizer.
        
        Args:
            model: Trained model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_attention_maps(
        self, 
        images: torch.Tensor,
        metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate attention maps for given images.
        
        Args:
            images: Input images
            metadata: Input metadata
            
        Returns:
            Attention maps
        """
        with torch.no_grad():
            attention_maps = self.model.get_attention_maps(images)
        return attention_maps
    
    def visualize_attention(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create attention visualization.
        
        Args:
            images: Input images
            metadata: Input metadata
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Generate attention maps
        attention_maps = self.generate_attention_maps(images, metadata)
        
        if attention_maps is None:
            logger.warning("No attention maps available")
            return None
        
        batch_size = images.size(0)
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Attention map
            att_map = attention_maps[i].cpu().numpy()
            if len(att_map.shape) == 3:
                att_map = np.mean(att_map, axis=0)
            
            axes[i, 1].imshow(att_map, cmap='hot')
            axes[i, 1].set_title('Attention Map')
            axes[i, 1].axis('off')
            
            # Overlay
            overlay = img.copy()
            att_resized = cv2.resize(att_map, (img.shape[1], img.shape[0]))
            att_resized = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min())
            
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], att_resized)
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
        return fig

class FeatureVisualizer:
    """
    Visualize learned features and model representations.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize feature visualizer.
        
        Args:
            model: Trained model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_features(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from different layers.
        
        Args:
            images: Input images
            metadata: Input metadata
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        with torch.no_grad():
            # Extract backbone features
            backbone_features = self.model.backbone(images)
            features['backbone'] = backbone_features
            
            # Apply attention if available
            if hasattr(self.model, 'attention') and self.model.attention is not None:
                attended_features = self.model.attention(backbone_features)
                features['attended'] = attended_features
            
            # Global average pooling
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(
                backbone_features, (1, 1)
            ).view(backbone_features.size(0), -1)
            features['pooled'] = pooled_features
            
            # Final model output
            model_output = self.model(images, metadata)
            features['predictions'] = model_output['predictions']
            features['fused_features'] = model_output['fused_features']
        
        return features
    
    def visualize_feature_maps(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        layer_name: str = 'backbone',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize feature maps from specified layer.
        
        Args:
            images: Input images
            metadata: Input metadata
            layer_name: Name of layer to visualize
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        features = self.extract_features(images, metadata)
        feature_maps = features[layer_name]
        
        if len(feature_maps.shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = feature_maps.shape
            n_cols = min(8, channels)
            n_rows = (channels + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i in range(min(channels, len(axes))):
                feature_map = feature_maps[0, i].cpu().numpy()  # First sample
                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(channels, len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature maps visualization saved to {save_path}")
        
        return fig

class InteractiveVisualizer:
    """
    Create interactive visualizations using Plotly.
    """
    
    def __init__(self):
        """Initialize interactive visualizer."""
        pass
    
    def create_prediction_scatter(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: np.ndarray,
        title: str = "Prediction vs Target"
    ) -> go.Figure:
        """
        Create interactive scatter plot.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metadata: Patient metadata
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Prepare data
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # Create hover text
        hover_text = []
        for i in range(len(predictions)):
            text = f"""
            Patient {i+1}<br>
            True KOOS-PS: {targets[i]:.2f}<br>
            Predicted KOOS-PS: {predictions[i]:.2f}<br>
            Error: {errors[i]:.2f}<br>
            Sex: {'Male' if metadata[i, 0] == 1 else 'Female'}<br>
            Age: {metadata[i, 1] * 100:.0f}<br>
            BMI: {metadata[i, 2] * 50:.1f}
            """
            hover_text.append(text)
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=targets,
            y=predictions,
            mode='markers',
            marker=dict(
                size=8,
                color=abs_errors,
                colorscale='Viridis',
                colorbar=dict(title="Absolute Error"),
                line=dict(width=1, color='black')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Predictions'
        ))
        
        # Add perfect prediction line
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='True KOOS-PS Score',
            yaxis_title='Predicted KOOS-PS Score',
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_residuals_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: np.ndarray
    ) -> go.Figure:
        """
        Create interactive residuals plot.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metadata: Patient metadata
            
        Returns:
            Plotly figure
        """
        residuals = predictions - targets
        
        # Create hover text
        hover_text = []
        for i in range(len(predictions)):
            text = f"""
            Patient {i+1}<br>
            True KOOS-PS: {targets[i]:.2f}<br>
            Predicted KOOS-PS: {predictions[i]:.2f}<br>
            Residual: {residuals[i]:.2f}<br>
            Sex: {'Male' if metadata[i, 0] == 1 else 'Female'}<br>
            Age: {metadata[i, 1] * 100:.0f}
            """
            hover_text.append(text)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions,
            y=residuals,
            mode='markers',
            marker=dict(
                size=8,
                color=metadata[:, 0],  # Color by sex
                colorscale=['pink', 'blue'],
                colorbar=dict(title="Sex (0=Female, 1=Male)"),
                line=dict(width=1, color='black')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Residuals'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Residuals Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals (Predicted - True)",
            width=800,
            height=600
        )
        
        return fig
    
    def create_metrics_dashboard(
        self,
        metrics: Dict[str, float],
        training_history: Dict[str, List]
    ) -> go.Figure:
        """
        Create interactive metrics dashboard.
        
        Args:
            metrics: Evaluation metrics
            training_history: Training history
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Progress', 'Key Metrics', 'Learning Rate', 'Loss Difference'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training progress
        epochs = list(range(1, len(training_history['train_loss']) + 1))
        
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['train_loss'], 
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['val_loss'], 
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Key metrics bar chart
        key_metrics = ['mae', 'rmse', 'r2', 'pearson_correlation']
        metric_values = [metrics.get(metric, 0) for metric in key_metrics]
        
        fig.add_trace(
            go.Bar(x=key_metrics, y=metric_values, name='Metrics'),
            row=1, col=2
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['learning_rate'], 
                      name='Learning Rate', line=dict(color='green')),
            row=2, col=1
        )
        
        # Loss difference
        loss_diff = np.array(training_history['val_loss']) - np.array(training_history['train_loss'])
        fig.add_trace(
            go.Scatter(x=epochs, y=loss_diff, 
                      name='Val - Train Loss', line=dict(color='purple')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Model Training Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig

def create_comprehensive_report(
    model: torch.nn.Module,
    evaluation_results: Dict[str, Any],
    training_history: Dict[str, List],
    config: Any,
    save_dir: str
) -> None:
    """
    Create comprehensive model report with all visualizations.
    
    Args:
        model: Trained model
        evaluation_results: Evaluation results
        training_history: Training history
        config: Configuration object
        save_dir: Directory to save report
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating comprehensive model report...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attention_viz = AttentionVisualizer(model, device)
    feature_viz = FeatureVisualizer(model, device)
    interactive_viz = InteractiveVisualizer()
    
    # Create visualizations
    predictions = evaluation_results['predictions']
    targets = evaluation_results['targets']
    metadata = evaluation_results['metadata']
    metrics = evaluation_results['metrics']
    
    # 1. Attention maps (if available)
    try:
        # Use a small sample for attention visualization
        sample_size = min(4, len(predictions))
        sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
        
        # This would need actual image data - placeholder for now
        # attention_fig = attention_viz.visualize_attention(
        #     sample_images, sample_metadata, 
        #     save_path=str(save_dir / "attention_maps.png")
        # )
        pass
    except Exception as e:
        logger.warning(f"Could not create attention maps: {e}")
    
    # 2. Interactive plots
    scatter_fig = interactive_viz.create_prediction_scatter(
        predictions, targets, metadata
    )
    scatter_fig.write_html(str(save_dir / "prediction_scatter.html"))
    
    residuals_fig = interactive_viz.create_residuals_plot(
        predictions, targets, metadata
    )
    residuals_fig.write_html(str(save_dir / "residuals_plot.html"))
    
    dashboard_fig = interactive_viz.create_metrics_dashboard(
        metrics, training_history
    )
    dashboard_fig.write_html(str(save_dir / "metrics_dashboard.html"))
    
    # 3. Save metrics summary
    with open(save_dir / "metrics_summary.txt", 'w') as f:
        f.write("Model Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Mean Absolute Error: {metrics['mae']:.4f}\n")
        f.write(f"Root Mean Squared Error: {metrics['rmse']:.4f}\n")
        f.write(f"R-squared: {metrics['r2']:.4f}\n")
        f.write(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}\n")
        f.write(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%\n")
        
        f.write(f"\nClinical Accuracy:\n")
        for threshold in [5, 10, 15, 20]:
            f.write(f"Within {threshold} points: {metrics.get(f'accuracy_within_{threshold}', 0):.2f}%\n")
    
    logger.info(f"Comprehensive report saved to {save_dir}")
