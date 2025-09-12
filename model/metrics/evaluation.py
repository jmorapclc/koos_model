"""
Comprehensive evaluation metrics for KOOS-PS prediction model.

This module provides extensive metrics for regression tasks including
statistical measures, clinical relevance metrics, and visualization tools.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class RegressionMetrics:
    """
    Comprehensive regression metrics calculator.
    
    Provides various metrics for evaluating regression performance
    including statistical, clinical, and visualization metrics.
    """
    
    def __init__(self, config: Any):
        """
        Initialize metrics calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.metrics_config = config.metrics
        
    def calculate_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        metadata: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metadata: Optional metadata for subgroup analysis
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics.update(self._calculate_basic_metrics(predictions, targets))
        
        # Advanced statistical metrics
        metrics.update(self._calculate_advanced_metrics(predictions, targets))
        
        # Clinical relevance metrics
        metrics.update(self._calculate_clinical_metrics(predictions, targets))
        
        # Confidence intervals
        if self.metrics_config.use_confidence_intervals:
            metrics.update(self._calculate_confidence_intervals(predictions, targets))
        
        # Subgroup analysis if metadata provided
        if metadata is not None:
            metrics.update(self._calculate_subgroup_metrics(predictions, targets, metadata))
        
        return metrics
    
    def _calculate_basic_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic regression metrics."""
        metrics = {}
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(targets, predictions)
        
        # Mean Squared Error
        metrics['mse'] = mean_squared_error(targets, predictions)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # R-squared
        metrics['r2'] = r2_score(targets, predictions)
        
        # Adjusted R-squared
        n = len(targets)
        p = 1  # Number of predictors (simplified)
        metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        # Mean Absolute Percentage Error
        metrics['mape'] = mean_absolute_percentage_error(targets, predictions) * 100
        
        # Explained Variance Score
        metrics['explained_variance'] = explained_variance_score(targets, predictions)
        
        return metrics
    
    def _calculate_advanced_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate advanced statistical metrics."""
        metrics = {}
        
        # Pearson correlation coefficient
        correlation, p_value = stats.pearsonr(targets, predictions)
        metrics['pearson_correlation'] = correlation
        metrics['pearson_p_value'] = p_value
        
        # Spearman correlation coefficient
        spearman_corr, spearman_p = stats.spearmanr(targets, predictions)
        metrics['spearman_correlation'] = spearman_corr
        metrics['spearman_p_value'] = spearman_p
        
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(targets, predictions)
        metrics['kendall_tau'] = kendall_tau
        metrics['kendall_p_value'] = kendall_p
        
        # Residuals analysis
        residuals = targets - predictions
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = stats.skew(residuals)
        metrics['residual_kurtosis'] = stats.kurtosis(residuals)
        
        # Normality test of residuals
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        metrics['residual_normality_shapiro'] = shapiro_stat
        metrics['residual_normality_p_value'] = shapiro_p
        
        # Durbin-Watson test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import durbin_watson
            dw_stat = durbin_watson(residuals)
            metrics['durbin_watson'] = dw_stat
        except ImportError:
            logger.warning("statsmodels not available for Durbin-Watson test")
        
        return metrics
    
    def _calculate_clinical_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clinically relevant metrics."""
        metrics = {}
        
        # Clinical accuracy thresholds (common in medical literature)
        thresholds = [5, 10, 15, 20]  # Points within which prediction is considered accurate
        
        for threshold in thresholds:
            within_threshold = np.abs(predictions - targets) <= threshold
            accuracy = np.mean(within_threshold) * 100
            metrics[f'accuracy_within_{threshold}'] = accuracy
        
        # Bland-Altman analysis metrics
        mean_values = (predictions + targets) / 2
        differences = predictions - targets
        
        metrics['bland_altman_mean_diff'] = np.mean(differences)
        metrics['bland_altman_std_diff'] = np.std(differences)
        metrics['bland_altman_limits_agreement_upper'] = metrics['bland_altman_mean_diff'] + 1.96 * metrics['bland_altman_std_diff']
        metrics['bland_altman_limits_agreement_lower'] = metrics['bland_altman_mean_diff'] - 1.96 * metrics['bland_altman_std_diff']
        
        # Percentage of points within limits of agreement
        within_limits = (
            (differences >= metrics['bland_altman_limits_agreement_lower']) &
            (differences <= metrics['bland_altman_limits_agreement_upper'])
        )
        metrics['bland_altman_within_limits_pct'] = np.mean(within_limits) * 100
        
        # Clinical significance thresholds
        # KOOS-PS scores typically range from 0-100
        # Changes of 10+ points are often considered clinically significant
        clinically_significant_threshold = 10
        clinically_significant = np.abs(predictions - targets) >= clinically_significant_threshold
        metrics['clinically_significant_errors_pct'] = np.mean(clinically_significant) * 100
        
        return metrics
    
    def _calculate_confidence_intervals(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate confidence intervals for key metrics."""
        metrics = {}
        confidence_level = self.metrics_config.confidence_level
        
        # Bootstrap confidence intervals for key metrics
        n_bootstrap = 1000
        bootstrap_mae = []
        bootstrap_r2 = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(targets), size=len(targets), replace=True)
            pred_boot = predictions[indices]
            target_boot = targets[indices]
            
            bootstrap_mae.append(mean_absolute_error(target_boot, pred_boot))
            bootstrap_r2.append(r2_score(target_boot, pred_boot))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        metrics['mae_ci_lower'] = np.percentile(bootstrap_mae, lower_percentile)
        metrics['mae_ci_upper'] = np.percentile(bootstrap_mae, upper_percentile)
        metrics['r2_ci_lower'] = np.percentile(bootstrap_r2, lower_percentile)
        metrics['r2_ci_upper'] = np.percentile(bootstrap_r2, upper_percentile)
        
        return metrics
    
    def _calculate_subgroup_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        metadata: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for different subgroups."""
        metrics = {}
        
        # Assuming metadata contains: [sex, age, bmi, side, type_tka, patellar_replacement, preop_koos]
        sex = metadata[:, 0]  # 0 or 1
        age = metadata[:, 1] * 100  # Denormalized age
        bmi = metadata[:, 2] * 50  # Denormalized BMI
        
        # Gender-based analysis
        male_mask = sex == 1
        female_mask = sex == 0
        
        if np.any(male_mask):
            metrics['mae_male'] = mean_absolute_error(targets[male_mask], predictions[male_mask])
            metrics['r2_male'] = r2_score(targets[male_mask], predictions[male_mask])
        
        if np.any(female_mask):
            metrics['mae_female'] = mean_absolute_error(targets[female_mask], predictions[female_mask])
            metrics['r2_female'] = r2_score(targets[female_mask], predictions[female_mask])
        
        # Age-based analysis (young vs old)
        age_median = np.median(age)
        young_mask = age < age_median
        old_mask = age >= age_median
        
        if np.any(young_mask):
            metrics['mae_young'] = mean_absolute_error(targets[young_mask], predictions[young_mask])
            metrics['r2_young'] = r2_score(targets[young_mask], predictions[young_mask])
        
        if np.any(old_mask):
            metrics['mae_old'] = mean_absolute_error(targets[old_mask], predictions[old_mask])
            metrics['r2_old'] = r2_score(targets[old_mask], predictions[old_mask])
        
        # BMI-based analysis (normal vs high BMI)
        bmi_threshold = 30  # Obesity threshold
        normal_bmi_mask = bmi < bmi_threshold
        high_bmi_mask = bmi >= bmi_threshold
        
        if np.any(normal_bmi_mask):
            metrics['mae_normal_bmi'] = mean_absolute_error(targets[normal_bmi_mask], predictions[normal_bmi_mask])
            metrics['r2_normal_bmi'] = r2_score(targets[normal_bmi_mask], predictions[normal_bmi_mask])
        
        if np.any(high_bmi_mask):
            metrics['mae_high_bmi'] = mean_absolute_error(targets[high_bmi_mask], predictions[high_bmi_mask])
            metrics['r2_high_bmi'] = r2_score(targets[high_bmi_mask], predictions[high_bmi_mask])
        
        return metrics

class ModelEvaluator:
    """
    Comprehensive model evaluator with visualization capabilities.
    """
    
    def __init__(self, config: Any, experiment_manager=None):
        """
        Initialize model evaluator.
        
        Args:
            config: Configuration object
            experiment_manager: Experiment manager for output directories
        """
        self.config = config
        self.experiment_manager = experiment_manager
        self.metrics_calculator = RegressionMetrics(config)
        
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            dataloader: Data loader for evaluation
            device: Device to run evaluation on
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_metadata = []
        all_hals_mrn = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                targets = batch['target'].to(device)
                metadata = batch['metadata'].to(device)
                hals_mrn = batch['hals_mrn']
                
                # Forward pass
                outputs = model(images, metadata)
                predictions = outputs['predictions']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_metadata.extend(metadata.cpu().numpy())
                all_hals_mrn.extend(hals_mrn)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        metadata = np.array(all_metadata)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            predictions, targets, metadata
        )
        
        # Create results dictionary
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'metadata': metadata,
            'hals_mrn': all_hals_mrn
        }
        
        # Save predictions if requested
        if save_predictions and self.config.metrics.save_predictions:
            self._save_predictions(results)
        
        return results
    
    def _save_predictions(self, results: Dict[str, Any]):
        """Save predictions to CSV file."""
        df = pd.DataFrame({
            'HALS_MRN': results['hals_mrn'],
            'True_KOOS_PS': results['targets'],
            'Predicted_KOOS_PS': results['predictions'],
            'Error': results['targets'] - results['predictions'],
            'Absolute_Error': np.abs(results['targets'] - results['predictions'])
        })
        
        # Add metadata columns
        metadata_df = pd.DataFrame(
            results['metadata'],
            columns=['Sex', 'Age', 'BMI', 'Side', 'Type_of_TKA', 'Patellar_Replacement', 'Preoperative_KOOS_PS']
        )
        
        # Denormalize metadata
        metadata_df['Age'] = metadata_df['Age'] * 100
        metadata_df['BMI'] = metadata_df['BMI'] * 50
        metadata_df['Preoperative_KOOS_PS'] = metadata_df['Preoperative_KOOS_PS'] * 100
        
        # Combine dataframes
        combined_df = pd.concat([df, metadata_df], axis=1)
        
        # Use experiment manager directory if available
        if self.experiment_manager:
            output_file = self.experiment_manager.get_experiment_dir() / "predictions.csv"
        else:
            output_file = f"{self.config.data.output_dir}/predictions.csv"
        
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
    
    def create_evaluation_plots(
        self, 
        results: Dict[str, Any],
        save_path: str = None
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive evaluation plots.
        
        Args:
            results: Evaluation results
            save_path: Path to save plots
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        # 1. Prediction vs Target scatter plot
        figures['scatter'] = self._create_scatter_plot(results)
        
        # 2. Residuals plot
        figures['residuals'] = self._create_residuals_plot(results)
        
        # 3. Bland-Altman plot
        figures['bland_altman'] = self._create_bland_altman_plot(results)
        
        # 4. Error distribution
        figures['error_distribution'] = self._create_error_distribution_plot(results)
        
        # 5. Subgroup analysis plots
        figures['subgroup'] = self._create_subgroup_plots(results)
        
        # 6. Metrics summary
        figures['metrics_summary'] = self._create_metrics_summary_plot(results)
        
        # Save plots if path provided
        if save_path:
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/{name}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        return figures
    
    def _create_scatter_plot(self, results: Dict[str, Any]) -> plt.Figure:
        """Create prediction vs target scatter plot."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        predictions = results['predictions']
        targets = results['targets']
        
        # Create scatter plot
        ax.scatter(targets, predictions, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display R²
        r2 = results['metrics']['r2']
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('True KOOS-PS Score')
        ax.set_ylabel('Predicted KOOS-PS Score')
        ax.set_title('Prediction vs Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _create_residuals_plot(self, results: Dict[str, Any]) -> plt.Figure:
        """Create residuals plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        predictions = results['predictions']
        targets = results['targets']
        residuals = targets - predictions
        
        # Residuals vs predictions
        ax1.scatter(predictions, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predictions')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_bland_altman_plot(self, results: Dict[str, Any]) -> plt.Figure:
        """Create Bland-Altman plot."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        predictions = results['predictions']
        targets = results['targets']
        
        mean_values = (predictions + targets) / 2
        differences = predictions - targets
        
        # Scatter plot
        ax.scatter(mean_values, differences, alpha=0.6)
        
        # Mean difference line
        mean_diff = np.mean(differences)
        ax.axhline(y=mean_diff, color='r', linestyle='-', linewidth=2, label=f'Mean Difference: {mean_diff:.2f}')
        
        # Limits of agreement
        std_diff = np.std(differences)
        upper_limit = mean_diff + 1.96 * std_diff
        lower_limit = mean_diff - 1.96 * std_diff
        
        ax.axhline(y=upper_limit, color='r', linestyle='--', linewidth=2, label=f'Upper Limit: {upper_limit:.2f}')
        ax.axhline(y=lower_limit, color='r', linestyle='--', linewidth=2, label=f'Lower Limit: {lower_limit:.2f}')
        
        ax.set_xlabel('Mean of True and Predicted Values')
        ax.set_ylabel('Difference (Predicted - True)')
        ax.set_title('Bland-Altman Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _create_error_distribution_plot(self, results: Dict[str, Any]) -> plt.Figure:
        """Create error distribution plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        predictions = results['predictions']
        targets = results['targets']
        errors = targets - predictions
        abs_errors = np.abs(errors)
        
        # Error distribution histogram
        ax1.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Absolute error distribution
        ax2.hist(abs_errors, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_xlabel('Absolute Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Absolute Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_subgroup_plots(self, results: Dict[str, Any]) -> plt.Figure:
        """Create subgroup analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metadata = results['metadata']
        predictions = results['predictions']
        targets = results['targets']
        
        # Gender analysis
        sex = metadata[:, 0]
        male_mask = sex == 1
        female_mask = sex == 0
        
        if np.any(male_mask) and np.any(female_mask):
            ax = axes[0]
            ax.scatter(targets[male_mask], predictions[male_mask], alpha=0.6, label='Male', s=50)
            ax.scatter(targets[female_mask], predictions[female_mask], alpha=0.6, label='Female', s=50)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2)
            ax.set_xlabel('True KOOS-PS')
            ax.set_ylabel('Predicted KOOS-PS')
            ax.set_title('Gender Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Age analysis
        age = metadata[:, 1] * 100
        age_median = np.median(age)
        young_mask = age < age_median
        old_mask = age >= age_median
        
        if np.any(young_mask) and np.any(old_mask):
            ax = axes[1]
            ax.scatter(targets[young_mask], predictions[young_mask], alpha=0.6, label='Young', s=50)
            ax.scatter(targets[old_mask], predictions[old_mask], alpha=0.6, label='Old', s=50)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2)
            ax.set_xlabel('True KOOS-PS')
            ax.set_ylabel('Predicted KOOS-PS')
            ax.set_title('Age Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # BMI analysis
        bmi = metadata[:, 2] * 50
        bmi_threshold = 30
        normal_bmi_mask = bmi < bmi_threshold
        high_bmi_mask = bmi >= bmi_threshold
        
        if np.any(normal_bmi_mask) and np.any(high_bmi_mask):
            ax = axes[2]
            ax.scatter(targets[normal_bmi_mask], predictions[normal_bmi_mask], alpha=0.6, label='Normal BMI', s=50)
            ax.scatter(targets[high_bmi_mask], predictions[high_bmi_mask], alpha=0.6, label='High BMI', s=50)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2)
            ax.set_xlabel('True KOOS-PS')
            ax.set_ylabel('Predicted KOOS-PS')
            ax.set_title('BMI Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Error by target value
        ax = axes[3]
        ax.scatter(targets, np.abs(targets - predictions), alpha=0.6)
        ax.set_xlabel('True KOOS-PS')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error vs Target Value')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_metrics_summary_plot(self, results: Dict[str, Any]) -> plt.Figure:
        """Create metrics summary visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics = results['metrics']
        
        # Key metrics to display
        key_metrics = {
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R²': metrics['r2'],
            'MAPE': metrics['mape'],
            'Pearson r': metrics['pearson_correlation']
        }
        
        # Create bar plot
        metric_names = list(key_metrics.keys())
        metric_values = list(key_metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Performance Metrics Summary')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits based on metric types
        ax.set_ylim(0, max(metric_values) * 1.2)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
