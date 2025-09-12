#!/usr/bin/env python3
# model/list_experiments.py
"""
Script to list and compare training experiments.

This script provides functionality to view all training experiments,
compare their results, and manage experiment directories.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Add model directory to path
sys.path.append(str(Path(__file__).parent))

from utils.experiment_manager import ExperimentManager

def list_experiments(base_output_dir: str) -> List[Dict[str, Any]]:
    """
    List all experiments with their metadata.
    
    Args:
        base_output_dir: Base output directory
        
    Returns:
        List of experiment information
    """
    experiment_manager = ExperimentManager(base_output_dir)
    experiments = experiment_manager.list_experiments()
    
    experiment_info = []
    
    for exp_dir in experiments:
        try:
            # Load experiment metadata
            metadata_file = exp_dir / 'experiment_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load metrics if available
                metrics_file = exp_dir / 'model_artifacts' / 'metrics.json'
                metrics = {}
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                
                experiment_info.append({
                    'experiment_id': metadata.get('experiment_id', exp_dir.name),
                    'start_time': metadata.get('start_time', 'Unknown'),
                    'training_time': metadata.get('training_time_formatted', 'Unknown'),
                    'best_val_loss': metadata.get('model_info', {}).get('best_val_loss', 'Unknown'),
                    'test_mae': metrics.get('mae', 'Unknown'),
                    'test_r2': metrics.get('r2', 'Unknown'),
                    'total_parameters': metadata.get('model_info', {}).get('total_parameters', 'Unknown'),
                    'gpu_used': metadata.get('system_info', {}).get('gpu_info', [{}])[0].get('name', 'CPU'),
                    'directory': str(exp_dir)
                })
            else:
                # Basic info without metadata
                experiment_info.append({
                    'experiment_id': exp_dir.name,
                    'start_time': 'Unknown',
                    'training_time': 'Unknown',
                    'best_val_loss': 'Unknown',
                    'test_mae': 'Unknown',
                    'test_r2': 'Unknown',
                    'total_parameters': 'Unknown',
                    'gpu_used': 'Unknown',
                    'directory': str(exp_dir)
                })
        except Exception as e:
            print(f"Error loading experiment {exp_dir.name}: {e}")
            continue
    
    return experiment_info

def print_experiments_table(experiments: List[Dict[str, Any]]):
    """
    Print experiments in a formatted table.
    
    Args:
        experiments: List of experiment information
    """
    if not experiments:
        print("No experiments found.")
        return
    
    # Create DataFrame for better formatting
    df = pd.DataFrame(experiments)
    
    # Reorder columns
    columns = ['experiment_id', 'start_time', 'training_time', 'best_val_loss', 
               'test_mae', 'test_r2', 'total_parameters', 'gpu_used']
    df = df[columns]
    
    # Format numeric columns
    for col in ['best_val_loss', 'test_mae', 'test_r2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].round(4)
    
    # Format parameters
    df['total_parameters'] = df['total_parameters'].apply(
        lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x)
    )
    
    print("\n" + "="*120)
    print("TRAINING EXPERIMENTS SUMMARY")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    print(f"Total experiments: {len(experiments)}")
    print("="*120)

def compare_experiments(experiments: List[Dict[str, Any]], metric: str = 'test_mae'):
    """
    Compare experiments by a specific metric.
    
    Args:
        experiments: List of experiment information
        metric: Metric to compare by
    """
    if not experiments:
        print("No experiments to compare.")
        return
    
    # Filter experiments with valid metrics
    valid_experiments = []
    for exp in experiments:
        if isinstance(exp.get(metric), (int, float)):
            valid_experiments.append(exp)
    
    if not valid_experiments:
        print(f"No experiments with valid {metric} data.")
        return
    
    # Sort by metric
    valid_experiments.sort(key=lambda x: x[metric])
    
    print(f"\nEXPERIMENTS RANKED BY {metric.upper()}:")
    print("-" * 80)
    
    for i, exp in enumerate(valid_experiments, 1):
        print(f"{i:2d}. {exp['experiment_id']} - {metric}: {exp[metric]:.4f}")
    
    # Show best and worst
    best = valid_experiments[0]
    worst = valid_experiments[-1]
    
    print(f"\nBest: {best['experiment_id']} ({best[metric]:.4f})")
    print(f"Worst: {worst['experiment_id']} ({worst[metric]:.4f})")
    print(f"Improvement: {worst[metric] - best[metric]:.4f}")

def show_experiment_details(experiment_id: str, base_output_dir: str):
    """
    Show detailed information about a specific experiment.
    
    Args:
        experiment_id: Experiment ID
        base_output_dir: Base output directory
    """
    experiment_dir = Path(base_output_dir) / experiment_id
    
    if not experiment_dir.exists():
        print(f"Experiment {experiment_id} not found.")
        return
    
    print(f"\nDETAILED INFORMATION FOR EXPERIMENT: {experiment_id}")
    print("="*80)
    
    # Load metadata
    metadata_file = experiment_dir / 'experiment_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Start Time: {metadata.get('start_time', 'Unknown')}")
        print(f"End Time: {metadata.get('end_time', 'Unknown')}")
        print(f"Training Duration: {metadata.get('training_time_formatted', 'Unknown')}")
        
        # System info
        system_info = metadata.get('system_info', {})
        print(f"\nSystem Information:")
        print(f"  Python Version: {system_info.get('python_version', 'Unknown')}")
        print(f"  PyTorch Version: {system_info.get('pytorch_version', 'Unknown')}")
        print(f"  CUDA Available: {system_info.get('cuda_available', 'Unknown')}")
        
        gpu_info = system_info.get('gpu_info', [])
        if gpu_info:
            print(f"  GPU Information:")
            for gpu in gpu_info:
                print(f"    GPU {gpu.get('device_id', '?')}: {gpu.get('name', 'Unknown')}")
                print(f"      Memory: {gpu.get('memory_total', 0) / 1024**3:.2f} GB")
        
        # Model info
        model_info = metadata.get('model_info', {})
        print(f"\nModel Information:")
        print(f"  Total Parameters: {model_info.get('total_parameters', 'Unknown'):,}")
        print(f"  Trainable Parameters: {model_info.get('trainable_parameters', 'Unknown'):,}")
        print(f"  Model Size: {model_info.get('model_size_mb', 'Unknown'):.2f} MB")
        print(f"  Architecture: {model_info.get('architecture', 'Unknown')}")
        print(f"  Backbone: {model_info.get('backbone', 'Unknown')}")
        print(f"  Has Attention: {model_info.get('has_attention', 'Unknown')}")
        print(f"  Has Metadata Fusion: {model_info.get('has_metadata_fusion', 'Unknown')}")
    
    # Load metrics
    metrics_file = experiment_dir / 'model_artifacts' / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"\nPerformance Metrics:")
        print(f"  MAE: {metrics.get('mae', 'Unknown'):.4f}")
        print(f"  RMSE: {metrics.get('rmse', 'Unknown'):.4f}")
        print(f"  R²: {metrics.get('r2', 'Unknown'):.4f}")
        print(f"  Pearson Correlation: {metrics.get('pearson_correlation', 'Unknown'):.4f}")
        print(f"  MAPE: {metrics.get('mape', 'Unknown'):.2f}%")
    
    # Show available files
    print(f"\nAvailable Files:")
    for file_path in experiment_dir.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(experiment_dir)
            file_size = file_path.stat().st_size
            if file_size > 1024*1024:
                size_str = f"{file_size / 1024**2:.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} B"
            print(f"  {relative_path} ({size_str})")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='List and compare training experiments')
    parser.add_argument('--base_dir', type=str, default='model/outputs', 
                       help='Base output directory')
    parser.add_argument('--compare', type=str, choices=['test_mae', 'test_r2', 'best_val_loss'],
                       help='Compare experiments by metric')
    parser.add_argument('--details', type=str, help='Show details for specific experiment ID')
    parser.add_argument('--latest', action='store_true', help='Show details for latest experiment')
    
    args = parser.parse_args()
    
    # List experiments
    experiments = list_experiments(args.base_dir)
    
    if args.details:
        show_experiment_details(args.details, args.base_dir)
    elif args.latest and experiments:
        latest_exp = experiments[0]  # Already sorted by newest first
        show_experiment_details(latest_exp['experiment_id'], args.base_dir)
    elif args.compare:
        compare_experiments(experiments, args.compare)
    else:
        print_experiments_table(experiments)

if __name__ == "__main__":
    main()
