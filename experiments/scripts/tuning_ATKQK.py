import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import itertools
import random
from datetime import datetime
from run_ATKQK import main, parse_arguments

def create_parameter_configurations():
    """Define parameter configurations for different datasets"""
    # Define datasets and their modalities to test
    dataset_modalities = {
        "AD_CN": [["PET"], ["GM"], ["CSF"], ["MRI"]],
        "ROSMAP": [["meth"], ["mRNA"], ["miRNA"]]
    }
    
    # Parameters to tune for each dataset-modality combo
    param_configs = {
        # Core architecture parameters
        "num_layers": [2, 3, 4, 5, 6],
        "num_head": [2, 3, 4, 5],
        "top_k": [1, 2, 3, 5, 7, 10, 20, 50, 100],
        
        # Training parameters
        "lr": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001],
        "patience": [20, 30, 50],
        
        # Model parameters
        "reg_coef": [0, 0.01, 0.05, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "embed_dim": [64, 128, 256]
    }
    
    return dataset_modalities, param_configs

def run_experiment(args):
    """Run a single experiment with given parameters and return results"""
    start_time = time.time()
    
    try:
        # Run the model
        df_summary, cf_summary, acc, summary_df = main(args)
        
        # Get metrics
        auc = summary_df.loc[summary_df['Metric'] == 'AUC', 'Mean'].values[0]
        sensitivity = summary_df.loc[summary_df['Metric'] == 'Sensitivity', 'Mean'].values[0]
        specificity = summary_df.loc[summary_df['Metric'] == 'Specificity', 'Mean'].values[0]
        
        runtime = time.time() - start_time
        
        return {
            "accuracy": acc,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "runtime_seconds": runtime,
            "success": True,
            "error": None
        }
    except Exception as e:
        runtime = time.time() - start_time
        print(f"Error in experiment: {str(e)}")
        return {
            "accuracy": 0,
            "auc": 0,
            "sensitivity": 0,
            "specificity": 0,
            "runtime_seconds": runtime,
            "success": False,
            "error": str(e)
        }

def get_param_config_description(param_config):
    """Get a string description of parameter configuration"""
    return "_".join([f"{k}-{v}" for k, v in param_config.items()])

def create_result_row(args, experiment_results, param_config):
    """Create a result dataframe row from experiment results and parameter config"""
    # Base result with experiment metrics
    result = {
        "dataset": args.dataset,
        "modality": "_".join(args.modality),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Add all experiment parameters
    for param, value in vars(args).items():
        # Skip complex objects and lists
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[param] = value
        elif isinstance(value, list):
            result[param] = "_".join(map(str, value))
    
    # Add experiment results
    result.update(experiment_results)
    
    # Add specific parameter configuration being tested
    result.update(param_config)
    
    return result

def grid_search(args, param_grid, output_dir):
    """
    Perform grid search over specified parameter combinations
    
    param_grid: Dictionary of parameters to grid search over
                e.g. {"num_layers": [2, 4], "num_head": [2, 4]}
    """
    # Create parameter-specific output directory (consistent with sequential)
    param_output_dir = f"{output_dir}/grid"
    os.makedirs(param_output_dir, exist_ok=True)
    
    # Results file
    results_file = f"{param_output_dir}/grid_search_results.csv"
    
    # Check for existing results
    existing_results = pd.DataFrame()
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        print(f"Found existing results: {len(existing_results)} experiments")
    
    # Create grid of parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Grid Search: Testing {total_combinations} parameter combinations")
    
    # Results storage
    all_results = []
    
    # Iterate through all combinations
    for i, values in enumerate(itertools.product(*param_values)):
        # Create parameter configuration
        param_config = dict(zip(param_names, values))
        
        # Check if this combination was already tested
        if not existing_results.empty:
            skip = True
            for param, value in param_config.items():
                if param not in existing_results.columns or not any(existing_results[param] == value):
                    skip = False
                    break
            
            if skip:
                matched_rows = existing_results
                for param, value in param_config.items():
                    matched_rows = matched_rows[matched_rows[param] == value]
                
                if len(matched_rows) > 0:
                    print(f"Skipping combination {i+1}/{total_combinations}: {param_config} (already tested)")
                    all_results.append(matched_rows.to_dict('records')[0])
                    continue
        
        # Set parameters
        for param, value in param_config.items():
            setattr(args, param, value)
        
        # Print progress
        print(f"Running combination {i+1}/{total_combinations}: {param_config}")
        
        # Run experiment
        experiment_results = run_experiment(args)
        
        # Create result row
        result = create_result_row(args, experiment_results, param_config)
        all_results.append(result)
        
        # Save incrementally
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        
        # Print results
        if experiment_results["success"]:
            print(f"  Results: acc={result['accuracy']:.4f}, auc={result['auc']:.4f}")
        else:
            print(f"  Error: {experiment_results['error']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Find best configuration
    if "accuracy" in results_df.columns and len(results_df) > 0:
        best_idx = results_df["accuracy"].idxmax()
        best_config = results_df.iloc[best_idx]
        
        print("\nBest configuration:")
        for param in param_names:
            print(f"  {param}: {best_config[param]}")
        print(f"  Accuracy: {best_config['accuracy']:.4f}")
        print(f"  AUC: {best_config['auc']:.4f}")
        
        # Save best configuration
        best_params = {param: best_config[param] for param in param_names}
        best_params.update({
            "dataset": args.dataset,
            "modality": "_".join(args.modality) if isinstance(args.modality[0], str) else "_".join(args.modality[0]),
            "best_accuracy": best_config["accuracy"],  # Change to match sequential_tuning format
            "best_auc": best_config["auc"]            # Change to match sequential_tuning format
        })
        
        pd.DataFrame([best_params]).to_csv(f"{output_dir}/best_parameters_grid_search.csv", index=False)
        
        # Return best_params dict instead of results_df to match sequential_tuning return type
        return best_params
    
    # If no valid results, return empty best_params with same format as sequential_tuning
    return {"dataset": args.dataset, "modality": "_".join(args.modality) if isinstance(args.modality[0], str) else "_".join(args.modality[0]), 
            "best_accuracy": 0, "best_auc": 0}

def sequential_tuning(args, param_configs, output_dir):
    """Tune parameters sequentially, optimizing one at a time"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters in order of importance 
    tuning_order = [
        "lr",         # Learning rate is usually most important
        "num_layers", # Model architecture - depth
        "num_head",   # Model architecture - attention mechanism
        "top_k",      # Graph construction
        "reg_coef",   # Regularization
        "patience",   # Training procedure
        "embed_dim",  # Model capacity
    ]
    
    # Only include parameters that are in our param_configs
    tuning_order = [p for p in tuning_order if p in param_configs]
    
    # Track best parameters
    best_params = {
        "dataset": args.dataset, 
        "modality": "_".join(args.modality),
        "best_accuracy": 0, 
        "best_auc": 0
    }
    
    # For each parameter, find the best value
    for param_name in tuning_order:
        param_desc = param_name.replace("_", " ").title()
        param_values = param_configs[param_name]
        
        print(f"\n{'='*50}")
        print(f"Tuning {param_desc} | Values: {param_values}")
        print(f"{'='*50}")
        
        # Create parameter-specific output directory
        param_output_dir = f"{output_dir}/sequential"
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Results file for this parameter
        results_file = f"{param_output_dir}/{param_name}_results.csv"
        
        # Results for this parameter
        param_results = []
        
        # Check for existing results
        if os.path.exists(results_file):
            existing_df = pd.read_csv(results_file)
            param_results = existing_df.to_dict('records')
            
            # Skip already tested values
            tested_values = set(existing_df[param_name])
            param_values = [v for v in param_values if v not in tested_values]
            print(f"Found existing results, {len(param_values)} values left to test")
        
        # Test each value
        for value in param_values:
            # Set parameter value
            setattr(args, param_name, value)
            
            print(f"Testing {param_name}={value}")
            
            # Run experiment
            experiment_results = run_experiment(args)
            
            # Create result row with just this parameter
            result = {
                "dataset": args.dataset,
                "modality": "_".join(args.modality),
                param_name: value,
                "accuracy": experiment_results["accuracy"],
                "auc": experiment_results["auc"],
                "sensitivity": experiment_results["sensitivity"],
                "specificity": experiment_results["specificity"],
                "runtime_seconds": experiment_results["runtime_seconds"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add current best parameters
            for p in best_params:
                if p not in result:
                    result[p] = best_params.get(p)
            
            param_results.append(result)
            
            # Save after each run
            pd.DataFrame(param_results).to_csv(results_file, index=False)
            
            if experiment_results["success"]:
                print(f"  Results: acc={result['accuracy']:.4f}, auc={result['auc']:.4f}")
            else:
                print(f"  Error: {experiment_results['error']}")
        
        # Find best value for this parameter
        results_df = pd.DataFrame(param_results)
        
        if len(results_df) > 0:
            # Find best by accuracy
            best_idx = results_df["accuracy"].idxmax()
            best_value = results_df.loc[best_idx, param_name]
            best_acc = results_df.loc[best_idx, "accuracy"]
            best_auc = results_df.loc[best_idx, "auc"]
            
            print(f"Best {param_name} = {best_value} (acc={best_acc:.4f}, auc={best_auc:.4f})")
            
            # Update the parameter for future experiments
            setattr(args, param_name, best_value)
            
            # Update best params
            best_params[param_name] = best_value
            if best_acc > best_params.get("best_accuracy", 0):
                best_params["best_accuracy"] = best_acc
            if best_auc > best_params.get("best_auc", 0):
                best_params["best_auc"] = best_auc
            
            # Create plots
            plt.figure(figsize=(10, 6))
            plt.plot(results_df[param_name], results_df["accuracy"], marker='o', label='Accuracy')
            plt.plot(results_df[param_name], results_df["auc"], marker='s', label='AUC')
            plt.title(f"Effect of {param_desc} on {args.dataset} {args.modality}")
            plt.xlabel(param_desc)
            plt.ylabel("Score")
            plt.grid(True)
            plt.legend()
            
            # Save the plot
            plt.savefig(f"{param_output_dir}/{param_name}_performance.png")
            plt.close()
    
    # Save final best parameters
    pd.DataFrame([best_params]).to_csv(f"{output_dir}/best_parameters_sequential.csv", index=False)
    
    print("\nSequential parameter tuning complete!")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params

def run_parameter_tuning(args, strategy='sequential'):
    """Run parameter tuning using the specified strategy"""
    dataset_modalities, param_configs = create_parameter_configurations()
    
    # Define base output directory
    base_output_dir = "results/tuning_parameter"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # If specific dataset and modality provided, only test those
    if args.dataset and args.modality:
        modalities_to_test = {args.dataset: [[args.modality]]}
    else:
        modalities_to_test = dataset_modalities
    
    # Track all best parameters
    all_best_params = []
    
    # For each dataset and modality
    for dataset, modalities_list in modalities_to_test.items():
        for modality in modalities_list:
            print(f"\n\n{'='*80}")
            print(f"TUNING FOR DATASET: {dataset}, MODALITY: {modality}")
            print(f"{'='*80}\n")
            
            # Set up base arguments
            tuning_args = parse_arguments(description="ATK Parameter Tuning")
            tuning_args.stratified = True
            tuning_args.n_repeats = 1
            tuning_args.val_size = 0.2
            tuning_args.k_fold = 10  
            
            # Apply dataset and modality
            tuning_args.dataset = dataset
            tuning_args.modality = modality
            
            # Set defaults for other parameters
            tuning_args.lr = 0.0004
            tuning_args.patience = 30
            tuning_args.epochs = 1000
            tuning_args.num_layers = 4
            tuning_args.num_head = 3
            tuning_args.top_k = 2
            tuning_args.reg_coef = 0.1
            tuning_args.embed_dim = 128
            
            # Apply any overrides passed from command line
            for k, v in vars(args).items():
                if hasattr(tuning_args, k) and v is not None:
                    setattr(tuning_args, k, v)
            
            # Output directory for this dataset-modality combination
            output_dir = f"{base_output_dir}/{dataset}_{'_'.join(modality)}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Run the chosen tuning strategy
            if strategy == 'grid':
                # Select key parameters for grid search (would be too expensive to do all)
                grid_params = {
                    "num_layers": param_configs["num_layers"],
                    "num_head": param_configs["num_head"],
                    "lr": param_configs["lr"]
                }
                best_params = grid_search(tuning_args, grid_params, output_dir)
            else:  # sequential
                best_params = sequential_tuning(tuning_args, param_configs, output_dir)
            
            all_best_params.append(best_params)
    
    # Save all best parameters
    if all_best_params:
        best_df = pd.DataFrame(all_best_params)
        best_df.to_csv(f"{base_output_dir}/all_best_parameters_{strategy}.csv", index=False)
        print(f"\nAll best parameters saved to {base_output_dir}/all_best_parameters_{strategy}.csv")
    
    return all_best_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ATK Comprehensive Parameter Tuning")
    
    # Dataset and modality selection
    parser.add_argument("--dataset", type=str, help="Dataset to tune (default: tune all)")
    parser.add_argument("--modality", type=str, help="Modality to tune (default: tune all)")
    
    # Tuning strategy
    parser.add_argument("--strategy", type=str, default="sequential", 
                        choices=["sequential", "grid"],
                        help="Tuning strategy to use")
    
    # Experiment settings
    parser.add_argument("--k_fold", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run parameter tuning
    run_parameter_tuning(args, args.strategy)