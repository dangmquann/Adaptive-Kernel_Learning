import os
import pandas as pd
import glob
from pathlib import Path

# Define the project root directory
# 1. Using relative path from the current file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 2. Or by detecting git repository root (more robust)
def find_git_root(path='.'):
    if os.path.exists(os.path.join(path, '.git')):
        return path
    parent = os.path.dirname(os.path.abspath(path))
    if parent == path:  # Reached root directory
        return None
    return find_git_root(parent)


PROJECT_ROOT = find_git_root() or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def extract_all_method_results():
    """
    Extract mean and std metrics for all methods across different datasets
    and organize them in a comparative table.
    """
    # Base directory for results (using relative path)
    base_dir = os.path.join(PROJECT_ROOT, "experiments", "scripts", "results")
    
    # Output directory (using relative path)
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "scripts", "result_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Dataset prefixes that should be kept together
    dataset_prefixes = {"AD_CN", "ROSMAP"}
    
    # Scan all method directories
    for method_dir in os.listdir(base_dir):
        method_path = os.path.join(base_dir, method_dir)
        
        # Skip if not a directory
        if not os.path.isdir(method_path):
            continue
            
        print(f"Processing method: {method_dir}")
        
        # Scan all dataset directories within this method
        for dataset_dir in os.listdir(method_path):
            dataset_path = os.path.join(method_path, dataset_dir)
            
            # Skip if not a directory
            if not os.path.isdir(dataset_path):
                continue
                
            # Check if summary_stats.csv exists
            summary_file = os.path.join(dataset_path, "summary_stats.csv")
            if not os.path.exists(summary_file):
                print(f"  No summary_stats.csv found for {method_dir}/{dataset_dir}")
                continue
                
            # Parse dataset name and modalities correctly
            parts = dataset_dir.split('_')
            
            # Properly handle dataset names with underscores
            dataset_name = None
            for prefix in dataset_prefixes:
                if dataset_dir.startswith(prefix):
                    dataset_name = prefix
                    modalities_start = len(prefix) + 1  # +1 for the underscore
                    modalities = dataset_dir[modalities_start:] if modalities_start < len(dataset_dir) else "All"
                    break
            
            # Default parsing if no known prefix found
            if dataset_name is None:
                dataset_name = parts[0]
                modalities = '_'.join(parts[1:]) if len(parts) > 1 else "N/A"
            
            # Read summary stats
            try:
                summary_df = pd.read_csv(summary_file)
                
                # Extract metrics
                metrics_dict = {
                    'Method': method_dir,
                    'Dataset': dataset_name,
                    'Modalities': modalities,
                    'Path': dataset_path
                }
                
                # Add each metric
                for _, row in summary_df.iterrows():
                    metric_name = row['Metric']
                    mean_value = row['Mean'] if 'Mean' in row else None
                    std_value = row['Std'] if 'Std' in row else None
                    formatted_value = row.get('Mean±Std', f"{float(mean_value)*100:.2f} ± {float(std_value)*100:.2f}")
                    
                    metrics_dict[f"{metric_name}_Mean"] = mean_value
                    metrics_dict[f"{metric_name}_Std"] = std_value
                    metrics_dict[f"{metric_name}_Formatted"] = formatted_value
                
                all_results.append(metrics_dict)
                print(f"  Extracted metrics from {method_dir}/{dataset_dir}")
                
            except Exception as e:
                print(f"  Error reading {summary_file}: {e}")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(all_results)
    
    # Create pivot tables for each metric
    metrics = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity']
    
    for metric in metrics:
        if f"{metric}_Formatted" in results_df.columns:
            pivot_df = pd.pivot_table(
                results_df,
                values=f"{metric}_Formatted",
                index=['Dataset', 'Modalities'],
                columns=['Method'],
                aggfunc=lambda x: ' '.join(x)
            )
            
            pivot_path = os.path.join(output_dir, f"comparison_{metric}.csv")
            pivot_df.to_csv(pivot_path)
            print(f"{metric} comparison table saved to {pivot_path}")
    
        # Create methods summary with separation between single and multi-modality methods
    single_modality_methods = []
    multi_modality_methods = []  # For All, concat_ADNI, concat_ROSMAP

    for dataset in results_df['Dataset'].unique():
        dataset_df = results_df[results_df['Dataset'] == dataset]
        
        for modality in dataset_df['Modalities'].unique():
            mod_df = dataset_df[dataset_df['Modalities'] == modality]
            
            for _, row in mod_df.iterrows():
                result_entry = {
                    'Dataset': dataset,
                    'Modalities': modality,
                    'Method': row['Method'],
                    'Accuracy': row.get('Accuracy_Formatted', 'N/A'),
                    'AUC': row.get('AUC_Formatted', 'N/A')
                }
                
                # Check if modality is one of the multi-modality ones
                if modality in ['All', 'concat_ADNI', 'concat_ROSMAP']:
                    multi_modality_methods.append(result_entry)
                else:
                    single_modality_methods.append(result_entry)

    # Create and save single modality dataframe
    single_df = pd.DataFrame(single_modality_methods)
    single_path = os.path.join(output_dir, "Summary_single_table2.csv")
    single_df.to_csv(single_path, index=False)
    print(f"Single modality methods summary saved to {single_path}")

    # Create and save multi-modality dataframe
    multi_df = pd.DataFrame(multi_modality_methods)
    multi_path = os.path.join(output_dir, "Summary_multi_table3.csv")
    multi_df.to_csv(multi_path, index=False)
    print(f"Multi-modality methods summary saved to {multi_path}")

    # combined_df = pd.concat([single_df, multi_df])
    # combined_path = os.path.join(output_dir, "Summary.csv")
    # combined_df.to_csv(combined_path, index=False)
    # print(f"Complete summary saved to {combined_path}")

    return results_df, single_df, multi_df

if __name__ == "__main__":
    print("Extracting results from all methods...")
    results_df, single_df, multi_df = extract_all_method_results()
    
    # Display summary of results
    print("\nResults Summary:")
    print(f"Found {len(results_df)} result entries across {results_df['Method'].nunique()} methods")
    print(f"Datasets analyzed: {', '.join(sorted(results_df['Dataset'].unique()))}")
    
    # Show Single-modality methods table
    print("\nSingle-modality methods summary:")
    print(single_df.head())
    # Show Multi-modality methods table
    print("\nMulti-modality methods summary:")
    print(multi_df.head())