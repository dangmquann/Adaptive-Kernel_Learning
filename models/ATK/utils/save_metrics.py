import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

def save_metrics_and_args(
    accuracies, sensitivities, specificities, aucs,
    args, chosen_modality, k_fold, save_dir,
    classification_reports=None
):
    """
    Compute average metrics, print results, and save metrics and arguments to CSV files.

    Args:
        accuracies (list or np.ndarray): List of accuracy values across folds.
        sensitivities (list or np.ndarray): List of sensitivity values.
        specificities (list or np.ndarray): List of specificity values.
        aucs (list or np.ndarray): List of AUC values.
        args (argparse.Namespace): Parsed arguments object.
        chosen_modality (str): Name of the modality (e.g., MRI, PET).
        k_fold (int): Number of folds in cross-validation.
        save_dir (str): Directory to save result CSV files.
        classification_reports (list, optional): List of classification report DataFrames.
    """

    print("="*25)
    # === Compute averages and standard deviations ===
    average_accuracy = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    average_sensitivity = np.mean(sensitivities)
    sensitivity_std = np.std(sensitivities)
    average_specificity = np.mean(specificities)
    specificity_std = np.std(specificities)
    average_auc = np.mean(aucs)
    auc_std = np.std(aucs)

    # === Print out the results ===
    print(f"Modality: {chosen_modality}")
    print(f"Average AUC: {average_auc * 100:.2f} ± {auc_std * (100 / k_fold):.2f}")
    # print(f"AUC Std: {auc_std * (100 / k_fold):.2f}%")
    print(f"Average Accuracy: {average_accuracy * 100:.2f} ± {accuracy_std * (100 / k_fold):.2f}")
    # print(f"Accuracy Std: {accuracy_std * (100 / k_fold):.2f}%")
    print(f"Average Sensitivity: {average_sensitivity * 100:.2f} ± {sensitivity_std * (100 / k_fold):.2f}")
    # print(f"Sensitivity Std: {sensitivity_std * (100 / k_fold):.2f}%")
    print(f"Average Specificity: {average_specificity * 100:.2f} ± {specificity_std * (100 / k_fold):.2f}")
    # print(f"Specificity Std: {specificity_std * (100 / k_fold):.2f}%")

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # === Save metrics to a CSV file ===
    metrics_df = pd.DataFrame({
        'Modality': [chosen_modality],
        'Average Accuracy': [round(average_accuracy * 100, 2)],
        'Accuracy Std': [round(accuracy_std * (100 / k_fold), 2)],
        'Average Sensitivity': [round(average_sensitivity * 100, 2)],
        'Sensitivity Std': [round(sensitivity_std * (100 / k_fold), 2)],
        'Average Specificity': [round(average_specificity * 100, 2)],
        'Specificity Std': [round(specificity_std * (100 / k_fold), 2)],
        'Average AUC': [round(average_auc * 100, 2)],
        'AUC Std': [round(auc_std * (100 / k_fold), 2)]
    })

    args_df = pd.DataFrame({
        'Argument': list(vars(args).keys()),
        'Value': list(vars(args).values())
    })

    # Save aggregated metrics and arguments
    metrics_csv_path = os.path.join(save_dir, f'accuracy_metrics_{chosen_modality}.csv')
    args_csv_path = os.path.join(save_dir, f'args_{chosen_modality}.csv')

    metrics_df.to_csv(metrics_csv_path, index=False)
    args_df.to_csv(args_csv_path, index=False)

    # Save classification reports if provided
    if classification_reports and len(classification_reports) > 0:
        # Combine all classification reports into a single DataFrame
        combined_report = pd.concat(
            classification_reports, 
            keys=[f'Fold_{i}' for i in range(len(classification_reports))]
        )
        
        # Save the combined classification report
        report_csv_path = os.path.join(save_dir, f'classification_reports_{chosen_modality}.csv')
        combined_report.to_csv(report_csv_path)
        print(f"Classification reports saved to {report_csv_path}")

    print(f"\nMetrics saved to {metrics_csv_path}")
    print(f"Arguments saved to {args_csv_path}")


def plot_roc_curves(fprs=None, tprs=None, aucs=None, metrics=None, dataset=None, modality=None, modalities=None, version=None, args=None):
    """
    Plot ROC curves and save mean ROC data for later comparison.
    
    Args:
        fprs: List of false positive rates for each fold
        tprs: List of true positive rates for each fold
        aucs: List of AUC values for each fold
        metrics: Dictionary containing metrics including 'fprs', 'tprs' and 'aucs'
        dataset: Name of the dataset
        modality: Name of the modality or list of modalities (legacy parameter)
        modalities: List of modalities used (preferred parameter)
        version: Version identifier
        args: Arguments from argparse for method information
    """
    # Handle input format compatibility
    if metrics is not None:
        fprs = metrics['fprs']
        tprs = metrics['tprs']
        aucs = metrics['aucs']
    
    # Handle modality parameter naming compatibility
    if modalities is None:
        modalities = modality
    
    plt.figure(figsize=(10, 8))
    
    # Define colors for consistent fold visualization
    colors = plt.cm.tab10(np.linspace(0, 1, len(fprs)))
    
    # Define mean FPR for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    
    # Store interpolated TPRs
    interpolated_tprs = []
    
    # Plot individual ROC curves for each fold
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        fold_auc = aucs[i]
        plt.plot(fpr, tpr, lw=1, alpha=0.6, color=colors[i],
                label=f'Fold {i} (AUC = {fold_auc:.3f})')
        
        # Interpolate TPR at standard FPR points for averaging
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', 
             alpha=0.8, label='Chance')
    
    # Calculate mean ROC and statistics
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    std_tpr = np.std(interpolated_tprs, axis=0)
    
    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, alpha=0.9,
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    # Add standard deviation band
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                    label=f'± 1 std. dev.')
    
    # Configure plot appearance
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    
    # Create title with dataset and modality
    mod_str = '_'.join(modalities) if isinstance(modalities, list) else modalities
    plt.title(f'ROC Curves - {dataset} - {mod_str}', fontsize=16)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save mean ROC data
    if args is not None:
        method_str = f"{args.kernel_level}_{args.kernel_type}" if hasattr(args, 'kernel_level') and hasattr(args, 'kernel_type') else "ATK"
        
        roc_data = {
            'dataset': dataset,
            'modalities': modalities,
            'method': method_str,
            'version': version,
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'mean_auc': mean_auc,
            'std_auc': std_auc
        }
        
        # Create directory for ROC data
        roc_data_dir = f'../../experiments/results/{dataset}/{version}/roc_data'
        os.makedirs(roc_data_dir, exist_ok=True)
        
        # Save ROC data
        filename = f'mean_roc_{dataset}_{mod_str}_{method_str}.pkl'
        with open(os.path.join(roc_data_dir, filename), 'wb') as f:
            pickle.dump(roc_data, f)
        print(f"Mean ROC data saved to: {os.path.join(roc_data_dir, filename)}")
    
    # Create directory for saving plots
    plot_dir = f'../../experiments/results/{dataset}/{version}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(f'{plot_dir}/roc_curves_{dataset}_{mod_str}.png', dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {plot_dir}/roc_curves_{dataset}_{mod_str}.png")
    
    # plt.close()