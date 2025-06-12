import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
import os
import pickle
import json

def plot_mean_roc_curve(y_true_folds, y_pred_folds, title='Mean ROC Curve (k-fold CV)', 
                        show_individual_curves=True, figsize=(10, 8), save_path=None):
    """
    Plot mean ROC curve with confidence interval across k-fold cross-validation.
    
    Parameters:
    -----------
    y_true_folds : list
        List containing true labels for each fold
    y_pred_folds : list
        List containing predicted probabilities for each fold
    title : str
        Title for the plot
    show_individual_curves : bool
        Whether to plot individual ROC curves for each fold
    figsize : tuple
        Figure size (width, height)
    save_path : str or None
        Path to save the figure, if None, figure is not saved
        
    Returns:
    --------
    fig : matplotlib figure
        Figure object containing the ROC curve plot
    mean_auc : float
        Mean AUC across all folds
    std_auc : float
        Standard deviation of AUC across all folds
    """
    # Check input lengths
    assert len(y_true_folds) == len(y_pred_folds), "Number of true label and prediction arrays must match"
    n_folds = len(y_true_folds)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Storage for interpolated curve data
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Plot ROC curve for each fold
    for i in range(n_folds):
        fpr, tpr, _ = roc_curve(y_true_folds[i], y_pred_folds[i])
        
        # Use interp1d instead of interp
        # First, handle edge cases to ensure interpolation works properly
        if fpr[0] != 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
        if fpr[-1] != 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)
            
        # Create interpolation function and apply to mean_fpr grid
        interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
        tpr_interp = interp_func(mean_fpr)
        
        tprs.append(tpr_interp)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        if show_individual_curves:
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                    label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')
    
    # Plot random chance line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    
    # Calculate and plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    # Đảm bảo đường ROC trung bình bắt đầu từ (0, 0)
    mean_tpr[0] = 0.0  # Đảm bảo điểm đầu tiên bắt đầu từ TPR = 0
    mean_tpr[-1] = 1.0  # Force end at (1,1)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
            lw=2, alpha=.8)
    
    # Calculate and plot standard deviation
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'±1 std. dev.')
    
    # Set plot details
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, mean_auc, std_auc


# Add these imports


def save_roc_data(y_true_folds, y_pred_folds, model_name, dataset_name, save_dir='results/roc_data'):
    """
    Save ROC data for later comparison.
    
    Parameters:
    -----------
    y_true_folds : list
        List containing true labels for each fold
    y_pred_folds : list
        List containing predicted probabilities for each fold
    model_name : str
        Name of the model/method
    dataset_name : str
        Name of the dataset
    save_dir : str
        Directory to save the results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate AUC for reporting
    aucs = []
    for i in range(len(y_true_folds)):
        fpr, tpr, _ = roc_curve(y_true_folds[i], y_pred_folds[i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    # Save the raw data as pickle
    data = {
        'y_true_folds': y_true_folds,
        'y_pred_folds': y_pred_folds,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'mean_auc': mean_auc,
        'std_auc': std_auc
    }
    
    file_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_roc_data.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Also save a summary as JSON for easy reference
    summary = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'num_folds': len(aucs)
    }
    
    json_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return file_path

def compare_roc_curves(model_files, title="Comparison of ROC Curves", figsize=(12, 10), 
                      save_path=None, colors=None, show_ci=True):
    """
    Compare ROC curves from multiple models.
    
    Parameters:
    -----------
    model_files : list
        List of file paths containing ROC data (from save_roc_data function)
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    save_path : str or None
        Path to save the figure
    colors : list or None
        List of colors for each model (if None, auto-generate)
    show_ci : bool
        Whether to show confidence intervals
        
    Returns:
    --------
    fig : matplotlib figure
        Figure object containing the comparison plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_files)))
    
    # Load and plot each model
    for i, file_path in enumerate(model_files):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        y_true_folds = data['y_true_folds']
        y_pred_folds = data['y_pred_folds']
        model_name = data['model_name']
        mean_auc = data['mean_auc']
        std_auc = data['std_auc']
        
        # Calculate mean ROC curve
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for j in range(len(y_true_folds)):
            fpr, tpr, _ = roc_curve(y_true_folds[j], y_pred_folds[j])
            
            # Handle edge cases
            if fpr[0] != 0:
                fpr = np.insert(fpr, 0, 0)
                tpr = np.insert(tpr, 0, 0)
            if fpr[-1] != 1:
                fpr = np.append(fpr, 1)
                tpr = np.append(tpr, 1)
            
            interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
            tpr_interp = interp_func(mean_fpr)
            tprs.append(tpr_interp)
        
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        
        # Plot mean ROC
        color = colors[i] if isinstance(colors, list) else colors
        ax.plot(mean_fpr, mean_tpr, lw=2, color=color,
                label=f'{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        
        # Plot confidence interval
        if show_ci:
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.1)
    
    # Plot random chance line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=.8)
    
    # Set plot details
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig