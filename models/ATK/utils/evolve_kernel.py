import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_attention_maps(attention_dir, epoch):
    """Load attention maps for a specific epoch.
    
    Args:
        attention_dir: Directory containing attention data organized by epoch
        epoch: Epoch number to load
        
    Returns:
        Dictionary containing attention maps and related data
    """
    epoch_dir = os.path.join(attention_dir, f'epoch_{epoch}')
    
    # Get the number of blocks by counting block_*_avg_attention.npy files
    block_files = list(Path(epoch_dir).glob('block_*_avg_attention.npy'))
    num_blocks = len(block_files)
    
    # Load each block's attention map
    block_attentions = []
    for i in range(num_blocks):
        block_path = os.path.join(epoch_dir, f'block_{i}_avg_attention.npy')
        if os.path.exists(block_path):
            block_attentions.append(np.load(block_path))
    
    # Load average attention
    avg_path = os.path.join(epoch_dir, 'global_avg_attention.npy')
    avg_attention = np.load(avg_path) if os.path.exists(avg_path) else None
    
    # Load indices and labels
    train_idx = np.load(os.path.join(epoch_dir, 'train_indices.npy'))
    val_idx = np.load(os.path.join(epoch_dir, 'val_indices.npy'))
    labels = np.load(os.path.join(epoch_dir, 'labels.npy'))
    
    return {
        'block_attentions': block_attentions,
        'global_avg_attention': avg_attention,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'num_blocks': num_blocks,
        'labels': labels
    }


def plot_attention_heatmap(attention_matrix, title, save_path=None, cmap="viridis", labels=None):
    """
    Plot attention matrix as a heatmap, optionally sorting by labels.
    
    Args:
        attention_matrix: 2D array containing attention weights
        title: Title for the plot
        save_path: Path to save the figure (if None, will display)
        cmap: Colormap for visualization
        labels: Array of labels to sort the matrix by (if None, no sorting is applied)
    """
    plt.figure(figsize=(5, 4))
    
    # Sort matrix by labels if provided
    if labels is not None:
        # Get indices that would sort the labels
        sorted_indices = np.argsort(labels)
        
        # Reindex both dimensions of the attention matrix
        sorted_matrix = attention_matrix[sorted_indices, :][:, sorted_indices]
        
        # Plot the sorted matrix
        sns.heatmap(sorted_matrix, cmap=cmap, square=True)
        
        # Add a line to separate different classes
        if len(np.unique(labels)) > 1:
            # Find where label changes
            sorted_labels = labels[sorted_indices]
            change_points = np.where(np.diff(sorted_labels) != 0)[0]
            
            for point in change_points:
                plt.axhline(y=point + 0.5, color='red', linestyle='-', linewidth=0.2)
                plt.axvline(x=point + 0.5, color='red', linestyle='-', linewidth=0.2)
    else:
        # Plot without sorting
        sns.heatmap(attention_matrix, cmap=cmap, square=True)
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_epoch_attention_maps(attn_data, epoch, figsize=(10, 4), save_path=None):
    """
    Plot all attention maps for an epoch in a single row.
    
    Args:
        attn_data: Dictionary containing attention data for the epoch
        epoch: Epoch number for the title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot attention maps for each block
    for block_idx in range(min(3, len(attn_data['block_attentions']))):
        ax = axes[block_idx]
        attention_matrix = attn_data['block_attentions'][block_idx]
        
        # Sort matrix by labels if provided
        if 'labels' in attn_data:
            sorted_indices = np.argsort(attn_data['labels'])
            sorted_matrix = attention_matrix[sorted_indices, :][:, sorted_indices]
            sorted_labels = attn_data['labels'][sorted_indices]
            
            # Plot the sorted matrix
            sns.heatmap(sorted_matrix, cmap='viridis', square=True, ax=ax, xticklabels=False, yticklabels=False, cbar=False)
            
            # Add lines to separate different classes
            if len(np.unique(sorted_labels)) > 1:
                change_points = np.where(np.diff(sorted_labels) != 0)[0]
                for point in change_points:
                    ax.axhline(y=point + 0.5, color='red', linestyle='-', linewidth=0.2)
                    ax.axvline(x=point + 0.5, color='red', linestyle='-', linewidth=0.2)
        else:
            sns.heatmap(attention_matrix, cmap='viridis', square=True, ax=ax, xticklabels=False, yticklabels=False, cbar=False)
            
        ax.set_title(f'Block {block_idx}')
    
    # Add overall title
    fig.suptitle(f'Adaptive Kernels at Epoch {epoch}', y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def visualize_sorted_kernel(kernel_matrix, labels, version, modality, normalize_kernel_fn=None):
    """
    Normalize, sort and visualize kernel matrix by labels.
    
    Args:
        kernel_matrix: Input kernel matrix (n x n)
        labels: Array of labels corresponding to each row of kernel
        version: Version directory to save image
        modality: Modality name for image filename
        normalize_kernel_fn: Function to normalize kernel, takes np.ndarray as input
    """
    # Step 1: Normalize
    if normalize_kernel_fn is None:
        normalized_kernel = kernel_matrix
    else:
        normalized_kernel = normalize_kernel_fn(kernel_matrix)
    
    # Step 2: Sort by labels
    sorted_indices = np.argsort(labels)
    print(f"Total number of positive samples: {np.sum(labels)}")
    
    sorted_kernel = normalized_kernel[sorted_indices, :][:, sorted_indices]
    
    # Step 3: Plot and save image
    plt.figure(figsize=(5, 4))
    plt.imshow(sorted_kernel, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Sorted Kernel Matrix - {modality}')
    plt.show()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    parser.add_argument('--attention_dir', type=str, 
                        default='/Users/macbook/Documents/WorkSpace/DeepMulti-KernelLearning/experiments/scripts/checkpoints/attention_maps_PET',
                        help='Directory containing attention map folders')
    parser.add_argument('--epochs', type=str, default='5,10,15,20,25,30,35',
                        help='Epochs to visualize, comma-separated')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--show', action='store_true',
                        help='Display plots instead of saving')
    
    args = parser.parse_args()
    
    # Create output directory if specified and not showing plots
    if args.output_dir and not args.show:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse epochs to process
    epochs_to_process = [int(e.strip()) for e in args.epochs.split(',')]
    
    # Process each epoch
    for epoch in sorted(epochs_to_process):
        print(f"Processing attention maps for epoch {epoch}...")
        
        # Load attention data
        try:
            attn_data = load_attention_maps(args.attention_dir, epoch)
            
            # Define save path if not showing plots
            save_path = None
            if args.output_dir and not args.show:
                save_path = os.path.join(args.output_dir, f'epoch_{epoch}_attention.png')
            
            # Plot attention maps
            plot_epoch_attention_maps(attn_data, epoch, save_path=save_path)
            
        except FileNotFoundError:
            print(f"Warning: Data for epoch {epoch} not found!")
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()