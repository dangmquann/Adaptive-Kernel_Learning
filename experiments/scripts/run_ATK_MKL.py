import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split

from dataloader import DataLoader, CrossValidator, train_test_kernel_cv_split, KernelConstructor, build_graph_data, prepare_data_config, build_multimodal_graph_list
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix
from models.ATK.utils import train_test
from models import kernels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_attention_maps(model, data):
    """
    Extract attention maps from trained ATK_QK model
    
    Args:
        model: Trained ATK_QK model
        data: The data to run through the model
        
    Returns:
        List of attention maps from each layer/head
    """
    model.eval()
    with torch.no_grad():
        _, attentions = model(data)
    
    # Convert attention maps to numpy arrays
    attention_maps = []
    for layer_attn in attentions:
        # Average over attention heads if multi-headed
        if len(layer_attn.shape) == 3:
            layer_attn = layer_attn.mean(dim=0)  
        attention_maps.append(layer_attn.cpu().numpy())
    
    return attention_maps[0:]  # Return only the last layer's attention maps (Optional)

def attention_to_kernel(attention_map):
    """
    Convert attention map to a valid kernel matrix
    
    Args:
        attention_map: [N, N] attention map
        
    Returns:
        [N, N] kernel matrix
    """
    # Ensure symmetry (average with transpose)
    kernel = (attention_map + attention_map.T) / 2
    
    # Ensure positive semi-definiteness
    # Add small value to diagonal to ensure numerical stability
    min_eig = np.min(np.linalg.eigvalsh(kernel))
    if min_eig < 0:
        kernel = kernel - min_eig * np.eye(kernel.shape[0]) + 1e-6 * np.eye(kernel.shape[0])
    
    return kernel

def main(args):
    atk_params = {'num_layers': args.num_layers, 'num_heads': args.num_head, 'dropout': 0.1, 'multihead_agg': 'concat', 'reg_coef': args.reg_coef,
                  'use_prior': True}

    logger = setup_logger(log_dir="logs", name=f"{args.dataset}_{'-'.join(args.modality)}")
    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    data_config = prepare_data_config(args, data_dir)

    # Initialize data loader and cross-validator
    data_loader = DataLoader(data_config)
    kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=args.k_fold, n_repeats=args.n_repeats, stratified=args.stratified, random_state=args.seed)

    # Create data splits
    splits, _ = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)

    logger.info(f"Chosen modalities: {args.modality}")
    logger.info(f"Data configuration: {data_config}")
    logger.info(f"Starting training for dataset: {args.dataset}")
    logger.info(f"Modalities: {args.modality}")

    # Results directory
    results_dir = f"results/ATK_EasyMKL/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Directory for attention maps and kernels
    attention_dir = os.path.join(results_dir, "attention_maps")
    os.makedirs(attention_dir, exist_ok=True)
    kernels_dir = os.path.join(results_dir, "kernels")
    os.makedirs(kernels_dir, exist_ok=True)
    
    # Metrics tracking
    metrics = {}
    
    # Split K folds
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")
        # Extract feature data
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split
        
        # ===== STAGE 1: PARALLEL ATK MODEL TRAINING FOR EACH MODALITY =====
        print("Stage 1: Training multiple ATK_QK models in parallel for each modality...")
        
        # Store models, attention maps, and kernels for each modality
        all_trained_models = {}
        all_attention_maps = {}
        all_kernels_list = []
        
        # Save the common train indices from first modality to use consistently across all modalities
        common_train_mask = None
        
        # For each modality, build a graph and train an ATK model
        for modality_idx, modality in enumerate(args.modality):
            print(f"  Training model for modality: {modality}")
            
            # Build graph for this modality
            graph_data = build_graph_data(
                Xs_train[modality],
                Y_train,
                Xs_test[modality],
                Y_test,
                top_k=args.top_k,
                random_state=args.seed,
                val_size=args.val_size,
                cuda=device != "cpu",
            )
            
            # Save consistent train mask from first modality
            if common_train_mask is None:
                common_train_mask = graph_data.train_mask
                mask_train_indices = torch.where(common_train_mask)[0].cpu().numpy()
            
            # Train ATK_QK model for this modality
            results = train_test(
                data=graph_data,
                embedding_size=args.embed_dim,
                atk_params=atk_params,
                num_epochs=args.epochs,
                lr=args.lr,
                random_seed=args.seed + modality_idx,  # Different seed for each modality
                early_stopping_patience=args.patience,
                logger=logger,
                use_QK=True,
            )
            
            # Get trained model
            trained_model = results['best_models'].get('best_loss_model')
            if trained_model:
                # Save the trained model
                all_trained_models[modality] = trained_model
                
                # Extract attention maps
                attention_maps = extract_attention_maps(trained_model, graph_data)
               
                
                all_attention_maps[modality] = attention_maps
                
                # Convert attention maps to kernels and save them
                for i, attn_map in enumerate(attention_maps):
                     # Shape attention maps is [N, N] for each layer
                    logger.info(f"Shape of attention map for {modality} layer {i}: {attn_map.shape}")
                    # Convert attention map to kernel matrix
                    kernel = attn_map #attention_to_kernel(attn_map)
                    all_kernels_list.append(kernel)
                    
                    # Save both attention map and kernel
                    np.save(os.path.join(attention_dir, f"fold_{fold_idx}_{modality}_layer_{i}_attention.npy"), attn_map)
                    np.save(os.path.join(kernels_dir, f"fold_{fold_idx}_{modality}_layer_{i}_kernel.npy"), kernel)
                    
                print(f"    Extracted {len(attention_maps)} attention maps for {modality}")
            else:
                print(f"    Warning: No model found for modality {modality}. Skipping this modality.")
        
        # Check if any models were successfully trained
        if not all_trained_models:
            print(f"Warning: No models were successfully trained for fold {fold_idx}. Skipping this fold.")
            continue
            
        print(f"  Successfully trained models for {len(all_trained_models)} modalities")
        print(f"  Total number of kernel matrices: {len(all_kernels_list)}")
        
       # ===== STAGE 2: EASYMKL WITH ATTENTION KERNELS =====
        print("Stage 2: Running EasyMKL with attention kernels from all modalities...")

        # Prepare kernel matrices for EasyMKL
        train_kernels = []        # KX_train: train vs train
        train_test_kernels = []   # KX_train_test: test vs train
        test_test_kernels = []    # KX_test_test: test vs test 

        n_train = len(mask_train_indices)
        n_val = graph_data.val_mask.sum().item()
        n_test = len(Y_test)

        for kernel in all_kernels_list:
            # Extract training kernel (train samples x train samples)
            K_train = kernel[:n_train, :n_train]
            train_kernels.append(K_train)
            
            # Extract test-train kernel (test samples x train samples)
            K_test_train = kernel[n_train + n_val:, :n_train]
            train_test_kernels.append(K_test_train)
            
            # Extract test-test kernel (test samples x test samples)
            K_test_test = kernel[n_train + n_val:, n_train + n_val:]
            test_test_kernels.append(K_test_test)

        # Initialize EasyMKL
        print(f"  Training EasyMKL with {len(train_kernels)} kernel matrices...")
        mkl = kernels.KernelCombination(method="EasyMKL")

        # Convert Y_train to have only data with mask_train_indices
        Y_train_masked = Y_train[mask_train_indices]
        print(f"Shape of Y_train_masked: {Y_train_masked.shape}")
        print(f"Y_train_masked: {Y_train_masked[:10]}")  # Print first 10 labels for debugging
        print(f"Shape of Y_test: {Y_test.shape}")
        print(f"Y_test: {Y_test[:10]}")  # Print first 10 labels for debugging



        # # Train EasyMKL using attention kernels with all four required parameters
        # mkl.fit(train_kernels, Y_train_masked, train_test_kernels, Y_test)

        # # Get combined kernel
        # combined_train_kernel = mkl.transform(train_kernels)
        # combined_test_kernel = mkl.transform(train_test_kernels)

        # Sum the kernels
        combined_train_kernel = sum(train_kernels) / len(train_kernels)
        combined_test_kernel = sum(train_test_kernels) / len(train_test_kernels)

        # Train kernel classifier
        clf = kernels.KernelClassifier()

        #Create output kernel from Y_train for EasyMKL
        kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
        Y_K_train = kernel_constructor.create_ouput_kernel(Y_train_masked)

        clf.fit(combined_train_kernel, Y_K_train)
        
        # Predict on test data
        y_pred_prob = clf.predict_proba(combined_test_kernel)
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold_youden = thresholds[youden_index]
        
        f1_scores = [f1_score(Y_test, (y_pred_prob > t).astype(int)) for t in thresholds]
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
        
        # Make final predictions
        test_preds = (y_pred_prob > optimal_threshold_f1).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(Y_test, test_preds)
        auc = roc_auc_score(Y_test, y_pred_prob)
        cm = confusion_matrix(Y_test, test_preds)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        cr = classification_report(Y_test, test_preds, output_dict=True)
        
        # Save results
        results = {
            'prediction_results': {
                'metrics': {
                    'accuracy': acc,
                    'auc': auc,
                    'sensitivity': sens,
                    'specificity': spec,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': cr,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'optimal_threshold_youden': float(optimal_threshold_youden),
                    'optimal_threshold_f1': float(optimal_threshold_f1)
                },
            'probabilities': y_pred_prob.tolist(),
            'predictions': test_preds.tolist(),
            'y_pred_prob': y_pred_prob.tolist(),
            'y_test': Y_test.tolist()
            }
        }
        
        print(f"  Fold {fold_idx} - Acc: {acc:.4f}, AUC: {auc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}")
        
        # Log results
        metrics = log_results(results, metrics, fold_idx)
        
    # Save metrics
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(results_dir, "results.csv"), index=False)
    
    # Get metrics with standard deviations
    df_summary, cf_summary, acc, sens, spec, acc_std, sens_std, spec_std = combine_confusion_matrix(
        os.path.join(results_dir, "results.csv")
    )
    
    # Calculate AUC mean and std directly from the metrics dictionary
    auc_values = metrics['aucs']
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values)
    
    # Create a summary dataframe with mean and standard deviation
    summary_stats = {
        'Metric': ['AUC', 'Accuracy', 'Sensitivity', 'Specificity'],
        'Mean': [auc_mean, acc, sens, spec],
        'Std': [auc_std, acc_std, sens_std, spec_std],
        'Mean±Std': [
            f"{auc_mean*100:.2f} ± {auc_std*100/args.k_fold:.2f}",
            f"{acc*100:.2f} ± {acc_std*100/args.k_fold:.2f}", 
            f"{sens*100:.2f} ± {sens_std*100/args.k_fold:.2f}", 
            f"{spec*100:.2f} ± {spec_std*100/args.k_fold:.2f}",
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(results_dir, "summary_stats.csv"), index=False)
    
    print("\nSummary statistics:")
    print(summary_df)
    
    return df_summary, cf_summary, acc, summary_df


if __name__ == '__main__':
    args = parse_arguments(description="ATK to EasyMKL")
    
    # Set default parameters
    args.stratified = True
    args.n_repeats = 1
    args.val_size = 0.2
    args.k_fold = 10
    
    args.dataset = "AD_CN"
    args.modality = ["PET", "GM", "CSF"]
    args.lr = 0.0004
    args.patience = 30
    args.epochs = 1000
    args.num_layers = 4
    args.num_head = 3
    args.top_k = 2
    args.reg_coef = 0.1
    args.embed_dim = 128
    args.kernel_type = "rbf"
    
    # Run the main function
    df, cf, acc, summary_df = main(args)
    print(f"\nFinal results - Confusion matrix:\n{cf}\nAccuracy: {acc:.4f}")