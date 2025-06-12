import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from dataloader import DataLoader, CrossValidator, train_test_kernel_cv_split, KernelConstructor, build_graph_data, prepare_data_config
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
        _, attentions = model(data['train_data'])
    
    # Convert attention maps to numpy arrays
    attention_maps = []
    for layer_attn in attentions:
        # Average over attention heads if multi-headed
        if len(layer_attn.shape) == 3:
            layer_attn = layer_attn.mean(dim=0)  
        attention_maps.append(layer_attn.cpu().numpy())
    
    return attention_maps

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

    logger = setup_logger(log_dir="logs", name=f"{args.modality[0]}")
    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    data_config = prepare_data_config(args, data_dir)

    # Initialize data loader and cross-validator
    data_loader = DataLoader(data_config)
    kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=args.k_fold, n_repeats=args.n_repeats, stratified=args.stratified, random_state=args.seed)

    # Create data splits
    splits, _ = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)

    logger.info(f"Starting training for dataset: {args.dataset}")
    logger.info(f"Modalities: {args.modality}")

    # Results directory for ATK
    atk_dir = f"results/ATKQK/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(atk_dir, exist_ok=True)
    
    # Results directory for ATK-SVM comparison
    results_dir = f"results/ATK_SVM_Comparison/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Directories for attention maps and kernels
    attention_dir = os.path.join(results_dir, "attention_maps")
    os.makedirs(attention_dir, exist_ok=True)
    kernels_dir = os.path.join(results_dir, "kernels")
    os.makedirs(kernels_dir, exist_ok=True)

    # Metrics dictionaries for different methods
    metrics_atk = {}
    metrics_adaptive_kernel = {}
    metrics_prior_kernel = {}

    # Split K folds
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")

        [Xs_train, Y_train, Xs_test, Y_test] = feature_split
        
        # Get feature matrices for the selected modality
        X_train = Xs_train[args.modality[0]]
        X_test = Xs_test[args.modality[0]]

        # Build graph data for ATK
        data = build_graph_data(
            X_train,
            Y_train,
            X_test,
            Y_test,
            top_k=args.top_k,
            random_state=args.seed,
            val_size=args.val_size,
            cuda=device != "cpu",
        )

        # Train ATK model
        results = train_test(
            data=data,
            embedding_size=args.embed_dim,
            atk_params=atk_params,
            num_epochs=args.epochs,
            lr=args.lr,
            random_seed=args.seed,
            early_stopping_patience=args.patience,
            logger=logger,
            use_QK=True,
        )

        # Log ATK results
        metrics_atk = log_results(results, metrics_atk, fold_idx)
        
        # Get the trained model from results
        trained_model = results['best_models'].get('best_loss_model', None)
        
        if trained_model is not None:
            # Extract attention maps
            attention_maps = extract_attention_maps(trained_model, data)
            
            for i, attn in enumerate(attention_maps):
                attn_file = os.path.join(attention_dir, f"fold_{fold_idx}_layer_{i}.npy")
                np.save(attn_file, attn)
                print(f"Saved attention map for fold {fold_idx}, layer {i} to {attn_file}")

            # Convert attention maps to kernel matrices
            kernel_matrices = [attention_to_kernel(attn) for attn in attention_maps]
            
            for i, kernel in enumerate(kernel_matrices):
                kernel_file = os.path.join(kernels_dir, f"fold_{fold_idx}_layer_{i}.npy")
                np.save(kernel_file, kernel)
                print(f"Saved kernel matrix for fold {fold_idx}, layer {i} to {kernel_file}")
            
            # Use the last layer's attention map as the adaptive kernel
            adaptive_kernel = kernel_matrices[-1]
            
            # ===== Classification using Adaptive Kernel =====
            print("Classification using Adaptive Kernel...")
            
            # Get indices for training and testing sets
            train_indices = np.where(data.train_mask.cpu().numpy())[0]
            test_indices = np.where(data.test_mask.cpu().numpy())[0]
            
            # Extract subsets of the kernel for training and testing
            K_train = adaptive_kernel[np.ix_(train_indices, train_indices)]
            K_test = adaptive_kernel[np.ix_(test_indices, train_indices)]
            
            # Train and test with kernels using SVC with precomputed kernel
            classifier = kernels.KernelClassifier()  
            #Create output kernel from Y_train for EasyMKL
            kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
            Y_K_train = kernel_constructor.create_ouput_kernel(Y_train[train_indices])
            classifier.fit(K_train, Y_K_train)
            
            # Predict with adaptive kernel
            y_pred_adaptive = classifier.predict(K_test)
            y_pred_prob_adaptive = classifier.predict_proba(K_test)[:, 1] if len(np.unique(Y_test)) == 2 else None
            
            # Calculate metrics
            acc_adaptive = accuracy_score(Y_test, y_pred_adaptive)
            if y_pred_prob_adaptive is not None:
                auc_adaptive = roc_auc_score(Y_test, y_pred_prob_adaptive)
            else:
                auc_adaptive = None
                
            cm_adaptive = confusion_matrix(Y_test, y_pred_adaptive)
            tn, fp, fn, tp = cm_adaptive.ravel() if cm_adaptive.size == 4 else (0, 0, 0, 0)
            sens_adaptive = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec_adaptive = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"Adaptive Kernel - Acc: {acc_adaptive:.4f}, AUC: {auc_adaptive if auc_adaptive is not None else 'N/A'}")
            
            # Save adaptive kernel results
            adaptive_kernel_results = {
                'prediction_results': {
                    'metrics': {
                        'accuracy': acc_adaptive,
                        'auc': auc_adaptive if auc_adaptive is not None else 0,
                        'sensitivity': sens_adaptive,
                        'specificity': spec_adaptive,
                        'confusion_matrix': cm_adaptive.tolist()
                    },
                    'predictions': y_pred_adaptive.tolist(),
                    'probabilities': y_pred_prob_adaptive.tolist() if y_pred_prob_adaptive is not None else None,
                    'y_test': Y_test.tolist()
                }
            }
            
            metrics_adaptive_kernel = log_results(adaptive_kernel_results, metrics_adaptive_kernel, fold_idx)
            
            # ===== Classification using Prior Kernel =====
            print("Classification using Prior Kernel...")
            
            # Extract prior guide kernel (với KNN)
            prior_guide_kernel = data.prior_guide.cpu().numpy()

            # Get indices for training and testing sets
            train_indices = np.where(data.train_mask.cpu().numpy())[0]
            test_indices = np.where(data.test_mask.cpu().numpy())[0]

            # Extract subsets of the kernel for training and testing
            K_train_prior_guide = prior_guide_kernel[np.ix_(train_indices, train_indices)]
            K_test_prior_guide = prior_guide_kernel[np.ix_(test_indices, train_indices)]
            
            
            # Train and test with prior kernel
            classifier_prior = kernels.KernelClassifier()
            classifier_prior.fit(K_train_prior_guide, Y_K_train)
            y_pred_prior = classifier_prior.predict(K_test_prior_guide)
            y_pred_prob_prior = classifier_prior.predict_proba(K_test_prior_guide)[:, 1] if len(np.unique(Y_test)) == 2 else None
            
            # Calculate metrics
            acc_prior = accuracy_score(Y_test, y_pred_prior)
            if y_pred_prob_prior is not None:
                auc_prior = roc_auc_score(Y_test, y_pred_prob_prior)
            else:
                auc_prior = None
                
            cm_prior = confusion_matrix(Y_test, y_pred_prior)
            tn, fp, fn, tp = cm_prior.ravel() if cm_prior.size == 4 else (0, 0, 0, 0)
            sens_prior = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec_prior = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"Prior Kernel - Acc: {acc_prior:.4f}, AUC: {auc_prior if auc_prior is not None else 'N/A'}")
            
            # Save prior kernel results
            prior_kernel_results = {
                'prediction_results': {
                    'metrics': {
                        'accuracy': acc_prior,
                        'auc': auc_prior if auc_prior is not None else 0,
                        'sensitivity': sens_prior,
                        'specificity': spec_prior,
                        'confusion_matrix': cm_prior.tolist()
                    },
                    'predictions': y_pred_prior.tolist(),
                    'probabilities': y_pred_prob_prior.tolist() if y_pred_prob_prior is not None else None,
                    'y_test': Y_test.tolist()
                }
            }
            
            metrics_prior_kernel = log_results(prior_kernel_results, metrics_prior_kernel, fold_idx)
        
        else:
            print(f"No best model found for fold {fold_idx}. Skipping kernel extraction and SVM evaluation.")

    # Save results to CSV
    pd.DataFrame(metrics_atk).to_csv(os.path.join(atk_dir, "results_atk.csv"), index=False)
    pd.DataFrame(metrics_adaptive_kernel).to_csv(os.path.join(results_dir, "results_adaptive_kernel.csv"), index=False)
    pd.DataFrame(metrics_prior_kernel).to_csv(os.path.join(results_dir, "results_prior_kernel.csv"), index=False)
    
    # Compute summary statistics
    df_summary_atk, cf_summary_atk, acc_atk, sens_atk, spec_atk, acc_std_atk, sens_std_atk, spec_std_atk = combine_confusion_matrix(os.path.join(atk_dir, "results_atk.csv"))
    
    if os.path.exists(os.path.join(results_dir, "results_adaptive_kernel.csv")):
        df_summary_adaptive, cf_summary_adaptive, acc_adaptive, sens_adaptive, spec_adaptive, acc_std_adaptive, sens_std_adaptive, spec_std_adaptive = combine_confusion_matrix(os.path.join(results_dir, "results_adaptive_kernel.csv"))
    else:
        acc_adaptive, sens_adaptive, spec_adaptive = 0, 0, 0
        acc_std_adaptive, sens_std_adaptive, spec_std_adaptive = 0, 0, 0
    
    if os.path.exists(os.path.join(results_dir, "results_prior_kernel.csv")):
        df_summary_prior, cf_summary_prior, acc_prior, sens_prior, spec_prior, acc_std_prior, sens_std_prior, spec_std_prior = combine_confusion_matrix(os.path.join(results_dir, "results_prior_kernel.csv"))
    else:
        acc_prior, sens_prior, spec_prior = 0, 0, 0
        acc_std_prior, sens_std_prior, spec_std_prior = 0, 0, 0
    
    # Calculate AUC means and stds
    auc_values_atk = metrics_atk.get('aucs', [0])
    auc_mean_atk = np.mean(auc_values_atk)
    auc_std_atk = np.std(auc_values_atk)
    
    auc_values_adaptive = metrics_adaptive_kernel.get('aucs', [0])
    auc_mean_adaptive = np.mean(auc_values_adaptive)
    auc_std_adaptive = np.std(auc_values_adaptive)
    
    auc_values_prior = metrics_prior_kernel.get('aucs', [0])
    auc_mean_prior = np.mean(auc_values_prior)
    auc_std_prior = np.std(auc_values_prior)
    
    # Create comparison dataframe
    comparison = {
        'Method': ['ATK-QK', 'Adaptive Kernel SVM', 'Prior Kernel SVM'],
        'Accuracy': [acc_atk, acc_adaptive, acc_prior],
        'AUC': [auc_mean_atk, auc_mean_adaptive, auc_mean_prior],
        'Sensitivity': [sens_atk, sens_adaptive, sens_prior],
        'Specificity': [spec_atk, spec_adaptive, spec_prior],
        'Accuracy_std': [acc_std_atk, acc_std_adaptive, acc_std_prior],
        'AUC_std': [auc_std_atk, auc_std_adaptive, auc_std_prior],
        'Sensitivity_std': [sens_std_atk, sens_std_adaptive, spec_std_prior],
        'Specificity_std': [spec_std_atk, spec_std_adaptive, spec_std_prior]
    }
    
    # Add formatted mean±std columns
    comparison['Accuracy_fmt'] = [f"{acc*100:.2f} ± {std*100/args.k_fold:.2f}" for acc, std in zip(comparison['Accuracy'], comparison['Accuracy_std'])]
    comparison['AUC_fmt'] = [f"{auc*100:.2f} ± {std*100/args.k_fold:.2f}" for auc, std in zip(comparison['AUC'], comparison['AUC_std'])]
    comparison['Sensitivity_fmt'] = [f"{sens*100:.2f} ± {std*100/args.k_fold:.2f}" for sens, std in zip(comparison['Sensitivity'], comparison['Sensitivity_std'])]
    comparison['Specificity_fmt'] = [f"{spec*100:.2f} ± {std*100/args.k_fold:.2f}" for spec, std in zip(comparison['Specificity'], comparison['Specificity_std'])]
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(results_dir, "method_comparison.csv"), index=False)
    
    # Print comparison
    print("\n===== Method Comparison =====")
    print(comparison_df[['Method', 'Accuracy_fmt', 'AUC_fmt', 'Sensitivity_fmt', 'Specificity_fmt']])
    
    return df_summary_atk, cf_summary_atk, acc_atk, comparison_df

if __name__ == '__main__':
    args = parse_arguments(description="ATK with Kernel SVM Comparison")
    args.stratified = True
    args.n_repeats = 1
    args.val_size = 0.2
    args.k_fold = 10
    args.kernel_level = "early"
    args.kernel_type = "linear"

    args.dataset = "AD_CN"
    args.modality = ["PET"]
    args.lr = 0.0004
    args.patience = 30
    args.epochs = 1000
    args.num_layers = 4
    args.num_head = 3
    args.top_k = 2
    args.reg_coef = 0.1
    args.embed_dim = 128

    df, cf, acc, comparison_df = main(args)
    print(f"Complete! Final accuracy: {acc:.4f}")