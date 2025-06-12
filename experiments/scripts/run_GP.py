import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score
)

from dataloader import DataLoader, CrossValidator,train_test_kernel_cv_split, KernelConstructor,  build_graph_data, prepare_data_config, build_multimodal_graph_list
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix

# Set up logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPC")

def main(args):
    k_fold = 10
    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    data_config = {m: os.path.join(data_dir, f"{m}.csv") for m in args.modality}
    data_config["label"] = os.path.join(data_dir, f"{args.dataset}_label.csv")

    logger.info(f"Chosen modalities: {args.modality}")
    logger.info(f"Data config: {data_config}")
    data_loader = DataLoader(data_config)

    kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=k_fold, n_repeats=1, stratified=False, random_state=1)
    splits, _ = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)

    result_dir = f"results/GPC/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(result_dir, exist_ok=True)

    metrics = {}
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        logger.info(f"===== FOLD {fold_idx} =====")
        Xs_train, Y_train, Xs_test, Y_test = feature_split
        X_train = np.hstack([np.array(Xs_train[m]) for m in args.modality])
        X_test = np.hstack([np.array(Xs_test[m]) for m in args.modality])
        Y_train = np.array(Y_train).ravel()
        Y_test = np.array(Y_test).ravel()

        kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=0.1)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=2,
            max_iter_predict=200,
            random_state=42
        )
        gpc.fit(X_train, Y_train)
        y_pred_prob = gpc.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold_youden = thresholds[youden_index]
        f1_scores = [f1_score(Y_test, (y_pred_prob > t).astype(int)) for t in thresholds]
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
        test_preds = (y_pred_prob > optimal_threshold_f1).astype(int)

        acc = accuracy_score(Y_test, test_preds)
        auc = roc_auc_score(Y_test, y_pred_prob)
        cm = confusion_matrix(Y_test, test_preds)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        cr = classification_report(Y_test, test_preds, output_dict=True)

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

        metrics = log_results(results, metrics, fold_idx)

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(result_dir, "results.csv"), index=False)

    # Get metrics with standard deviations
    df_summary, cf_summary, acc, sens, spec, acc_std, sens_std, spec_std = combine_confusion_matrix(os.path.join(result_dir, "results.csv"))
    
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
            f"{auc_mean*100:.2f} ± {auc_std*100/k_fold:.2f}",
            f"{acc*100:.2f} ± {acc_std*100/k_fold:.2f}", 
            f"{sens*100:.2f} ± {sens_std*100/k_fold:.2f}", 
            f"{spec*100:.2f} ± {spec_std*100/k_fold:.2f}",
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(result_dir, "summary_stats.csv"), index=False)
    
    print("Summary statistics:")
    print(summary_df)
    
    return df_summary, cf_summary, acc, summary_df


if __name__ == "__main__":
    args = parse_arguments("GP")
    # args.dataset = "AD_CN"
    # args.modality = ["PET"]
    # args.dataset = "ROSMAP"
    # args.modality = ["meth", "miRNA", "mRNA"]
    df_summary, cf_summary, acc_summary, summary_df = main(args)
    print(f"confusion matrix: {cf_summary}, accuracy: {acc_summary}")