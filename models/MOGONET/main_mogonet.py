""" Example for MOGONET classification
"""
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from .train_test import train_test
except:
    from train_test import train_test

import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc



def combine_confusion_matrix(result_path):
    # Read the CSV file
    df = pd.read_csv(result_path)

    # Extract the confusion matrices from the DataFrame
    confusion_matrix_all = df['confusion_matrices'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Combine/sum all confusion matrices into one confusion matrix
    confusion_matrix_combined = np.zeros((2, 2))
    for i in range(len(confusion_matrix_all)):
        confusion_matrix_combined += confusion_matrix_all[i]
    print(confusion_matrix_combined)
    acc = (confusion_matrix_combined[0][0] + confusion_matrix_combined[1][1]) / np.sum(confusion_matrix_combined)
    sensitivity = confusion_matrix_combined[1][1] / (confusion_matrix_combined[1][1] + confusion_matrix_combined[0][1])
    specificity = confusion_matrix_combined[0][0] / (confusion_matrix_combined[0][0] + confusion_matrix_combined[1][0])

    print("acc: ", acc)
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    return df,confusion_matrix_combined, acc, sensitivity, specificity

def log_results(results, metrics, fold_idx):
    test_metrics = results['prediction_results']['metrics']

    # Ensure keys exist
    for key in ['confusion_matrices', 'classification_reports', 'sensitivities',
                'specificities','fprs', 'tprs'  ,'aucs', 'thresholds', 'y_pred', 'y_test']:
        if key not in metrics:
            metrics[key] = []

    metrics['confusion_matrices'].append(test_metrics['confusion_matrix'])
    metrics['classification_reports'].append(pd.DataFrame(test_metrics['classification_report']))
    metrics['sensitivities'].append(test_metrics['sensitivity'])
    metrics['specificities'].append(test_metrics['specificity'])
    metrics['aucs'].append(test_metrics['auc'])
    metrics['fprs'].append(test_metrics['fpr'])
    metrics['tprs'].append(test_metrics['tpr'])
    metrics['thresholds'].append(test_metrics['thresholds'])
    metrics['y_pred'].append(results['prediction_results']['y_pred_prob'])
    metrics['y_test'].append(results['prediction_results']['y_test'])

    print(f"Fold {fold_idx} accuracy: {test_metrics['accuracy']:.2f}")
    print(f"Fold {fold_idx} sensitivity: {test_metrics['sensitivity']:.2f}")
    print(f"Fold {fold_idx} specificity: {test_metrics['specificity']:.2f}")
    print(f"Fold {fold_idx} AUC: {test_metrics['auc']:.2f}")

    return metrics


def main_mogonet(args):
    metrics = {}
    for fold_idx in range(10):
        print(f"===== FOLD {fold_idx} =====")
        data_folder = f'/home/Thanh/Documents/tvu/E2e_Shallow_Adaptive-KernelLearning/data/{args.dataset}_10fold/fold_{fold_idx}'
        view_list = [1, 2, 3]

        results = train_test(data_folder, view_list, args.num_class,
                             args.lr_e_pretrain, args.lr_e, args.lr_c,
                             args.num_epoch_pretrain, args.num_epoch, args.dim_he_list)
        metrics = log_results(results, metrics, fold_idx)
    df = pd.DataFrame(metrics)
    file_name = f"results/ROSMAP_10fold"
    os.makedirs(file_name, exist_ok=True)
    df, cf, acc, _, _ = combine_confusion_matrix(os.path.join(file_name, "results.csv"))
    print(f"Confusion matrix: {cf}, Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOGONET")
    args = parser.parse_args()

    args.num_epoch_pretrain = 500
    args.num_epoch = 2500
    args.lr_e_pretrain = 1e-3
    args.lr_e = 5e-4
    args.lr_c = 1e-3
    args.num_class = 2
    args.dim_he_list = [200, 200, 100]


    main_mogonet(args)
    # df.to_csv(os.path.join(file_name, "results.csv"), index=False)