import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score
)
from dataloader import (
    DataLoader, KernelConstructor, CrossValidator,
    train_test_kernel_cv_split
)
from utils import parse_arguments, log_results, combine_confusion_matrix
from models import kernels


def main(args):
    chosen_modality = args.modality
    version = args.version
    kernel_level = args.kernel_level

    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    result_dir = f"results/SVM/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(result_dir, exist_ok=True)

    if args.dataset == "ROSMAP":
        data_config = {m: os.path.join(data_dir, f"{m}.csv") for m in ["meth", "miRNA", "mRNA"]}
    else:
        data_config = {m: os.path.join(data_dir, f"{m}.csv") for m in ["GM", "PET", "CSF", "MRI"]}
    data_config["label"] = os.path.join(data_dir, f"{args.dataset}_label.csv")

    keep_keys = chosen_modality + ["label"]
    data_config = {k: v for k, v in data_config.items() if k in keep_keys}
    print("Chosen modalities:", chosen_modality)
    print("Data configuration:", data_config)
    data_loader = DataLoader(data_config)

    k_fold = 10
    kernel_constructor = KernelConstructor(kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=k_fold, n_repeats=1, stratified=False, random_state=1)
    splits, idxes = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)
    idx = next(idxes)

    metrics = {}

    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")
        [Xs_kernel_train, Y_K_train, Xs_kernel_train_test, Xs_kernel_test_test, Y_K_test] = kernel_split
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split

        # X_train_kernel = Xs_kernel_train[chosen_modality[0]]
        # X_train_test_kernel = Xs_kernel_train_test[chosen_modality[0]]
        # X_test_test_kernel = Xs_kernel_test_test[chosen_modality[0]]
        X_train_kernel = np.mean([Xs_kernel_train[m] for m in chosen_modality], axis=0)
        X_train_test_kernel = np.mean([Xs_kernel_train_test[m] for m in chosen_modality], axis=0)
        X_test_test_kernel = np.mean([Xs_kernel_test_test[m] for m in chosen_modality], axis=0)


        clf = kernels.KernelClassifier()
        clf.fit(X_train_kernel, Y_K_train)

        y_pred = clf.predict(X_train_test_kernel)
        y_pred_prob = clf.predict_proba(X_train_test_kernel)
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

        acc = accuracy_score(Y_test, y_pred)
        auc = roc_auc_score(Y_test, y_pred_prob)
        cm = confusion_matrix(Y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        cr = classification_report(Y_test, y_pred, output_dict=True)

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
                    # 'optimal_threshold_youden': float(optimal_threshold_youden),
                    # 'optimal_threshold_f1': float(optimal_threshold_f1)
                },
                # 'probabilities': y_pred_prob.tolist(),
                'predictions': y_pred.tolist(),
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

if __name__ == '__main__':
    args = parse_arguments("early_kernel classifier")
    args.dataset = "AD_CN"
    args.modality = ["PET"]
    args.kernel_level = "early"
    args.kernel_type = "cosine"

    df_summary, cf_summary, acc_summary, summary_df = main(args)
