import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score

from dataloader import DataLoader, CrossValidator,train_test_kernel_cv_split, KernelConstructor, prepare_data_config
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix
import utils
from models import kernels






def main(args):
    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    data_config = prepare_data_config(args, data_dir)
    data_loader = DataLoader(data_config)

    chosen_modality = args.modality

    kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=args.k_fold, n_repeats=1, stratified=False, random_state=1)
    
    splits, idxes = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)
    metrics = {}

    result_dir = f"results/{args.kernel_combination}/{args.dataset}"
    os.makedirs(result_dir, exist_ok=True)

    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")
        
        # Extract the data
        [Xs_kernel_train, Y_K_train, Xs_kernel_train_test, Xs_kernel_test_test, Y_K_test] = kernel_split
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split
        
        # Stack kernel matrices
        Xs_kernel_train = utils.stack_kernel_matrix_from_dict(Xs_kernel_train)
        Xs_kernel_train_test = utils.stack_kernel_matrix_from_dict(Xs_kernel_train_test)
        Xs_kernel_test_test = utils.stack_kernel_matrix_from_dict(Xs_kernel_test_test)

        kernel_combiners = {}
        kernels_train = {}
        kernels_train_test = {}
        kernels_test = {}
        print(args.kernel_combination)
        if args.kernel_combination == "sum":
            kernel_combiner = kernels.KernelCombination(method=args.kernel_combination)
            kernel_combiner.fit(Xs_kernel_train, Y_K_train, Xs_kernel_train_test, Xs_kernel_test_test)
            combined_train_kernel = kernel_combiner.transform(Xs_kernel_train)
            combined_test_kernel = kernel_combiner.transform(Xs_kernel_test_test)
            combined_train_test_kernel = kernel_combiner.transform(Xs_kernel_train_test)
        elif args.kernel_combination == "AverageMKL":
            list_of_kernels_train = [Xs_kernel_train[i] for i in range(Xs_kernel_train.shape[0])]
            list_of_kernels_train_test = [Xs_kernel_train_test[i].T for i in range(Xs_kernel_train_test.shape[0])]

            kernel_combiner = kernels.KernelCombination(method=args.kernel_combination)
            kernel_combiner.fit(list_of_kernels_train, Y_train, list_of_kernels_train_test, Y_test)

            combined_train_kernel = kernel_combiner.transform(list_of_kernels_train)
            # combined_test_kernel = kernel_combiner.transform(list_of_kernels_test)
            combined_train_test_kernel = kernel_combiner.transform(list_of_kernels_train_test)

        elif args.kernel_combination == "EasyMKL":
            list_of_kernels_train = [Xs_kernel_train[i] for i in range(Xs_kernel_train.shape[0])]
            list_of_kernels_train_test = [Xs_kernel_train_test[i].T for i in range(Xs_kernel_train_test.shape[0])]

            kernel_combiner = kernels.KernelCombination(method=args.kernel_combination)
            kernel_combiner.fit(list_of_kernels_train, Y_train, list_of_kernels_train_test, Y_test)

            combined_train_kernel = kernel_combiner.transform(list_of_kernels_train)
            # combined_test_kernel = kernel_combiner.transform(list_of_kernels_test)
            combined_train_test_kernel = kernel_combiner.transform(list_of_kernels_train_test)

        else:
            for m,modality in enumerate(chosen_modality):
                kernels_train_i, kernels_train_test_i, kernels_test_i = kernels.utils.train_test_DL(
                    Xs_kernel_train[m],
                    Y_K_train,
                    Xs_kernel_train_test[m],
                    Y_K_test,
                    method=args.kernel_combination,
                    num_kernels=args.num_kernels,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience
                )
                kernels_train[modality] = kernels_train_i
                kernels_train_test[modality] = kernels_train_test_i
                kernels_test[modality] = kernels_test_i

                kernels_train[modality] = kernels_train_i
                kernels_train_test[modality] = kernels_train_test_i
                kernels_test[modality] = kernels_test_i

            # Combine kernels across modalities
            combined_train_kernel = sum(kernels_train[modality] for modality in chosen_modality)
            combined_train_test_kernel = sum(kernels_train_test[modality] for modality in chosen_modality)
            combined_test_kernel = sum(kernels_test[modality] for modality in chosen_modality)



        print(f"Combined kernel shape: {combined_train_kernel.shape}, {combined_train_test_kernel.shape}, {combined_test_kernel.shape}")
        clf = kernels.KernelClassifier()
        clf.fit(combined_train_kernel, Y_K_train)

        y_pred = clf.predict(combined_train_kernel)
        y_pred_prob = clf.predict_proba(combined_train_test_kernel)

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
    df.to_csv(os.path.join(result_dir, f"results.csv"), index=False)

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
            f"{auc_mean*100:.2f} ± {auc_std*100/args.k_fold:.2f}",
            f"{acc*100:.2f} ± {acc_std*100/args.k_fold:.2f}", 
            f"{sens*100:.2f} ± {sens_std*100/args.k_fold:.2f}", 
            f"{spec*100:.2f} ± {spec_std*100/args.k_fold:.2f}",
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(result_dir, "summary_stats.csv"), index=False)
    
    print("Summary statistics:")
    print(summary_df)
    
    return df_summary, cf_summary, acc, summary_df


def compare_kernel_combination(args):
    kernel_combinations = ["sum", "AverageMKL", "EasyMKL"]
    results = pd.DataFrame(columns=["kernel_combination", "accuracy"])

    for kernel_combination in kernel_combinations:
        args.kernel_combination = kernel_combination
        df, cf, acc = main(args)
        results.loc[len(results)] = {
            "kernel_combination": kernel_combination,
            "accuracy": acc
        }

    plt.figure(figsize=(6, 6))
    plt.plot(results["kernel_combination"], results["accuracy"], marker='o')
    plt.title("Effect of Kernel Combination")
    plt.xlabel("Kernel Combination Method")
    plt.ylabel("Accuracy")
    plt.xticks(ticks=results["kernel_combination"], labels=results["kernel_combination"], rotation=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    args = parse_arguments("MKL")
    args.k_fold = 10
    # args.dataset = "AD_CN"
    # args.modality = ["PET", "GM", "CSF"]
    
    # args.dataset = "ROSMAP"
    # args.modality = ["meth", "miRNA", "mRNA"]

    # args.kernel_combination = "AverageMKL" # sum, AverageMKL, EasyMKL
    # args.kernel_level = "early"
    # args.kernel_type = "rbf" #polynominal, rbf, linear, ensemble

    df_summary, cf_summary, acc_summary, summary_df = main(args)
    print("Summary Confusion Matrix:\n", cf_summary)

    # results = compare_kernel_combination(args)
    # print("Results of Kernel Combination:\n", results)

    # TODO: AMKL, SKL, DKL are not working
