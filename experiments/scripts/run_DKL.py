import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.DKL.utils import train_test_DKL
from dataloader import DataLoader, CrossValidator,train_test_kernel_cv_split, KernelConstructor
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix



def main(args):
    k_fold = 10
    data_dir = os.path.join(os.getcwd(), '..', '..', f'data/{args.dataset}')
    data_config = {m: os.path.join(data_dir, f"{m}.csv") for m in args.modality}
    data_config['label'] = os.path.join(data_dir, f"{args.dataset}_label.csv")
    data_loader = DataLoader(data_config)

    cv = CrossValidator(n_splits=k_fold, n_repeats=1, stratified=False, random_state=0)
    kernel_constructor = KernelConstructor('early', method='linear')
    splits, _ = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)

    logger = setup_logger(log_dir="logs", name=f"DKL_{args.modality}")
    file_name = f"results/DKL/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(file_name, exist_ok=True)
    metrics = {}
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== Fold {fold_idx} =====")
        Xs_train, Y_train, Xs_test, Y_test = feature_split
        X_train = Xs_train[args.modality[0]]
        X_test = Xs_test[args.modality[0]]
        results = train_test_DKL(X_train, Y_train, X_test, Y_test, epochs=args.epochs,batch_size=args.batch_size, lr=args.lr,
                patience=args.patience)
        metrics = log_results(results, metrics, fold_idx)
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(file_name, "results.csv"), index=False)

   # Get metrics with standard deviations
    df_summary, cf_summary, acc, sens, spec, acc_std, sens_std, spec_std = combine_confusion_matrix(os.path.join(file_name, "results.csv"))
    
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
    summary_df.to_csv(os.path.join(file_name, "summary_stats.csv"), index=False)
    
    print("Summary statistics:")
    print(summary_df)
    
    return df_summary, cf_summary, acc, summary_df


def effect_of_lr(args):
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    results = pd.DataFrame(columns=["lr", "accuracy"])
    for n in lrs:
        args.lr = n
        df, cf, acc = main(args)
        results.loc[len(results)] = {"lr": n, "accuracy": acc}
    plt.figure(figsize=(6, 6))
    plt.plot(results["lr"], results["accuracy"], marker='o')
    plt.title("Effect of Learning Rate")
    plt.xscale('log')
    plt.show()
    return results

def effect_of_batch_size(args):
    batch_sizes = [16, 32, 64, 128, 231]
    results = pd.DataFrame(columns=["batch_size", "accuracy"])
    for n in batch_sizes:
        args.batch_size = n
        df, cf, acc = main(args)
        results.loc[len(results)] = {"batch_size": n, "accuracy": acc}
    plt.figure(figsize=(6, 6))
    plt.plot(results["batch_size"], results["accuracy"], marker='o')
    plt.title("Effect of Batch Size")
    plt.xscale('log')
    plt.show()
    return results

def effect_of_patience(args):
    patience = [10, 20, 30, 40, 50]
    results = pd.DataFrame(columns=["patience", "accuracy"])
    for n in patience:
        args.patience = n
        df, cf, acc = main(args)
        results.loc[len(results)] = {"patience": n, "accuracy": acc}
    plt.figure(figsize=(6, 6))
    plt.plot(results["patience"], results["accuracy"], marker='o')
    plt.title("Effect of Patience")
    plt.xscale('log')
    plt.show()
    return results


if __name__ == '__main__':
    args = parse_arguments(description="Deep Kernel Learning Experiment")
    # args.dataset = "AD_CN"
    # args.modality = ["PET"]
    # args.dataset = "ROSMAP"
    # args.modality = ["meth", "miRNA", "mRNA"]
    # args.epochs = 10000
    # args.patience = 40
    # args.batch_size = 32
    # args.lr = 0.01

    df,cf,acc, summary_df = main(args)
    print(f"Final Results: cf: {cf}, acc: {acc}")

    # results = effect_of_lr(args)
    # print(f"Final Results: {results}")
    #
    # results = effect_of_batch_size(args)
    # print(f"Final Results: {results}")

    # results = effect_of_patience(args)
    # print(f"Final Results: {results}")