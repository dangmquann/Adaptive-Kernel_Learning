'''
Copyright (c) 2025 quanmd

Created Date: Thu, June 12 2025
Author: quanmd

See ../../LICENSE for clarification regarding multiple authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataloader import DataLoader, CrossValidator,train_test_kernel_cv_split, KernelConstructor,  build_graph_data, prepare_data_config
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix
from models.ATK.utils import train_test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    atk_params = {'num_layers': args.num_layers, 'num_heads': args.num_head, 'dropout': 0.1, 'multihead_agg': 'concat', 'reg_coef': args.reg_coef,
                  'use_prior': True}


    logger = setup_logger(log_dir="logs", name="{args.modality}")
    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    data_config = prepare_data_config(args, data_dir)

    # Initialize data loader and cross-validator
    data_loader = DataLoader(data_config)
    kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=args.k_fold, n_repeats=args.n_repeats, stratified=args.stratified, random_state=args.seed)

    # Create data splits
    splits, _ = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)

    logger.info("Chosen modalities:", args.modality)
    logger.info("Data configuration:", data_config)
    logger.info(f"Starting training for dataset: {args.dataset}")
    logger.info(f"Modalities: {args.modality}")

    metrics = {}
    file_name = f"results/ATKQK/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(file_name, exist_ok=True)

    # Split K folds
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")
        # [_, _, _, _, _] = kernel_split
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split
        X_train = Xs_train[args.modality[0]]
        X_test = Xs_test[args.modality[0]]

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
            f"{auc_mean*100:.2f} ± {auc_std*100/args.k_fold:.2f}",
            f"{acc*100:.2f} ± {acc_std*100/args.k_fold:.2f}", 
            f"{sens*100:.2f} ± {sens_std*100/args.k_fold:.2f}", 
            f"{spec*100:.2f} ± {spec_std*100/args.k_fold:.2f}",
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(file_name, "summary_stats.csv"), index=False)
    
    print("Summary statistics:")
    print(summary_df)
    
    return df_summary, cf_summary, acc, summary_df

def effect_of_n_layers(args):
    # Effect of number of layers
    num_layers = range(1, 9)
    results = pd.DataFrame(columns=["num_layers", "accuracy"])
    for n in num_layers:
        args.num_layers = n
        df,cf,acc, summary_df = main(args)
        results.loc[len(results)] = {"num_layers": n, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["num_layers"], results["accuracy"], marker='o')
    plt.title("Effect of Number of Layers")
    plt.show()
    return results

# def effect_of_n_heads(args):
#     # Effect of number of heads
#     num_heads = range(2, 11)
#     results = pd.DataFrame(columns=["num_heads", "accuracy"])
#     for n in num_heads:
#         args.num_head = n
#         df,cf,acc, summary_df = main(args)
#         results.loc[len(results)] = {"num_heads": n, "accuracy": acc}

#     plt.figure(figsize=(6, 6))
#     plt.plot(results["num_heads"], results["accuracy"], marker='o')
#     plt.title("Effect of Number of Heads")
#     plt.show()
#     return results

def effect_of_n_heads(args):
    # Effect of number of heads
    num_heads = range(2, 11)
    
    # Add all parameters you want to track to the columns
    results = pd.DataFrame(columns=[
        "num_heads", "accuracy", "dataset", "modality", 
        "lr", "patience", "epochs", "num_layers", 
        "top_k", "reg_coef"
    ])
    
    for n in num_heads:
        args.num_head = n
        df, cf, acc, summary_df = main(args)
        
        # Save all parameters in the results
        results.loc[len(results)] = {
            "num_heads": n, 
            "accuracy": acc,
            "dataset": args.dataset,
            "modality": "_".join(args.modality),  # Convert list to string
            "lr": args.lr,
            "patience": args.patience,
            "epochs": args.epochs,
            "num_layers": args.num_layers,
            "top_k": args.top_k,
            "reg_coef": args.reg_coef
        }
    
    # Create results directory if it doesn't exist
    os.makedirs("results/parameter_studies", exist_ok=True)
    
    # Save to CSV
    csv_filename = f"results/parameter_studies/head_effect_{args.dataset}_layers{args.num_layers}_{'-'.join(args.modality)}.csv"
    results.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    
    return results

def effect_of_lr(args):
    # Effect of learning rate
    lrs = [0.0006, 0.0003, 0.0007, 0.0004]#0.0001, 0.0005, 0.001, 0.005, 0.01
    results = pd.DataFrame(columns=["lr", "accuracy"])
    for lr in lrs:
        args.lr = lr
        df,cf,acc, summary_df = main(args)
        results.loc[len(results)] = {"lr": lr, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["lr"], results["accuracy"], marker='o')
    plt.title("Effect of Learning Rate")
    plt.show()
    return results

def effect_of_patience(args):
    # Effect of patience
    patience = range(10, 101, 10)
    results = pd.DataFrame(columns=["patience", "accuracy"])
    for p in patience:
        args.patience = p
        df,cf,acc, summary_df = main(args)
        results.loc[len(results)] = {"patience": p, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["patience"], results["accuracy"], marker='o')
    plt.title("Effect of Patience")
    plt.show()
    return results

def effect_of_embed_dim(args):
    # Effect of embedding dimension
    embed_dims = [32, 64, 128, 256]
    results = pd.DataFrame(columns=["embed_dim", "accuracy"])
    for dim in embed_dims:
        args.embed_dim = dim
        df,cf,acc, summary_df = main(args)
        results.loc[len(results)] = {"embed_dim": dim, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["embed_dim"], results["accuracy"], marker='o')
    plt.title("Effect of Embedding Dimension")
    plt.show()
    return results

def effect_of_top_k(args):
    # Effect of top k
    top_k = range(1,10,1)
    results = pd.DataFrame(columns=["top_k", "accuracy"])
    for k in top_k:
        args.top_k = k
        df,cf,acc, summary_df = main(args)
        results.loc[len(results)] = {"top_k": k, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["top_k"], results["accuracy"], marker='o')
    plt.title("Effect of Top K")
    plt.show()
    return results

def effect_of_regularization(args):
    # Effect of regularization
    reg_coef = np.linspace(0, 1, 11)
    results = pd.DataFrame(columns=["reg_coef", "accuracy"])
    for reg in reg_coef:
        args.reg_coef = reg
        df,cf,acc, summary_df = main(args)
        results.loc[len(results)] = {"reg_coef": reg, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["reg_coef"], results["accuracy"], marker='o')
    plt.title("Effect of Regularization")
    plt.show()
    return results


if __name__ == '__main__':
    args = parse_arguments(description="ATK")
    args.stratified = True
    args.n_repeats = 1
    args.val_size = 0.2
    args.k_fold = 10

    # args.dataset = "AD_CN"
    # args.modality = ["PET"]
    # # args.dataset = "ROSMAP"
    # # args.modality = ["meth"]
    # args.lr = 0.0004
    # args.patience = 30
    # args.epochs = 5000
    # args.num_layers = 4
    # args.num_head = 3
    # args.top_k = 2
    # args.reg_coef = 0.1


    df,cf,acc, summary_df = main(args)
    print(f" cf: {cf}, acc: {acc}")

    # results = effect_of_n_layers(args)
    # print(results)

    # results = effect_of_lr(args) # 0.0001
    # print(results)

    # results = effect_of_patience(args) # =30
    # print(results)

    # results = effect_of_embed_dim(args) #128
    # print(results)

    # results = effect_of_top_k(args) # effect
    # print(results)

    # results = effect_of_n_heads(args) # 4 best
    # print(results)

    #
    # results = effect_of_regularization(args) # effect
    # print(results)
