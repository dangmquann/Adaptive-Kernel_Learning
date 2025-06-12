import os
import json
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import DataLoader, CrossValidator,train_test_kernel_cv_split, KernelConstructor, prepare_data_config, build_multimodal_graph_list
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix
from models.ATK.utils import train_test,train_test_progressive

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    atk_params = {'num_layers': args.num_layers, 'num_heads': args.num_head, 'dropout': 0.1, 'multihead_agg': 'concat', 'reg_coef': 0,
                  'use_prior': True}
    # Parse arguments
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    modality_str = '_'.join(args.modality)
    output_dir = f"results/ATK_Fusion_{args.dataset}_{modality_str}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir=log_dir, name=f"{args.dataset}_{modality_str}")

    
    # Save args
    args_file = os.path.join(output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Prepare data configuration
    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    data_config = prepare_data_config(args, data_dir)
    logger.info(f"Data configuration: {data_config}")
    
    # Initialize data loader and cross-validator
    data_loader = DataLoader(data_config)
    kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=args.k_fold, n_repeats=1, stratified=False, random_state=args.seed)
    
    # Create data splits
    splits, idxes = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)
    next(idxes)  # Advance iterator once
    
    # Track performance metrics
    metrics = {}
    file_name = f"results/ATK_late/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(file_name, exist_ok=True)
    
    logger.info("="*50)
    logger.info(f"Starting training for dataset: {args.dataset}")
    logger.info(f"Modalities: {args.modality}")
    logger.info(f"Version: {args.version}")
    logger.info("="*50)

    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        logger.info(f"===== FOLD {fold_idx} =====")

        [Xs_test, Y_test, Xs_train, Y_train] = feature_split

        multimodal_graphs = build_multimodal_graph_list(
            Xs_train,
            Y_train,
            Xs_test,
            Y_test,
            top_k=args.top_k,
            random_state=args.seed,
            val_size=args.val_size,
            cuda=device != "cpu",
        )

        results = train_test(
            data=multimodal_graphs,
            embedding_size=args.embed_dim,
            atk_params=atk_params,
            num_epochs=args.epochs,
            lr=args.lr,
            # fine_tune_lr=5e-4,  # Fine-tune learning rate
            random_seed=args.seed,
            early_stopping_patience=args.patience,
            logger=logger,
            fusion=True,  # Set fusion to True for late fusion
        )
        metrics = log_results(results, metrics, fold_idx)
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(file_name, "results.csv"), index=False)

    df, cf, acc,_,_ = combine_confusion_matrix(os.path.join(file_name, "results.csv"))
    return df,cf,acc
    





def effect_of_n_layers(args):
    # Effect of number of layers
    num_layers = range(1, 9)
    results = pd.DataFrame(columns=["num_layers", "accuracy"])
    for n in num_layers:
        args.num_layers = n
        df, cf, acc = main(args)
        results.loc[len(results)] = {"num_layers": n, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["num_layers"], results["accuracy"], marker='o')
    plt.title("Effect of Number of Layers")
    plt.show()
    return results

def effect_of_n_heads(args):
    # Effect of number of heads
    num_heads = range(2, 11)
    results = pd.DataFrame(columns=["num_heads", "accuracy"])
    for n in num_heads:
        args.num_head = n
        df, cf, acc = main(args)
        results.loc[len(results)] = {"num_heads": n, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["num_heads"], results["accuracy"], marker='o')
    plt.title("Effect of Number of Heads")
    plt.show()
    return results

def effect_of_lr(args):
    # Effect of learning rate
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01] # best 0.0005
    results = pd.DataFrame(columns=["lr", "accuracy"])
    for lr in lrs:
        args.lr = lr
        df, cf, acc = main(args)
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
        df, cf, acc = main(args)
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
        df, cf, acc = main(args)
        results.loc[len(results)] = {"embed_dim": dim, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["embed_dim"], results["accuracy"], marker='o')
    plt.title("Effect of Embedding Dimension")
    plt.show()
    return results

def effect_of_top_k(args):
    # Effect of top k
    top_k = range(1,200,5)
    results = pd.DataFrame(columns=["top_k", "accuracy"])
    for k in top_k:
        args.top_k = k
        df, cf, acc = main(args)
        results.loc[len(results)] = {"top_k": k, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["top_k"], results["accuracy"], marker='o')
    plt.title("Effect of Top K")
    plt.show()
    return results

def effect_of_regularization(args):
    # Effect of regularization
    reg_coef = np.linspace(0, 1, 11)/10
    results = pd.DataFrame(columns=["reg_coef", "accuracy"])
    for reg in reg_coef:
        args.reg_coef = reg
        df, cf, acc = main(args)
        results.loc[len(results)] = {"reg_coef": reg, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["reg_coef"], results["accuracy"], marker='o')
    plt.title("Effect of Regularization")
    plt.show()
    return results

if __name__ == '__main__':
    args = parse_arguments(description="ATK")
    # args.dataset = "AD_CN"
    # args.modality = ["PET", "GM", "CSF"]
    args.dataset = "ROSMAP"
    args.modality = ["meth", "miRNA", "mRNA"]
    args.stratified = True
    args.n_repeats = 1
    args.val_size = 0.2
    args.k_fold = 10
    args.lr = 0.0005
    args.patience = 30
    args.epochs = 5000
    args.num_layers = 4
    args.num_head = 3


    df,cf,acc = main(args)
    print(f" cf: {cf}, acc: {acc}")

    # results = effect_of_lr(args)
    # print(results)

    # results = effect_of_patience(args) # no effect
    # print(results)

    # results = effect_of_embed_dim(args) #128
    # print(results)

    # results = effect_of_n_layers(args)
    # print(f"Results of effect of number of layers: {results}")

    # results = effect_of_reg_coef(args)
    # print(f"Results of effect of regularization coefficient: {results}")

    # results = effect_of_top_k(args)
    # print(f"Results of effect of top k: {results}")

    # results = effect_of_val_size(args)
    # print(f"Results of effect of validation size: {results}")

    # results = effect_of_embed_dim(args)
    # print(f"Results of effect of embedding dimension: {results}")

    # results = effect_of_dropout(args)
    # print(f"Results of effect of dropout: {results}")

    # results = effect_of_n_heads(args) #>2, no effect
    # print(results)