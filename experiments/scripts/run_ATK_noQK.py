import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataloader import (DataLoader, CrossValidator,train_test_kernel_cv_split, KernelConstructor,  
                        build_graph_data, prepare_data_config, build_graph_data_multimodal_prior)
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix
from models.ATK.utils import train_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    atk_params = {'num_layers': args.num_layers, 'num_heads': args.num_head, 'dropout': 0.1, 'multihead_agg': 'concat', 'reg_coef': args.reg_coef,
                  'use_prior': args.use_prior}

    logger = setup_logger(log_dir="logs", name=f"{args.modality}")
    data_dir = os.path.join(os.getcwd(), "..", "..", f"data/{args.dataset}")
    data_config = prepare_data_config(args, data_dir)

    # Initialize data loader and cross-validator
    data_loader = DataLoader(data_config)
    kernel_constructor = KernelConstructor(args.kernel_level, method=args.kernel_type)
    cv = CrossValidator(n_splits=args.k_fold, n_repeats=args.n_repeats, stratified=args.stratified, random_state=args.seed)

    # Create data splits
    splits, _ = train_test_kernel_cv_split(data_loader, cv, kernel_constructor)

    logger.info(f"Use prior: {args.use_prior}")
    if len(args.modality) > 1 and args.use_prior:
        logger.info(f"Primary modality: {args.selected_modality}")
        logger.info(f"All modalities: {args.modality}")
        if args.prior_method == 1:
            logger.info("Prior method: Sum kernel (average of all modalities)")
        elif args.prior_method == 2:
            logger.info("Prior method: Primary modality kernel only")
    logger.info("Data configuration:", data_config)
    logger.info(f"Starting training for dataset: {args.dataset}")

    metrics = {}
    file_name = f"results/ATK_noQK/{args.dataset}_{''.join(args.selected_modality)}"
    if not args.use_prior:
        file_name += "_random_prior"
    elif args.prior_method == 1:
        file_name += "_sum_kernel"
    elif args.prior_method == 2:
        file_name += "_prior_kernel"
    
    os.makedirs(file_name, exist_ok=True)

    # Split K folds
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split
        selected_modality = args.selected_modality  # Primary modality
        # Select data creation method
        if len(args.modality) > 1 and args.use_prior:
            # Multi-modal case
            if args.prior_method == 1:
                # Method 1: Average all modality kernels (default in build_graph_data_multimodal_prior)
                logger.info("Using multimodal average kernel approach")
                data = build_graph_data_multimodal_prior(
                    Xs_train_dict=Xs_train,
                    Y_train=Y_train,
                    Xs_test_dict=Xs_test,
                    Y_test=Y_test,
                    selected_modality=selected_modality,
                    top_k=args.top_k,
                    random_state=args.seed,
                    val_size=args.val_size,
                    cuda=device != "cpu",
                )
            elif args.prior_method == 2:
                # Method 2: Use only primary modality for prior but keep multimodal information
                logger.info("Using primary modality kernel only")
                # Just use single modality approach with primary modality
                X_train = Xs_train[selected_modality]
                X_test = Xs_test[selected_modality]
                
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
        else:
            # Single modality case
            X_train = Xs_train[selected_modality] 
            X_test = Xs_test[selected_modality] 
            
            # If not using prior, we need to modify the prior guide to be random
            use_random = not args.use_prior
            logger.info(f"Using standard approach with modality: {selected_modality}") 
            logger.info(f"Prior guide: {'random' if use_random else 'kernel-based'}")
            
            # Note: need to modify build_graph_data to support random prior
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
            
            
            n = data.prior_guide.shape[0]
            # random_prior = torch.rand(n, n)
                # This ensures the random prior is different in each run
            random_gen = torch.Generator()
            # Either leave unseeded (uses system entropy) or set a new random seed each time
            random_gen.manual_seed(int(torch.randint(0, 10000, (1,)).item()))
            
            # Generate random prior using this generator
            random_prior = torch.rand(n, n, generator=random_gen)
            # Make it symmetric
            random_prior = (random_prior + random_prior.t()) / 2
            # Set diagonal to 1
            random_prior.fill_diagonal_(1.0)
            
            if device != "cpu" and torch.cuda.is_available():
                random_prior = random_prior.cuda()
                
            # Replace the prior guide
            data.prior_guide = random_prior
            logger.info(f"Random prior guide shape: {data.prior_guide.shape}")
            logger.info(f"Random prior guide: {data.prior_guide}")
            logger.info("Replaced kernel-based prior with random prior")

        results = train_test(
            data=data,
            embedding_size=args.embed_dim,
            atk_params=atk_params,
            num_epochs=args.epochs,
            lr=args.lr,
            random_seed=args.seed,
            early_stopping_patience=args.patience,
            logger=logger,
            use_QK=False,  # Set to True if using ATKQK
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
        df, cf, acc, summary_df = main(args)
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
        df, cf, acc, summary_df = main(args)
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
        df, cf, acc, summary_df = main(args)
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
        df, cf, acc, summary_df = main(args)
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
        df, cf, acc, summary_df = main(args)
        results.loc[len(results)] = {"embed_dim": dim, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["embed_dim"], results["accuracy"], marker='o')
    plt.title("Effect of Embedding Dimension")
    plt.show()
    return results

def effect_of_top_k(args):
    # Effect of top k
    top_k = range(1,100,5)
    results = pd.DataFrame(columns=["top_k", "accuracy"])
    for k in top_k:
        args.top_k = k
        df, cf, acc, summary_df = main(args)
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
        df, cf, acc, summary_df = main(args)
        results.loc[len(results)] = {"reg_coef": reg, "accuracy": acc}

    plt.figure(figsize=(6, 6))
    plt.plot(results["reg_coef"], results["accuracy"], marker='o')
    plt.title("Effect of Regularization")
    plt.show()
    return results


if __name__ == '__main__':
    args = parse_arguments(description="ATK_noQK")

    args.stratified = True
    args.n_repeats = 1
    args.val_size = 0.2
    args.k_fold = 10

    # args.dataset = "AD_CN"
    # args.modality = ["PET", "GM", "CSF"]  # Prior modality kernels
    # args.selected_modality = "CSF"  # Primary modality for prior guide
    # # args.dataset = "ROSMAP"
    # # args.modality = ["meth", "miRNA", "mRNA"]
    # # args.selected_modality = "meth"  # Primary modality for prior guide

    # args.lr = 0.001
    # args.patience = 20
    # args.epochs = 2000
    # args.num_layers = 3
    # args.num_head = 4
    # args.top_k = 2
    # args.reg_coef = 0.1 # Regularization coefficient
    
    # # New flags for data creation method
    # args.use_prior = True  #
    # args.prior_method = 2  # 1: sum/avg kernel (multimodal), 2: primary modality kernel only

    df,cf,acc,summary_df = main(args)
    print(f" cf: {cf}, acc: {acc}")



    # results = effect_of_n_layers(args)
    # print(results)

    # results = effect_of_lr(args)
    # print(results)

    # results = effect_of_patience(args) # no effect
    # print(results)

    # results = effect_of_embed_dim(args) #128
    # print(results)

    # results = effect_of_top_k(args)
    # print(results)

    # results = effect_of_n_heads(args) # best: 5 layers, 10 heads for ADCN and 6 heads for   (5,10)
    # print(results)

    # results = effect_of_regularization(args) # effect
    # print(results)
