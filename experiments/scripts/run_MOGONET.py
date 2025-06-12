import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataloader import DataLoader, CrossValidator,train_test_kernel_cv_split, KernelConstructor,  build_graph_data, prepare_data_config, build_multimodal_graph_list
from utils import parse_arguments, log_results, setup_logger, combine_confusion_matrix, combine_confusion_matrix_from_df
from models.MOGONET.utils import train_test_MOGONET
from models.MOGONET.models import init_model_dict, init_optim
from models.MOGONET.main_mogonet import main_mogonet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):


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
    file_name = f"results/ATK/{args.dataset}_{'_'.join(args.modality)}"
    os.makedirs(file_name, exist_ok=True)

    # Split K folds
    output_dir_0 = f"/home/Thanh/Documents/tvu/E2e_Shallow_Adaptive-KernelLearning/data/{args.dataset}_{args.k_fold}fold/"
    os.makedirs(output_dir_0, exist_ok=True)
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split
        # X_train = Xs_train[args.modality[0]]
        # X_test = Xs_test[args.modality[0]]
        # Save labels once (train/test)
        output_dir = os.path.join(output_dir_0, f"fold_{fold_idx}")
        os.makedirs(output_dir, exist_ok=True)
        # if fold_idx == 0:
        pd.DataFrame(Y_train).to_csv(os.path.join(output_dir, "labels_tr.csv"), index=False, header=False)
        pd.DataFrame(Y_test).to_csv(os.path.join(output_dir, "labels_te.csv"), index=False, header=False)

        for i, modality in enumerate(Xs_train.keys(), start=1):
            train_file = os.path.join(output_dir, f"{i}_tr.csv")
            test_file = os.path.join(output_dir, f"{i}_te.csv")
            feat_file = os.path.join(output_dir, f"{i}_featname.csv")

            # Save train/test features
            pd.DataFrame(Xs_train[modality]).to_csv(train_file, index=False, header=False)
            pd.DataFrame(Xs_test[modality]).to_csv(test_file, index=False, header=False)

            # Save feature names (f1, f2, ...)
            num_features = Xs_train[modality].shape[1]
            feat_names = pd.DataFrame([f"f{j + 1}" for j in range(num_features)])
            feat_names.to_csv(feat_file, index=False, header=False)
        # Save kernel matrices
    df, cf, acc, _, _ = main_mogonet(args)
    _, cf, acc, _, _ = combine_confusion_matrix_from_df(df)
    return df, cf, acc


if __name__ == '__main__':
    args = parse_arguments(description="MOGONET")
    # args.dataset = "ROSMAP"
    # args.modality = ["meth", "miRNA", "mRNA"]
    args.dataset = "AD_CN"
    args.modality = ["PET", "GM", "CSF"]
    args.stratified = True
    args.n_repeats = 1
    args.val_size = 0.1
    args.k_fold = 10
    # args.lr = 0.0005
    # args.patience = 10
    # args.epochs = 5000
    # args.dim_list = [200, 200, 200]
    # args.lr_e_pretrain = 1e-4
    # args.lr_c = 1e-4
    # args.top_k = 20

    args.num_epoch_pretrain = 500
    args.num_epoch = 2500
    args.lr_e_pretrain = 1e-3
    args.lr_e = 5e-4
    args.lr_c = 1e-3
    args.num_class = 2
    # args.dim_he_list = [200, 200, 100]
    args.dim_he_list = [128, 128, 128]


    main(args)
    # print(f" cf: {cf}, acc: {acc}")

