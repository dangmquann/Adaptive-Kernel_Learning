import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os
import argparse
import numpy as np


def prepare_data_config(args, data_dir):
    """Prepare data configuration based on arguments."""
    if args.dataset == "ROSMAP":
        # Define paths for ROSMAP dataset
        data_config = {
            "meth": os.path.join(data_dir, "meth.csv"),
            "miRNA": os.path.join(data_dir, "miRNA_scaled.csv"),
            "mRNA": os.path.join(data_dir, "mRNA.csv"),
            "concat_ROSMAP": os.path.join(data_dir, "concat_ROSMAP.csv"),
            "label": os.path.join(data_dir, f"{args.dataset}_label.csv")
        }
    else:
        # Define all available data sources
        data_config = {
            "GM": os.path.join(data_dir, "GM.csv"),
            "PET": os.path.join(data_dir, "PET.csv"),
            "CSF": os.path.join(data_dir, "CSF_scaled.csv"),
            "MRI": os.path.join(data_dir, "MRI.csv"),
            "concat_ADNI": os.path.join(data_dir, "concat_ADNI.csv"),
            "label": os.path.join(data_dir, f"{args.dataset}_label.csv")
        }

    # Filter config to only include selected modalities and label
    keep_keys = args.modality + ["label"]
    return {k: v for k, v in data_config.items() if k in keep_keys}


def train_test_kernel_cv_split(data_loader, cv, kernel_constructor):
    all_subjects, labels = data_loader.get_all_subjects_labels()
    modalities = data_loader.get_modalities()
    print("modality before processing: ", modalities)

    splits = []
    for fold, (train_index, test_index) in enumerate(cv.get_splits(all_subjects, labels)):
        print(f"Processing fold {fold}")
        y = data_loader.get_data("label")
        y_train = y[train_index]
        y_test = y[test_index]
        Y_K_train = kernel_constructor.create_ouput_kernel(y_train)
        Y_K_test = kernel_constructor.create_ouput_kernel(y_test)

        Xs_kernel_train, Xs_kernel_train_test, Xs_kernel_test_test = {}, {}, {}
        Xs_train, Xs_test = {}, {}
        for modality in modalities:
            X = data_loader.get_data(modality)

            X_train = X[train_index]
            y_train = y[train_index]

            X_test = X[test_index]
            X_K_train, _ = kernel_constructor.fit_transform(X_train, y_train)
            X_K_train_test, X_K_test_test = kernel_constructor.transform(X_test)

            Xs_kernel_train[modality] = X_K_train
            Xs_kernel_train_test[modality] = X_K_train_test
            Xs_kernel_test_test[modality] = X_K_test_test

            Xs_train[modality] = X_train
            Xs_test[modality] = X_test
        splits.append([(Xs_kernel_train, Y_K_train, Xs_kernel_train_test, Xs_kernel_test_test, Y_K_test),
                       (Xs_train, y_train, Xs_test, y_test)])

    return splits, cv.get_splits(all_subjects, labels)


def load_data(dataPath, input=True):
    if input:
        data = pd.read_csv(dataPath, header=None)
        data = data.values
        return data

    else:
        data = pd.read_csv(dataPath)
        data = data['encoded']
        le = LabelEncoder()
        y = le.fit_transform(data)
        return y
