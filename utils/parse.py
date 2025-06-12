import os
import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(description="Experiment"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, default="AD_CN", choices=["AD_CN", "AD_MCI", "CN_MCI", "ROSMAP"],
                        help="Dataset to use")
    parser.add_argument("--modality", type=str, nargs='+', default=["PET", "GM", "CSF"],
                        choices=["PET", "GM", "MRI", "CSF", "concat_ROSMAP", "meth", "miRNA", "mRNA", "concat_ADNI"],
                        help="Modalities to use")
    parser.add_argument("--selected_modality", type=str, default="PET", help="Selected modality for the experiment")
    parser.add_argument("--use_prior", type=str2bool, default=True, 
                        help="Whether to use prior knowledge for the experiment")
    parser.add_argument("--prior_method", type=int, default=1, choices=[1,2],
                        help="Prior method (1: sum/avg kernel, 2: primary modality kernel)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    parser.add_argument("--num_head", type=int, default=3, help="Number of attention heads")
    parser.add_argument("--version", type=str, default="v1", help="Version of the experiment")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--max_order", type=int, default=3, help="Maximum interaction order (None = use all)")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Top-k neighbors to use in graph construction (None = use all)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--kernel_type", type=str, default="linear",
                        choices=["polynomial", "rbf", "linear", "ensemble"],
                        help="Kernel method to use for initial similarity")
    parser.add_argument("--kernel_level", type=str, default="early",
                        choices=["early", "middle", "late", "multi-kernel"], help="Kernel level to use")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of cross validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kernel_combination", type=str, default="sum",
                        choices=["sum", "SKL", "AMKL", "EasyMKL", "AverageMKL", "mogonet", "DKL"],
                        help="Kernel combination method to use")
    parser.add_argument("--classifier", type=str, default="SVC", choices=["SVC", "MOGONET", "GCT"],
                        help="Classifier to use")
    parser.add_argument('--num_kernels', type=int, default=10, help='Number of RBF kernels')
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of pretraining epochs')
    parser.add_argument('--lambda_reg', type=float, default=1e-3, help='Regularization coefficient')
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of repeats for cross-validation')
    parser.add_argument('--reg_coef', type=float, default=0.1, help='Regularization coefficient for ATK')
    parser.add_argument("--strategy", type=str, default="sequential", 
                        choices=["sequential", "grid"],
                        help="Tuning strategy to use")

    return parser.parse_args()