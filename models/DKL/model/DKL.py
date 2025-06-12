import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import ScaleKernel, RBFKernel




class FeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[3]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Define the GP model using variational inference
class DKLModel(ApproximateGP):
    def __init__(self, feature_extractor, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x.add_jitter(1e-3))
