import os
import logging
from datetime import datetime

import pandas as pd


def log_results(results, metrics, fold_idx):
    test_metrics = results['prediction_results']['metrics']

    # Ensure keys exist
    for key in ['confusion_matrices', 'classification_reports', 'sensitivities',
                'specificities','fprs', 'tprs'  ,'aucs', 'thresholds', 'y_pred', 'y_test']:
        if key not in metrics:
            metrics[key] = []

    metrics['confusion_matrices'].append(test_metrics['confusion_matrix'])
    metrics['classification_reports'].append(pd.DataFrame(test_metrics['classification_report']))
    metrics['sensitivities'].append(test_metrics['sensitivity'])
    metrics['specificities'].append(test_metrics['specificity'])
    metrics['aucs'].append(test_metrics['auc'])
    metrics['fprs'].append(test_metrics['fpr'])
    metrics['tprs'].append(test_metrics['tpr'])
    metrics['thresholds'].append(test_metrics['thresholds'])
    metrics['y_pred'].append(results['prediction_results']['y_pred_prob'])
    metrics['y_test'].append(results['prediction_results']['y_test'])

    print(f"Fold {fold_idx} accuracy: {test_metrics['accuracy']:.2f}")
    print(f"Fold {fold_idx} sensitivity: {test_metrics['sensitivity']:.2f}")
    print(f"Fold {fold_idx} specificity: {test_metrics['specificity']:.2f}")
    print(f"Fold {fold_idx} AUC: {test_metrics['auc']:.2f}")

    return metrics


def setup_logger(log_dir="logs", name=""):
    """Setup logger configuration"""
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name with timestamp and modality
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

    # Get the root logger
    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logger