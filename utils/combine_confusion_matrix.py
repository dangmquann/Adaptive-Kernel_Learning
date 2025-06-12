# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc




# def combine_confusion_matrix_from_df(df):
#     # Read the CSV file
#     # Extract the confusion matrices from the DataFrame
#     confusion_matrix_all = df['confusion_matrices'].apply(lambda x: eval(x) if isinstance(x, str) else x)

#     # Combine/sum all confusion matrices into one confusion matrix
#     confusion_matrix_combined = np.zeros((2, 2))
#     for i in range(len(confusion_matrix_all)):
#         confusion_matrix_combined += confusion_matrix_all[i]
#     print(confusion_matrix_combined)
#     acc = (confusion_matrix_combined[0][0] + confusion_matrix_combined[1][1]) / np.sum(confusion_matrix_combined)
#     sensitivity = confusion_matrix_combined[1][1] / (confusion_matrix_combined[1][1] + confusion_matrix_combined[0][1])
#     specificity = confusion_matrix_combined[0][0] / (confusion_matrix_combined[0][0] + confusion_matrix_combined[1][0])

#     print("acc: ", acc)
#     print("sensitivity: ", sensitivity)
#     print("specificity: ", specificity)
#     return df,confusion_matrix_combined, acc, sensitivity, specificity

# def combine_confusion_matrix(result_path):
#     # Read the CSV file
#     df = pd.read_csv(result_path)
#     return combine_confusion_matrix_from_df(df)

# if __name__ == "__main__":
#     df, combine_confusion_matrix, _,_,_ = combine_confusion_matrix("/home/Thanh/Documents/tvu/E2e_Shallow_Adaptive-KernelLearning/experiments/scripts/results/ATK/AD_CN_PET_v1/metrics.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def combine_confusion_matrix_from_df(df):
    # Extract the confusion matrices from the DataFrame
    confusion_matrix_all = df['confusion_matrices'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Arrays to store individual metrics
    accuracies = []
    sensitivities = []
    specificities = []
    
    # Calculate metrics for each individual confusion matrix
    for cm in confusion_matrix_all:
        acc = (cm[0][0] + cm[1][1]) / np.sum(cm)
        
        # Handle cases where denominators could be zero
        if (cm[1][1] + cm[0][1]) > 0:
            sensitivity = cm[1][1] / (cm[1][1] + cm[0][1])
        else:
            sensitivity = 0
            
        if (cm[0][0] + cm[1][0]) > 0:
            specificity = cm[0][0] / (cm[0][0] + cm[1][0])
        else:
            specificity = 0
            
        accuracies.append(acc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # Combine/sum all confusion matrices into one confusion matrix
    confusion_matrix_combined = np.zeros((2, 2))
    for i in range(len(confusion_matrix_all)):
        confusion_matrix_combined += confusion_matrix_all[i]
    print(confusion_matrix_combined)
    
    # Calculate combined metrics
    acc = (confusion_matrix_combined[0][0] + confusion_matrix_combined[1][1]) / np.sum(confusion_matrix_combined)
    sensitivity = confusion_matrix_combined[1][1] / (confusion_matrix_combined[1][1] + confusion_matrix_combined[0][1])
    specificity = confusion_matrix_combined[0][0] / (confusion_matrix_combined[0][0] + confusion_matrix_combined[1][0])

    # Calculate standard deviations
    acc_std = np.std(accuracies)
    sensitivity_std = np.std(sensitivities)
    specificity_std = np.std(specificities)
    
    print("acc: ", acc, "±", acc_std)
    print("sensitivity: ", sensitivity, "±", sensitivity_std)
    print("specificity: ", specificity, "±", specificity_std)
    
    return df, confusion_matrix_combined, acc, sensitivity, specificity, acc_std, sensitivity_std, specificity_std

def combine_confusion_matrix(result_path):
    # Read the CSV file
    df = pd.read_csv(result_path)
    return combine_confusion_matrix_from_df(df)

if __name__ == "__main__":
    df, combine_confusion_matrix, acc, sensitivity, specificity, acc_std, sensitivity_std, specificity_std = combine_confusion_matrix("/home/Thanh/Documents/tvu/E2e_Shallow_Adaptive-KernelLearning/experiments/scripts/results/ATK/AD_CN_PET_v1/metrics.csv")