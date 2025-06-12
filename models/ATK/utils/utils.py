import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch



def train_test_kernel_cv_split(data_loader,cv,kernel_constructor):
    all_subjects, labels = data_loader.get_all_subjects_labels()
    modalities = data_loader.get_modalities()
    print("modality before processing: ", modalities)

    splits = []
    for fold, (train_index, test_index) in enumerate(cv.get_splits(all_subjects,labels)):
        print(f"Processing fold {fold}")
        y = data_loader.get_data("label")
        y_train = y[train_index]
        y_test = y[test_index]
        Y_K_train = kernel_constructor.create_ouput_kernel(y_train)
        Y_K_test = kernel_constructor.create_ouput_kernel(y_test)

        Xs_kernel_train, Xs_kernel_train_test, Xs_kernel_test_test  = {}, {}, {}
        Xs_train, Xs_test = {}, {}
        for modality in modalities:
            X = data_loader.get_data(modality)


            X_train = X[train_index]
            y_train = y[train_index]

            X_test = X[test_index]
            X_K_train, _ = kernel_constructor.fit_transform(X_train, y_train)
            X_K_train_test, X_K_test_test  = kernel_constructor.transform(X_test)

            Xs_kernel_train[modality] = X_K_train
            Xs_kernel_train_test[modality] = X_K_train_test
            Xs_kernel_test_test[modality] = X_K_test_test

            Xs_train[modality] = X_train
            Xs_test[modality] = X_test
        splits.append([(Xs_kernel_train, Y_K_train, Xs_kernel_train_test, Xs_kernel_test_test,Y_K_test), (Xs_train, y_train, Xs_test, y_test)])

    return splits, cv.get_splits(all_subjects,labels)



def multimodal_supervised_kernel(X_list,y,random_state=42):
    K_X = []
    # Z = np.array(range(len(y)))
    indices = np.argsort(y)
    for X in X_list:
        K,K_train, K_val,K_test,Z_ = supervised_kernel(X,y,random_state,indices=indices)
        K_X.append(K)
    K_X = np.stack(K_X)
    print(K_X.shape)
    K_X = torch.from_numpy(K_X)
    # print(Z_.shape,np.array(y).shape)
    y = np.array(y)[indices]
    y_kernel = gt_kernel(y)
    y_kernel = torch.tensor(y_kernel)
    y_kernel = y_kernel.unsqueeze(dim=0)
    y_kernel = torch.tensor(y_kernel,  dtype=torch.float32)
    K_X = torch.tensor(K_X,  dtype=torch.float32)
    return K_X, y_kernel, Z_
def normalize_similarity_matrix(similarity_matrix):
    # Find the minimum and maximum values in the matrix
    min_val = torch.min(similarity_matrix)
    max_val = torch.max(similarity_matrix)

    # Normalize the matrix to the range [0, 1]
    normalized_matrix = (similarity_matrix - min_val) / (max_val - min_val)

    return normalized_matrix


def gt_kernel(y):
  label_encoder = LabelEncoder()
  encoded_labels = label_encoder.fit_transform(y)
  encoded_new_label = encoded_labels*2-1
  y = encoded_new_label.reshape(1,-1)
  K = y.T @ y
  # K = np.zeros(len(y)) + np.eye(len(y))
  # for i in range(len(y)):
  #   for j in range(i+1, len(y)):
  #     if y[i] == y[j]:
  #       K[i,j] = K[j,i] = 1
  return K

def centralized_kernel(K):
  n = K.shape[0]
  H = torch.eye(n) - torch.ones((n, n)) / n
  K = H @  K @ H
  return K


def sort_XyZ(X,y,Z):
  indices = np.argsort(y)
  y = y[indices]
  X = X[indices,:]
  Z = Z[indices]
  return X,y,Z 
def supervised_kernel(X,y,n_estimators=100, plot=True, scaler=True, Z=None,random_state=42,indices=None):
  
	if indices is None:
		indices = np.argsort(y)
                
	y = y[indices]
	if Z is None:
		Z = np.array(range(len(y)))

	

	#===================================================
	X = X[indices,:]
	if scaler:
		scaler = StandardScaler()
		X = scaler.fit_transform(X)

	Z_train, Z_test,  y_train, y_test, = train_test_split( Z,y, test_size=0.2, random_state=random_state) 
	X_train,y_train, X_test, y_test = X[Z_train,:], y[Z_train],X[Z_test,:], y[Z_test]
	X_train, X_val, y_train, y_val , Z_train, Z_val= train_test_split(X_train, y_train, Z_train,test_size=0.2, random_state=1)

	X_train,y_train,Z_train = sort_XyZ(X_train,y_train,Z_train)
	X_val,y_val,Z_val = sort_XyZ(X_val,y_val,Z_val)
	X_test,y_test,Z_test = sort_XyZ(X_test,y_test,Z_test)



	# Create a Random Forest classifier
	rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42,
																				#  max_features=5,
																				#  max_depth=5
																				)

	rf_classifier.fit(X_train, y_train)

	# Get the number of trees in the forest
	num_trees = len(rf_classifier.estimators_)



	# =================
	# Make predictions on the training data
	y_pred = rf_classifier.predict(X_train)
	# ==========================
	# Make predictions on the testing data
	y_pred = rf_classifier.predict(X_val)

	# Evaluate the model
	accuracy = accuracy_score(y_val, y_pred)
	print(f"Accuracy test: {accuracy:.2f}")

	# Display the classification report
	print("Classification Report test:")
	print(classification_report(y_val, y_pred))


	#  =============
	print("Validation")
	# Initialize a matrix to store pairwise similarities
	pairwise_similarities = [[0] * len(X_train) for _ in range(len(X_train))]

	
	# Iterate over each tree in the forest
	for tree in rf_classifier.estimators_:
			# Get the leaf indices for each example
			leaf_indices = tree.apply(X_train)

			# Update pairwise similarities based on leaf indices
			for i in range(len(X_train)):
					for j in range(i + 1, len(X_train)):
							if leaf_indices[i] == leaf_indices[j]:
									pairwise_similarities[i][j] += 1
									pairwise_similarities[j][i] += 1

	# Normalize the pairwise similarities by the number of trees
	normalized_pairwise_similarities = [[similarity / num_trees for similarity in row] for row in pairwise_similarities]
	normalized_pairwise_similarities_train = np.array(normalized_pairwise_similarities)


	# =============
	print("All")
	# Initialize a matrix to store pairwise similarities
	pairwise_similarities = [[0] * len(X) for _ in range(len(X))]

	# Iterate over each tree in the forest
	for tree in rf_classifier.estimators_:
			# Get the leaf indices for each example
			leaf_indices = tree.apply(X)

			# Update pairwise similarities based on leaf indices
			for i in range(len(X)):
					for j in range(i + 1, len(X)):
							if leaf_indices[i] == leaf_indices[j]:
									pairwise_similarities[i][j] += 1
									pairwise_similarities[j][i] += 1

	# Normalize the pairwise similarities by the number of trees
	normalized_pairwise_similarities = [[similarity / num_trees for similarity in row] for row in pairwise_similarities]

	normalized_pairwise_similarities_all = np.array(normalized_pairwise_similarities)
	if plot:
		plt.figure()
		plt.imshow(np.array(normalized_pairwise_similarities_all),cmap="jet")
		plt.colorbar()
		plt.clim(0,1)
		plt.xticks([np.where(y == 0)[0][-1]], "-")
		plt.yticks([np.where(y == 0)[0][-1]], "-")

	y_pred = rf_classifier.predict(X)
	conf_matrix = confusion_matrix(y, y_pred)
	print('Confusion Matrix:')
	print(conf_matrix)


	print("Validation")
	pairwise_similarities = [[0] * len(X_val) for _ in range(len(X_val))]

	# Iterate over each tree in the forest
	for tree in rf_classifier.estimators_:
			# Get the leaf indices for each example
			leaf_indices = tree.apply(X_val)

			# Update pairwise similarities based on leaf indices
			for i in range(len(X_val)):
					for j in range(i + 1, len(X_val)):
							if leaf_indices[i] == leaf_indices[j]:
									pairwise_similarities[i][j] += 1
									pairwise_similarities[j][i] += 1

	# Normalize the pairwise similarities by the number of trees
	normalized_pairwise_similarities_val = [[similarity / num_trees for similarity in row] for row in pairwise_similarities]
	if plot:
		plt.figure()
		plt.imshow(np.array(normalized_pairwise_similarities_val),cmap="jet")
		plt.colorbar()
		plt.clim(0,1)
		plt.xticks([np.where(y_val == 0)[0][-1]], "-")
		plt.yticks([np.where(y_val == 0)[0][-1]], "-")
	# Compute the confusion matrix
	y_pred = rf_classifier.predict(X_val)
	conf_matrix = confusion_matrix(y_val, y_pred)
	print('Confusion Matrix:')
	print(conf_matrix)



	# ==============================
	print("Testing")
	# Initialize a matrix to store pairwise similarities
	pairwise_similarities = [[0] * len(X_test) for _ in range(len(X_test))]

	# Iterate over each tree in the forest
	for tree in rf_classifier.estimators_:
			# Get the leaf indices for each example
			leaf_indices = tree.apply(X_test)

			# Update pairwise similarities based on leaf indices
			for i in range(len(X_test)):
					for j in range(i + 1, len(X_test)):
							if leaf_indices[i] == leaf_indices[j]:
									pairwise_similarities[i][j] += 1
									pairwise_similarities[j][i] += 1

	# Normalize the pairwise similarities by the number of trees
	normalized_pairwise_similarities = [[similarity / num_trees for similarity in row] for row in pairwise_similarities]

	# Display the normalized pairwise similarities
	# print("Normalized Pairwise Similarities:")
	# for i, row in enumerate(normalized_pairwise_similarities):
	#     print(f"Example {i}: {row}")
	normalized_pairwise_similarities_test = np.array(normalized_pairwise_similarities)
	if plot:
		plt.figure()
		plt.imshow(np.array(normalized_pairwise_similarities_test),cmap="jet")
		plt.colorbar()
		plt.clim(0,1)
		plt.xticks([np.where(y_test == 0)[0][-1]], "-")
		plt.yticks([np.where(y_test == 0)[0][-1]], "-")
		y_pred = rf_classifier.predict(X_test)
	# Compute the confusion matrix
	conf_matrix = confusion_matrix(y_test, y_pred)
	print('Confusion Matrix:')
	print(conf_matrix)
	return normalized_pairwise_similarities_all, normalized_pairwise_similarities_train,normalized_pairwise_similarities_val, normalized_pairwise_similarities_test, (Z_train,Z_val,Z_test)




import numpy as np
from scipy.spatial.distance import cdist

def k_nearest_values(array, index, k):

    # Calculate the absolute differences between each element and the element at the given index
    differences = np.abs(array - array[index])

    # Sort the indices based on the absolute differences
    sorted_indices = np.argsort(differences)

    # Select the k nearest values
    k_nearest_indices = sorted_indices[1:k+1]
    k_nearest_values = np.mean(array[k_nearest_indices])
    # k_nearest_values = array[index]/np.sum(array[k_nearest_indices]+1e-3)

    return k_nearest_values

def k_farest_values(array, index, k):

    # Calculate the absolute differences between each element and the element at the given index
    differences = np.abs(array - array[index])

    # Sort the indices based on the absolute differences
    sorted_indices = np.argsort(differences)

    # Select the k nearest values
    k_nearest_indices = sorted_indices[-k:]
    k_nearest_values = np.mean(array[k_nearest_indices])

    return k_nearest_values

def create_nearest_neighbor_similarity_matrix(similarity_matrix, k=5):
    similarity_matrix = np.array(similarity_matrix)
    num_nodes = similarity_matrix.shape[0]

    # Initialize the new similarity matrix
    new_similarity_matrix = np.zeros_like(similarity_matrix)

    for i in range(num_nodes):
      for j in range(i+1,num_nodes):
          new_similarity_matrix[i,j] = k_nearest_values(similarity_matrix[i,:], j,5)
          new_similarity_matrix[j,i] = new_similarity_matrix[i,j]
    return new_similarity_matrix


def create_farest_neighbor_similarity_matrix(similarity_matrix, k=5):
    similarity_matrix = np.array(similarity_matrix)
    num_nodes = similarity_matrix.shape[0]

    # Initialize the new similarity matrix
    new_similarity_matrix = np.zeros_like(similarity_matrix)

    for i in range(num_nodes):
      for j in range(i+1,num_nodes):
          new_similarity_matrix[i,j] = k_farest_values(similarity_matrix[i,:], j,5)
          new_similarity_matrix[j,i] = new_similarity_matrix[i,j]
    return new_similarity_matrix

from itertools import combinations, permutations,combinations_with_replacement
from tqdm import tqdm
def wise_multiply(matrix_list):

    # Check if all matrices have the same shape
    shapes = [np.array(matrix).shape for matrix in matrix_list]
    # Perform element-wise multiplication
    result = np.ones(shapes[0])
    for matrix in matrix_list:
        result *= matrix
        # result = chi2_kernel(result,matrix)

    return result
def wise_add(matrix_list):

    # Check if all matrices have the same shape
    shapes = [np.array(matrix).shape for matrix in matrix_list]
    # Perform element-wise multiplication
    result = np.ones(shapes[0])
    for matrix in matrix_list:
        result += matrix

    return result
def get_kernel_combinations(values,max_interaction=2):
    count = 0
    K_list = []

    for r in range(1, max_interaction+1):
        # Generate combinations of length r
        # combinations_r = combinations(values, r)
        combinations_r = combinations_with_replacement(values, r)

        # combinations_with_replacement

        # Print each combination
        for combination in combinations_r:
            K_i = wise_multiply(combination)
            K_list.append(K_i)
            # print(K_i.shape)
            count += 1
    print(count)
    # K_list.append(wise_add(values))
    return K_list

def update_kernel_list(kernel_list, k_list=[2,5,7]):
  kernel_list_ = []
  for knn in k_list:
    for i,k in enumerate((kernel_list)):
      kernel_list_.append(create_nearest_neighbor_similarity_matrix(k, k=knn))
      # kernel_list_.append(create_farest_neighbor_similarity_matrix(k, k=knn))

  kernel_list.extend(kernel_list_)
  return kernel_list


def load_data(dataPath, input=True):
    if input:
        data = pd.read_csv(dataPath,header=None)
        data = data.values
        return data

    else:
        data = pd.read_csv(dataPath)    
        data = data['encoded']
        le = LabelEncoder()
        y = le.fit_transform(data)
        return y
    


def classify_kernel(outputs,y_train, PET_Z, verbose=False):

  umap = UMAP(n_components=10, random_state=42)

  X_umap = umap.fit_transform(outputs)
  y = y_train[0,0,:]
  Z_train,Z_val, Z_test = PET_Z[0], PET_Z[1],PET_Z[2]
  Z_train = np.concatenate([Z_train, Z_val], axis=0)
  X_train, X_test, y_train, y_test = X_umap[Z_train,:], X_umap[Z_test,:], y[Z_train], y[Z_test]


  # X_train, X_test, y_train, y_test =  train_test_split(X_umap,y, test_size=0.2, random_state=41)
  # Create a Random Forest classifier
  rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

  # Fit the model to the training data
  rf_classifier.fit(X_train, y_train)

  # Get the number of trees in the forest
  num_trees = len(rf_classifier.estimators_)



  # =================
  # Make predictions on the training data
  y_pred = rf_classifier.predict(X_train)

  # Evaluate the model
  accuracy = accuracy_score(y_train, y_pred)
  if verbose:
    print(f"Accuracy train: {accuracy:.2f}")

    # Display the classification report
    print("Classification Report train:")
    print(classification_report(y_train, y_pred))


  # ==========================
  # Make predictions on the testing data
  y_pred = rf_classifier.predict(X_test)

  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  if verbose:
    print(f"Accuracy test: {accuracy:.2f}")

    # Display the classification report
    print("Classification Report test:")
    print(classification_report(y_test, y_pred))
  return accuracy


def normalize_kernel(K):
    K_min = np.min(K)
    K_max = np.max(K)
    return (K - K_min) / (K_max - K_min + 1e-8)