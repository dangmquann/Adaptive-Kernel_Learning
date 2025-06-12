import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, data_config):
        self.data_config = data_config
        self.modalities = []
        self.data = {}

        # self.labels = None
        self.load_data()

    def get_all_subject(self):
        return self.all_subject
    def get_all_subjects_labels(self):
        return self.all_subject, self.labels

    def get_modalities(self):
        return self.modalities
    def load_data(self):
        """
        Load all datasets specified in the data_config dictionary.

        Raises:
            FileNotFoundError: If any of the specified files do not exist.
        """
        for key, file_path in self.data_config.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            if key == "label":
                data = pd.read_csv(file_path)
                le = LabelEncoder()
                y = le.fit_transform(data['encoded'].values)
                self.data[key] = y
                self.all_subject = data['Subject ID'].values
                self.labels = y
            else:
                data = pd.read_csv(file_path)
                self.data[key] = data.iloc[:, 1:].values # skip the first column (Subject ID)
                self.modalities.append(key)

    def get_data(self, key):
        """
        Retrieve a specific dataset by key.

        Args:
            key (str): The key of the dataset to retrieve.

        Returns:
            pd.DataFrame: The requested dataset.

        Raises:
            KeyError: If the specified key is not found in the loaded data.
        """
        if key not in self.data:
            raise KeyError(f"Data for key '{key}' has not been loaded.")
        return self.data[key]

