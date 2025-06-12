import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from . import DataLoader
class DataLoaderAdvanced(DataLoader):
    def __init__(self, data_config,representation_type):
        super().__init__(data_config)
        self.representation_type = representation_type
        assert (self.representation_type in ["feature", "kernel"] , "representation_type must be either 'feature' or 'kernel'")

    def get_data(self, key):
        pass