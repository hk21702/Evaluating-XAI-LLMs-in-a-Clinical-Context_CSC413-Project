import torch
import pandas as pd
from torch.utils.data import Dataset

class MimicDataset(Dataset):
    """Mimic Dataset - a dataset of diagnostic reports and their corresponding icd9 labels"""

    def __init__(self, csv_file_code, csv_file_labels):
        """
        Arguments:
            csv_file (string): Path to the csv file with data and annotations
        """
        self.mimic_dataset = pd.read_csv(csv_file_code)
        # self.full_icd_labels = pd.read_csv(csv_file_labels)

    def __len__(self):
        return len(self.mimic_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = torch.to_tensor(self.mimic_dataset.iloc[idx, 4])
        icd = torch.to_tensor(self.mimic_dataset.iloc[idx, 7])
        
        return text, icd