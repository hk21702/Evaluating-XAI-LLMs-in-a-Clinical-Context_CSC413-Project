import torch
import pandas as pd
from torch.utils.data import Dataset
import ast

class MimicDataset(Dataset):
    """Mimic Dataset - a dataset of diagnostic reports and their corresponding icd9 labels"""

    def __init__(self, csv_file_code, csv_file_labels, tokenizer):
        """
        Arguments:
            csv_file (string): Path to the csv file with data and annotations
        """
        self.mimic_dataset = pd.read_csv(csv_file_code)
        
        # create a dictionary of icd codes and their corresponding index in the list
        self.icd_labels = pd.read_csv(csv_file_labels)
        self.icd_labels = self.icd_labels["icd_code"].tolist()
        self.icd_labels_dict = {self.icd_labels[i]: i for i in range(len(self.icd_labels))}
        
        self.icd_size = len(self.icd_labels)
        
        self.tokenizer = tokenizer
        self.tokenizer_max_length = 512
    

    def __len__(self):
        return len(self.mimic_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
          
        # get the text and icd codes from the dataset
        print(self.mimic_dataset.iloc[idx])
        text = self.mimic_dataset.iloc[idx, 4]
        # using the icd_proc column, let me know if this is incorrect
        icd_codes = ast.literal_eval(self.mimic_dataset.iloc[idx, 6])
        
        # tokenize the text
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt")
        
        # create the label tensor
        label_tensor = torch.zeros(self.icd_size)
        for code in icd_codes:
            label_tensor[self.icd_labels_dict[code]] = 1
            
        return inputs, label_tensor
        