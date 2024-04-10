import torch
import pandas as pd
from torch.utils.data import Dataset
import ast

class MimicDataset(Dataset):
    """Mimic Dataset - a dataset of diagnostic reports and their corresponding icd9 labels"""

    def __init__(self, tokenized_dataset, labels, csv_file_labels):
        """
        Arguments:
            csv_file (string): Path to the csv file with data and annotations
        """
        self.tokenized_dataset = tokenized_dataset
        
        # create a dictionary of icd codes and their corresponding index in the list
        self.labels = labels
        self.icd_labels = csv_file_labels
        self.icd_labels = self.icd_labels["icd_code"].tolist()
        self.icd_labels_dict = {self.icd_labels[i]: i for i in range(len(self.icd_labels))}
        
        self.icd_size = len(self.icd_labels)
    

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
          
        # get the tokenized data
        item = self.tokenized_dataset[idx]

        icd_codes = ast.literal_eval(self.labels[idx])
        
        # create the label tensor
        label_tensor = torch.zeros(self.icd_size)
        for code in icd_codes:
            label_tensor[self.icd_labels_dict[code]] = 1
            
        item["labels"] = label_tensor
        
        # squeeze item to remove the extra dimension
        item["input_ids"] = item["input_ids"].squeeze()
        item["attention_mask"] = item["attention_mask"].squeeze()
        
        return item