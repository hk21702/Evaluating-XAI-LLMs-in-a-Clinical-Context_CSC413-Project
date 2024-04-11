import torch
import pandas as pd
from torch.utils.data import Dataset
import ast


class MimicDataset(Dataset):
    """Mimic Dataset - a dataset of diagnostic reports and their corresponding icd9 labels"""

    def __init__(
        self,
        text,
        labels,
        classes: list,
        class2id: dict | None = None,
        id2class: dict | None = None,
    ):
        """
        Arguments:
            csv_file (string): Path to the csv file with data and annotations
        """
        self.text = text

        # create a dictionary of icd codes and their corresponding index in the list
        self.labels = labels
        self.icd_labels = classes

        self.class2id = (
            {class_: id for id, class_ in enumerate(classes)}
            if class2id is None
            else class2id
        )

        self.id2class = (
            {id: class_ for class_, id in self.class2id.items()}
            if id2class is None
            else id2class
        )

        self.icd_size = len(self.icd_labels)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get the tokenized data
        item = self.text[idx]

        icd_codes = ast.literal_eval(self.labels[idx])

        # create the label tensor
        label_tensor = torch.zeros(self.icd_size)
        for code in icd_codes:
            label_tensor[self.class2id[code]] = 1.0

        item["labels"] = label_tensor

        # squeeze item to remove the extra dimension
        item["input_ids"] = item["input_ids"].squeeze()
        item["attention_mask"] = item["attention_mask"].squeeze()

        return item
