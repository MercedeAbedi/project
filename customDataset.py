import torch
from torch.utils.data.dataset import Dataset
import pandas as pd

class bostonDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.data = self.data.values
        self.x = torch.Tensor(self.data[:, :11]).float()
        self.y = torch.Tensor(self.data[:,  11]).float()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.x[idx, :], self.y[idx])
        return sample
