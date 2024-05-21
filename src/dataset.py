import math

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Character dataset Class for Pytorch loader
    """

    def __init__(self, data, block_size):
        
        self.block_size = block_size        # Max seq length                   
        self.data = data                    # Dataset
        
        chars = sorted(list(set(data)))     # Set of all unique characterrs
        self.vocab_size = len(chars)        
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }   # string 2 index dict
        self.itos = { i:ch for i,ch in enumerate(chars) }   # index 2 string dict
        
        print(f"The data has {self.vocab_size} unique characters with total length : {len(data)} ")
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):

        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)    # Upto second last
        y = torch.tensor(dix[1:], dtype=torch.long)     # One shifted right
        
        return x, y

