import pandas as pd 
import numpy as np 

import os

import torch

from sklearn.model_selection import train_test_split

from tqdm import tqdm


class AlzDataset(torch.utils.data.Dataset):

    def __init__(self,X, y):
        self.files = X
        self.labels = y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = (self.files.iloc[idx], self.labels.iloc[idx])
        return sample


def get_dataloaders(args):

    train_labels = pd.read_csv(args.train_labels_path)

    list_videos = os.listdir(args.train_videos_path)

    train_labels = train_labels[train_labels.filename.isin(list_videos)]


    X_train, X_valid, y_train, y_valid = train_test_split(train_labels.filename,train_labels.stalled,stratify=train_labels.stalled,shuffle=True,test_size=.2)

    train_dataset = AlzDataset(X_train,y_train)
    valid_dataset = AlzDataset(X_valid,y_valid)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset),
        batch_size=args.train_batch_size
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=torch.utils.data.SequentialSampler(valid_dataset),
        batch_size=args.valid_batch_size
    )

    return train_dataloader, valid_dataloader