import argparse

import pandas as pd 
import numpy as np 

import torch
import torch.nn as nn

import os

from tqdm import tqdm

import math


def main(args):

    test_metadata = pd.read_csv(args.test_metadata_path)
    test_features = np.load(args.test_videos_npz_path)

    filenames = test_metadata.filename.values
    labels = []

    size = test_metadata.shape[0]
    rango = 4000

    model = torch.load(args.model_path)
    print(model)

    device = torch.device(args.device)
    model.to(device)

    model.eval()

    with tqdm(total=math.ceil(size/rango)) as pbar:
        for idx in range(0,size,rango):

            df = test_metadata[idx:idx+rango]
            test_dataloader = get_dataloader(df,test_features)

            logits = []
            
            for step, batch in enumerate(tqdm(test_dataloader)):
                inputs = batch[0].to(device)
                with torch.no_grad():
                    b_logits = model(inputs)
                    
                logits.extend(b_logits)
            
            y_probas = nn.Sigmoid()(torch.tensor(logits))
            y_labels = y_probas.round().detach().numpy().astype(np.int)
            labels.extend(y_labels)

            pbar.update(1)


    subm = pd.DataFrame()
    subm['filename'] = filenames
    subm['stalled'] = labels

    subm.to_csv(args.submission_path,index=False)

    print('SUBMISSION DONE')



def get_dataloader(test_metadata, test_features):

    features = []

    for _, row in test_metadata.iterrows():
            feature = np.array(test_features[str(row.filename)]) / 255
            features.append(torch.tensor(np.sum(feature,axis=0)/100,dtype=torch.float))

    test_dataset = torch.utils.data.TensorDataset(torch.stack(features,dim=0).unsqueeze(1))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
        batch_size=args.batch_size
    )

    return test_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_metadata_path", type=str, default="../data/raw/test/test_metadata.csv")
    parser.add_argument("--test_videos_path",type=str,default="../data/raw/test/videos/")
    parser.add_argument("--test_videos_npz_path", type=str, default="../data/processed/test_pad_bc.npz")
    parser.add_argument("--submission_path", type=str, default="../submission/submission_145.csv")

    parser.add_argument("--model_path",type=str,default='../models/145_model.pth')

    parser.add_argument("--device",type=str,default='cuda')
    parser.add_argument("--batch_size",type=int,default=200)

    args = parser.parse_args()

    print(args)

    main(args)
    