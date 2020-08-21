import argparse

import pandas as pd 
import numpy as np

from data import get_dataloaders
from model import R2plus1D_18, CNN_LSTM, CNN_3D, SUM_2D

import torch
import torch.nn as nn

from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

import warnings

def main(args):

    train_dataloader, dev_dataloader = get_dataloaders(args)
    print('Dataloaders obtained')

    features = np.load(args.train_videos_npz_path)
    print('Features obtained')

    device = torch.device(args.device)
    print('Device: ', device)

    model =  R2plus1D_18()
    print('MODEL\n',model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = args.learning_rate,
        eps = args.epsilon
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    print('TRAINING...')
    training_stats = []
    steps_stats = []

    for epoch_i in tqdm(range(0,args.epochs)):

        avg_train_loss, train_acc, train_mcc = train(model,train_dataloader,features,device,criterion,optimizer)
        print(avg_train_loss, train_acc, train_mcc)
        avg_dev_loss, dev_acc, dev_mcc = valid(model,dev_dataloader,features,device,criterion,scheduler)
        print(avg_dev_loss, dev_acc, dev_mcc)
        training_stats.append([avg_train_loss,train_acc,train_mcc,avg_dev_loss,dev_acc,dev_mcc])

        torch.save(model,str(args.save_model_path+str(epoch_i)+'_model.pth'))


    print('TRAINING COMPLETED')

    # Show training results
    col_names = ['train_loss','train_acc','train_mcc','dev_loss', 'dev_acc','dev_mcc']
    training_stats = pd.DataFrame(training_stats,columns=col_names)
    print(training_stats.head(args.epochs))


def train(model, train_dataloader,features,device,criterion,optimizer):
    total_train_loss = 0
    model.train()

    logits = []
    ground_truth = []
    
    for step, batch in enumerate(tqdm(train_dataloader)):

        model.zero_grad()

        batch = preprocess_input(batch,features)

        inputs = batch[0].to(device)
        targets = batch[1]

        b_logits = model(inputs)

        logits.extend(b_logits)
        ground_truth.extend(targets)
                
        loss = criterion(torch.squeeze(b_logits),targets.float())
        total_train_loss += loss.item()

        loss.backward()
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    y_probas = nn.Sigmoid()(torch.tensor(logits))
    y_labels = y_probas.round()

    train_mcc = metrics.matthews_corrcoef(ground_truth,y_labels)
    train_acc = metrics.accuracy_score(ground_truth,y_labels)
    avg_train_loss = total_train_loss/len(train_dataloader)

    return avg_train_loss, train_acc, train_mcc

def valid(model,dev_dataloader,features,device,criterion,scheduler):

    model.eval()

    total_dev_loss = 0

    logits = []
    ground_truth = []
            
    for step, batch in enumerate(dev_dataloader):

        batch = preprocess_input(batch,features)

        inputs = batch[0].to(device)
        targets = batch[1]

        with torch.no_grad():
            b_logits = model(inputs)

        logits.extend(b_logits)
        ground_truth.extend(targets)

        loss = criterion(torch.squeeze(b_logits),targets.float())
        total_dev_loss += loss.item()

    y_probas = nn.Sigmoid()(torch.Tensor(logits))
    y_labels = y_probas.round()

    dev_mcc = metrics.matthews_corrcoef(ground_truth,y_labels)
    dev_acc = metrics.accuracy_score(ground_truth,y_labels)
    avg_dev_loss = total_dev_loss/len(dev_dataloader)

    #scheduler.step(avg_dev_loss)

    return avg_dev_loss, dev_acc, dev_mcc

def preprocess_input(batch,features):

    inputs = []
    targets = []

    files, targets =  batch
    for file,target in zip(files,targets):
        feat = np.array(features[str(file)])
        #im_2d = np.sum(feat,axis=0)/feat.shape[0]
        #plt.imshow(im_2d)
        #plt.savefig('gg.png')
        inputs.append(torch.tensor(feat,dtype=torch.float))

    inputs = torch.stack(inputs,dim=0).transpose(1,4)

    return (inputs, targets)



if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_labels_path", type=str, default="../data/raw/train/train_labels.csv")
    parser.add_argument("--train_metadata_path", type=str, default="../data/raw/train/train_metadata.csv")
    parser.add_argument("--train_videos_path", type=str, default="../data/raw/train/nano/")
    parser.add_argument("--train_videos_npz_path", type=str, default="../data/processed/train_ff.npz")

    parser.add_argument("--train_batch_size",type=int,default=5)
    parser.add_argument("--valid_batch_size",type=int,default=5)

    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--learning_rate",type=float,default=2e-3)
    parser.add_argument("--epsilon",type=float,default=2e-8)

    parser.add_argument("--save_model_path",type=str,default='../models/')

    args = parser.parse_args()

    print(args)

    main(args)