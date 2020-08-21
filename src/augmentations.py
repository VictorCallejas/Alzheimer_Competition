import pandas as pd  
import numpy as np 

from scipy import ndimage

from tqdm import tqdm

import matplotlib.pyplot as plt

import argparse

import os

def main(args):

    train_labels = pd.read_csv(args.train_labels_path)
    list_videos = os.listdir(args.videos_path)

    train_labels = train_labels[train_labels.filename.isin(list_videos)]

    X = np.load(args.features_path)
    features = []

    df = pd.DataFrame(columns=['filename','stalled'])
    
    filenames = []
    stalled = []

    # HACER EN PARALELO
    for _, row in tqdm(train_labels.iterrows(),total=train_labels.shape[0]):

        f = row[0]
        s = row[1]
        
        # ORIGINAL
        feature = X[str(f)]
        f = f.replace('.mp4','')
        features.append(feature)
        filenames.append(f+'.mp4') 
        stalled.append(s)


        # ROTATIONS
        features.append(ndimage.rotate(feature,90,axes=(1,2)))
        filenames.append(f+'_r90'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.rotate(feature,180,axes=(1,2)))
        filenames.append(f+'_r180'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.rotate(feature,270,axes=(1,2)))
        filenames.append(f+'_r270'+'.mp4') 
        stalled.append(s)


        features.append(ndimage.rotate(feature,45,axes=(1,2),reshape=False))
        filenames.append(f+'_r45'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.rotate(feature,135,axes=(1,2),reshape=False))
        filenames.append(f+'_r135'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.rotate(feature,225,axes=(1,2),reshape=False))
        filenames.append(f+'_r225'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.rotate(feature,315,axes=(1,2),reshape=False))
        filenames.append(f+'_r315'+'.mp4') 
        stalled.append(s)

        # SHIFTED
        shift = 20
        features.append(ndimage.shift(feature,shift=(0,shift,shift,0)))
        filenames.append(f+'_s25u25r'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.shift(feature,shift=(0,-shift,-shift,0)))
        filenames.append(f+'_s25d25l'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.shift(feature,shift=(0,-shift,shift,0)))
        filenames.append(f+'_s25d25r'+'.mp4') 
        stalled.append(s)

        features.append(ndimage.shift(feature,shift=(0,shift,-shift,0)))
        filenames.append(f+'_s25u25l'+'.mp4') 
        stalled.append(s)


    np.savez(args.save_path_features,**{i:f for i, f in zip(filenames,features)})

    df.filename = filenames
    df.stalled = stalled
    df.to_csv(args.save_path_df,index=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_labels_path", type=str, default="../data/raw/train/train_labels.csv")
    parser.add_argument("--videos_path", type=str, default="../data/raw/train/nano/")
    parser.add_argument("--features_path", type=str, default='../data/processed/train_final.npz')
    parser.add_argument("--save_path_features",type=str,default='../data/processed/train_final_aug.npz')
    parser.add_argument("--save_path_df",type=str,default='../data/processed/train_aug_df.csv')

    args = parser.parse_args()

    print(args)
    
    main(args)