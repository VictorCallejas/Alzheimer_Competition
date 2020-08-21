import argparse

import pandas as pd
import numpy as np

import cv2

import os

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def main(args):

    metadata = pd.read_csv(args.metadata_path).filename
    list_videos = os.listdir(args.videos_path)

    files = metadata[metadata.isin(list_videos)].values

    features = []

    # HACER EN PARALELO
    # OPC PADDING
    with tqdm(total=len(files)) as pbar:
        for f in files:

            path = args.videos_path + f
            video = cv2.VideoCapture(path)

            tmp = f, normalize_video(video,args)

            features.append(tmp)
            pbar.update(1)

    np.savez(args.features_path,**{i:f for i, f in features})
    save_gif_example(args, files[0])


def normalize_video(video,args):

    result = []

    success, image = video.read()
    frames = []
    while(success):
        frames.append(image)
        success,image = video.read()
    
    n = args.n_frames - len(frames)
    if(args.n_frames != 0):
        if (n<0):
            idx = int(n/2)
            if (n%2 == 0):
                frames = frames[-idx:idx]
            else:
                frames = frames[-idx:idx-1]

    
    hsv = cv2.cvtColor(frames[0],cv2.COLOR_BGR2HSV)
    contour = cv2.inRange(hsv,args.lower_orange,args.upper_orange)

    x,y,w,h = cv2.boundingRect(contour)

    if args.padding > 0:
        x = x 
        y = y
        w = w + (args.padding*2)
        h = h + (args.padding*2)
        fondo = np.zeros((contour.shape[0]+(args.padding*2),contour.shape[1]+(args.padding*2)),dtype=np.uint8)
        fondo[args.padding:contour.shape[0]+args.padding,args.padding:contour.shape[1]+args.padding]+=contour
        contour = fondo

    for frame in frames:

        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        if args.padding > 0:
            fondo = np.zeros((frame.shape[0]+(args.padding*2),frame.shape[1]+(args.padding*2),3),dtype=np.uint8)
            fondo[args.padding:frame.shape[0]+args.padding,args.padding:frame.shape[1]+args.padding]+=frame
            frame = fondo

        croped_img = frame[y:y+h,x:x+w]
        croped_mask = contour[y:y+h,x:x+w]

        if args.mode == 'resize':
            resized_gray_img = cv2.resize(croped_img,(args.dim,args.dim))
            resized_mask = cv2.resize(croped_mask,(args.dim,args.dim))
            newH, newW = args.dim, args.dim
        elif args.mode == 'fit':
            scaleWidth = float(args.dim) / float(w)
            scaleHeight = float(args.dim) / float(h)

            if (scaleHeight > scaleWidth):
                imScale = scaleWidth
            else:
                imScale = scaleHeight

            newW, newH = int(w * imScale), int(h * imScale)
            h1 = 0
            h2 = newH
            w1 = 0
            w2 = newW

            fondo = np.zeros((args.dim,args.dim,3),dtype=np.uint8)
            fondo[h1:h2,w1:w2]+=cv2.resize(croped_img,(newW,newH))
            resized_gray_img = fondo

            fondo = np.zeros((args.dim,args.dim),dtype=np.uint8)
            fondo[h1:h2,w1:w2]+=cv2.resize(croped_mask,(newW,newH))
            resized_mask = fondo

        if args.contour_expansion:

            for row in range(0,newH):
                i = 0
                is_contour = False

                while((i < newW)&(not is_contour)):
                    if(resized_mask[row,i]):
                        is_contour = True
                        try:
                            resized_gray_img[row,i+1] = 0
                            resized_gray_img[row,i+2] = 0
                            resized_gray_img[row,i+3] = 0
                        except IndexError:
                            pass
                        try:
                            resized_gray_img[row,i-1] = 0
                            resized_gray_img[row,i-2] = 0
                            resized_gray_img[row,i-3] = 0
                        except IndexError:
                            pass
                        try:
                            resized_gray_img[row-1,i] = 0
                            resized_gray_img[row-2,i] = 0
                            resized_gray_img[row-3,i] = 0
                        except IndexError:
                            pass
                        try:
                            resized_gray_img[row+1,i] = 0
                            resized_gray_img[row+2,i] = 0
                            resized_gray_img[row+3,i] = 0
                        except IndexError:
                            pass
                    resized_gray_img[row,i] = 0
                    i+=1

                if not is_contour: continue
                is_contour = False

                i = newW - 1

                while((i >= 0)&(not is_contour)):
                    if(resized_mask[row,i]):
                        is_contour = True
                        try:
                            resized_gray_img[row,i+1] = 0
                            resized_gray_img[row,i+2] = 0
                            resized_gray_img[row,i+3] = 0
                        except IndexError:
                            pass
                        try:
                            resized_gray_img[row,i-1] = 0
                            resized_gray_img[row,i-2] = 0
                            resized_gray_img[row,i-3] = 0
                        except IndexError:
                            pass
                        try:
                            resized_gray_img[row-1,i] = 0
                            resized_gray_img[row-2,i] = 0
                            resized_gray_img[row-3,i] = 0
                        except IndexError:
                            pass
                        try:
                            resized_gray_img[row+1,i] = 0
                            resized_gray_img[row+2,i] = 0
                            resized_gray_img[row+3,i] = 0
                        except IndexError:
                            pass
                    resized_gray_img[row,i] = 0
                    i-=1

        img = resized_gray_img

        if args.black_contour:
            mask_inv = cv2.bitwise_not(resized_mask)      
            img = cv2.bitwise_and(img,img,mask = mask_inv)

        if args.bn:
            img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
            
        result.append(img)

    if(args.n_frames != 0):
        if (n > 0):
            result = np.concatenate((result,np.zeros((n,args.dim,args.dim,3),np.uint8)),axis=0)
        
    result = np.array(result).astype(np.uint8)
    #assert result.shape == (args.n_frames,args.dim,args.dim)
    return result

def save_gif_example(args, file):

    features = np.load(args.features_path)
    img = features[str(file)]

    fig = plt.figure()

    ims = []
    for frame in img:
        ims.append([plt.imshow(frame)])

    ani = animation.ArtistAnimation(fig, ims, interval=100,blit=True, repeat_delay=1000)

    ani.save('../data/processed/run_example.gif',writer=PillowWriter(fps=10))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_path", type=str, default="../data/raw/train/train_metadata.csv")
    parser.add_argument("--videos_path", type=str, default="../data/raw/train/nano/")
    parser.add_argument("--features_path", type=str, default='../data/processed/train_ff.npz')
    parser.add_argument("--mode",type=str,default='fit',choices=['resize','fit'])
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--contour_expansion", type=bool, default=False)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--black_contour", type=bool, default=False)
    parser.add_argument("--lower_orange",type=np.array,default=np.array([0,80,50],dtype=np.uint8))
    parser.add_argument("--upper_orange",type=np.array,default=np.array([255,255,255],dtype=np.uint8))
    args = parser.parse_args()

    print(args)

    main(args)

