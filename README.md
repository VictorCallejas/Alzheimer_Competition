# Alzheimer_Competition
### Driven data Clog Loss: Advance Alzheimer’s Research with Stall Catchers

https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/207/

### Leaderboard position: 35 of 922

## Overview
5.8 million Americans live with Alzheimer’s dementia, including 10% of all seniors 65 and older. Scientists at Cornell have discovered links between “stalls,” or clogged blood vessels in the brain, and Alzheimer’s. Stalls can reduce overall blood flow in the brain by 30%. The ability to prevent or remove stalls may transform how Alzheimer’s disease is treated.

Stall Catchers is a citizen science project that crowdsources the analysis of Alzheimer’s disease research data provided by Cornell University’s Department of Biomedical Engineering. It resolves a pressing analytic bottleneck: for each hour of data collection it would take an entire week to analyze the results in the lab, which means an entire experimental dataset would take 6-12 months to analyze. Today, the Stall Catchers players are collectively analyzing data 5x faster than the lab while exceeding data quality requirements.

The research team has realized there are aspects of this task that are best suited to uniquely human cognitive faculties as explained in this blog post. However, some portion of the data, the “low-hanging fruit,” may be within reach of machine learning models that are able to distinguish between easy and difficult data and are applied only in cases where they have been validated to meet the researchers’ data quality requirements. If a machine learning classifier could be used for 50% of the data, it would double the analytic throughput of Stall Catchers and could achieve the original goal of analyzing the data 10x faster than the lab. This could ultimately put finding an Alzheimer’s treatment target in reach within the next year or two.

Through the Stall Catchers project, there is now a one-of-a-kind dataset that can be used to train and evaluate ML models on this task. Each exemplar is an image stack (a 3D image) taken from a live mouse brain showing blood vessels and blood flow. Each stack has an outline drawn around a target vessel segment and has been converted to an mp4 video file. The objective of this competition is to classify the outlined vessel segment as flowing—if blood is moving through the vessel—or stalled if the vessel has no blood flow.

## Problem description
Your goal is to predict whether the outlined vessel segment in each video is stalled or not. A value of 1 indicates the vessel is stalled and has no blood flow. A value of 0 indicates the vessel is flowing and has blood moving through the vessel.

Features
Videos
Labels
Performance metric
Matthew's correlation coefficient
Submission Format
Format example

The features in this dataset
Videos
The main features in this challenge are the videos themselves! These are image stacks taken from live mouse brains showing blood vessels and blood flow. Each video is identified by its filename, which is a numeric string followed by ".mp4", e.g., 100000.mp4.

The target vessel segment for each video is outlined in orange. See the Stall Catcher's Tutorial for more detail on what a stall looks like. It's important to note that the z axis in these videos represents both depth—looking at successive layers of brain tissue—and time. A step downward in z is also a step forward in time. See how this manifests for a diagonal vessel, planar vessel, horseshoe-shaped vessel, and C-shaped vessel, as a few examples.

All videos are hosted in a public s3 bucket called drivendata-competition-clog-loss.

The full training dataset contains over 570,000 videos, which is around 1.4 terabytes! To help facilitate faster model prototyping, we've created two subsets of the dataset, referred to as "nano" and "micro." See the table below for details about each version. Note that the nano and micro subsets have been designed to have much more balanced classes than the full dataset.

Training Dataset Version	Size	Class ratio
(flowing / stalled)
Nano	3.8 GB	50 / 50
Micro	6.4 GB	70 / 30
Full	1.4 TB	99.7 / 0.3
You can use the nano or micro boolean column in the train_metadata.csv file to select samples in the desired subset. In addition, tar archives of the the videos in the nano and micro subsets are available on the data download page. When working with these subsets, use the filename column to subset the train_labels.csv file accordingly.

Metadata
In addition to the videos, you are provided with the following metadata. Each row corresponds to one video:

url - file location of the video in the public s3 bucket drivendata-competition-clog-loss
project_id - unique identifier for the research project that generated the video
num_frames - number of frames in the video
crowd_score - crowd-labeled probability that the video is stalled, ranging between 0 (flowing) and 1 (stalled)
tier1 - boolean variable indicating a highly confident label
micro - boolean variable indicating if the video is part of the micro subset
nano - boolean variable indicating if the video is part of the nano subset
For the test metadata, you are only provided with url and num_frames.

TIER 1 DATA
The tier1 column indicates the highest quality data. These are videos that either have an expert-validated label or a highly confident crowd label. We define a highly confident crowd label as one with a crowd_score equal to 0 (absolutely flowing) or greater than or equal to 0.75 (most likely stalled).

Videos in tier1 will be the most reliable examples of stalled or flowing vessels and therefore are a good place to start for model training. However, working with videos where the crowd is less confident may provide additional gains down the line. It's up to you to experiment!

CROWD SCORES
All videos have a crowd_score thanks to Stall Catchers, a citizen science game developed by the Human Computation Institute. You can read more about this on the About page.

A subset of the videos has also been reviewed by an expert, who has labeled videos either stalled or flowing. In train_labels.csv, stalled comes from the expert label where available. In cases where an expert label is not available, stalled is a thresholded crowd score value, where videos with crowd scores greater than or equal to 0.5 are designated to be stalled and those with crowd scores less than 0.5 are designated to be flowing. We provide you with the crowd_score so you are able to experiment with setting different thresholds during training.


Metadata example
For example, a single row in the train metadata, has these values:

100000.mp4
url	s3://drivendata-competition-clog-loss/train/100000.mp4
project_id	M
num_frames	54
crowd_score	0
tier1	True
micro	False
nano	False

Labels
The labels are integers, where 0 corresponds to a flowing vessel and 1 corresponds to a stalled vessel. These are a combination of expert and crowd labels.


Performance metric
Performance is evaluated according to Matthew's correlation coefficient (MCC). This metric takes into account all four components of the confusion matrix: true positives, true negatives, false positives, and false negatives. MCC ranges between +1 and -1. A coefficient of +1 represents a perfect prediction, 0 represents no better than random, and -1 represents complete disagreement between predictions and true values. The competitor that maximizes this metric will top the leaderboard.

In Python, you can easily calculate MCC using sklearn.metrics.matthews_corrcoef.

Submission format
The submission format is a .csv file with two columns: filename and stalled. Each row is a prediction (stalled) for the video (filename). Predictions should be integers, where 0 represents a flowing vessel and 1 represents a stalled vessel. There should not be a decimal point in your predictions: 0 is a valid integer but 0.0 is not.


For example, if you predicted that all vessels were flowing,

filename	stalled
100032.mp4	0
100037.mp4	0
100139.mp4	0
100182.mp4	0
100214.mp4	0
The .csv file that you submit would look like this:

filename,stalled
100032.mp4,0
100037.mp4,0
100139.mp4,0
100182.mp4,0
100214.mp4,0
...
