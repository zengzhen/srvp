from glob import glob
import os
import numpy as np
import random
import torch.utils.data as data
import json
import cv2
import math
import csv

from torchvision import datasets

from data.base import VideoDataset
import sys

from preprocessing.payments import make_test_set

def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def sigmoid(x):
    k = 1.0
    shift = 1.0
    return 1.0 / (1.0 + math.exp(-k*(x-shift)))

class Payments(VideoDataset):
  '''
  Payments dataset.
  '''
  def __init__(self, data, nx, seq_len, train):
    """
    Parameters
    ----------
    data : list
        When testing, list of testing videos represented as uint8 NumPy arrays (length, width, height).
        When training, list of digits shape to use when generating videos from the dataset.
    nx : int
        Width and height of the video frames.
    seq_len : int
        Number of frames to produce.
    train : bool
        Whether to use the training or testing dataset.
    """
    self.data = data
    self.frame_size = nx
    self.seq_len = seq_len
    self.train = train

    self.initial_size = 128
    self.accounts = make_test_set.initializeLocations(self.initial_size, margin=15)

    # self.train_count = 0

  def __getitem__(self, idx):
    if not self.train:
      # When testing, pick the selected video (from the precomputed testing set)
      return self.data[idx]

    # synthesize payment video
    vid_len = 7 * make_test_set.fpd
    self.data = make_test_set.getExample(vid_len, self.initial_size, self.frame_size, self.accounts)

    # Random timestep for the beginning of the video
    t0 = np.random.randint(len(self.data) - self.seq_len + 1)
    # Extract the sequence from synthesized video
    x = self.data[t0:t0+self.seq_len,:,:]

    # video_file = '/home/ubuntu/workspace/srvp/results/payments/train_%04d.mp4' % (self.train_count)
    # print(video_file)
    # out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 10, (self.frame_size, self.frame_size), isColor=False)

    # for i in range(len(x)):
    #     out.write(x[i])
    # out.release()
    # self.train_count += 1

    # if self.train_count == 10:
    #     sys.exit(1)

    return x

  def _filter(self, data):
    return self.__class__(data, self.frame_size, self.seq_len, self.train)

  def __len__(self):
    if self.train:
      # Arbitrary number.
      # The number is a trade-off for max efficiency
      # If too low, it is not good for batch size and multi-threaded dataloader
      # If too high, it is not good for shuffling and sampling
      return 30000
    return len(self.data)

  @classmethod
  def make_dataset(cls, data_dir, nx, seq_len, train):
    if train:
      data = []
    else:
      dataset = np.load(os.path.join(data_dir, 'test_payments.npz'),
                              allow_pickle=True)

      sequences = dataset['sequences']
      data = [sequences[:, i] for i in range(sequences.shape[1])]

    return cls(data, nx, seq_len, train)

