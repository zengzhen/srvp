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

def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def sigmoid(x):
    k = 1.0
    shift = 1.0
    return 1.0 / (1.0 + math.exp(-k*(x-shift)))

class MarketHeatMap(VideoDataset):
  '''
  Heatmap fo market dataset.
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
    self.max_perc = 25 # assume maximum percentage in gain/loss

    # self.train_count = 0

  def __getitem__(self, idx):
    if not self.train:
      # When testing, pick the selected video (from the precomputed testing set)
      return self.data[idx]

    # self.data: num_days x num_markets
    if len(self.data) < self.seq_len:
      print("Error: stock history length of bouncing balls should be longder than training seq_len")
      sys.exit(1)

    # Random timestep for the beginning of the video
    t0 = np.random.randint(len(self.data) - self.seq_len + 1)
    # Extract the sequence from stock_hist files
    n_markets = self.data[0].shape[0]
    images = np.zeros([self.seq_len, self.frame_size, self.frame_size, 3], np.uint8)
    for t in range(self.seq_len):
      for id in range(n_markets):
        # 9 markets visualzied in 3x3 grid
        tile_w = 21
        tile_h = 21
        perc = self.data[t0+t][id]
        x1, y1 = (int(math.floor(id/3.0)) * tile_h, id%3 * tile_w)
        x2, y2 = (x1+tile_h-2, y1+tile_w-2)
        # sanity check
        if perc > self.max_perc:
            print("Error: perc in gain exceed assumed max_perc")
            sys.exit(1)
        # draw tiles
        if perc >= 0: # green
            images[t] = cv2.rectangle(images[t], (x1, y1), (x2, y2), (0, 255*sigmoid(perc), 0), -1)
        else: # red
            images[t] = cv2.rectangle(images[t], (x1, y1), (x2, y2), (0, 0, 255*sigmoid(abs(perc))), -1)

    # video_file = '/home/ubuntu/workspace/srvp/results/market_heatmap/train_%04d.mp4' % (self.train_count)
    # print(video_file)
    # out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.frame_size, self.frame_size), isColor=True)

    # for i in range(len(images)):
    #     out.write(images[i])
    # out.release()
    # self.train_count += 1

    # if self.train_count == 10:
    #     sys.exit(1)

    return images

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
      # read in all CSV files: 06.29.2010 - 12.31.2018
      data = []
      for index, market in enumerate(list_files(data_dir, 'csv')):
          print(market)
          market_data = np.genfromtxt(os.path.join(data_dir, market), delimiter=',', skip_header=1)
          # column 4 - close values
          market_data = market_data[:,4]
          # calcualte gain/loss percentage based on close values
          market_data = (market_data[1:] - market_data[:-1])/market_data[:-1]*100
          if index==0:
              data = market_data
          else:
              data = np.column_stack((data, market_data))
    else:
      dataset = np.load(os.path.join(data_dir, 'test_market_heatmap.npz'),
                              allow_pickle=True)

      sequences = dataset['sequences']
      data = [sequences[:, i] for i in range(sequences.shape[1])]

    return cls(data, nx, seq_len, train)

