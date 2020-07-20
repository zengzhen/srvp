# build on DDPAE: https://github.com/zengzhen/DDPAE-video-prediction.git

from glob import glob
import os
import numpy as np
import random
import torch.utils.data as data
import json
import cv2

from torchvision import datasets

from data.base import VideoDataset
import sys

class BouncingBalls(VideoDataset):
  '''
  Bouncing balls dataset.
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
    self.scale = self.initial_size / 800
    self.radius = int(60 * self.scale)
    # self.train_count = 0

  def __getitem__(self, idx):
    if not self.train:
      # When testing, pick the selected video (from the precomputed testing set)
      return self.data[idx]
    # When training, pick a random video from the dataset, and extract a random sequence
    # Iterate until the selected video is long enough
    vid_id = np.random.randint(len(self.data))
    traj = self.data[vid_id] # traj size: (n_frames, n_balls, 4)
    vid_len = len(traj)
    if vid_len < self.seq_len:
      print("Error: trajecotry length of bouncing balls should be longder than training seq_len")
      sys.exit(1)

    # Random timestep for the beginning of the video
    t0 = np.random.randint(vid_len - self.seq_len + 1)
    # Extract the sequence from trajectory files
    vid_len, n_balls = traj.shape[:2]
    images = np.zeros([self.seq_len, self.frame_size, self.frame_size], np.uint8)
    for t in range(self.seq_len):
      temp = np.zeros([self.initial_size, self.initial_size], np.float32)
      for bid in range(n_balls):
        # each ball:
        ball = traj[t0+t, bid]
        x, y = int(round(self.scale * ball[0])), int(round(self.scale * ball[1]))
        temp = cv2.circle(temp, (x, y), int(self.radius * ball[3]),
                                 255, -1)
        # In case of overlap, brings back video values to [0, 255]
        temp[temp > 255] = 255
      images[t] = cv2.resize(temp.astype(np.uint8), (self.frame_size, self.frame_size))
    
    # video_file = '/home/ubuntu/workspace/srvp/results/BouncingBalls/train_%04d.mp4' % (self.train_count)
    # print(video_file)
    # out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 10, (self.frame_size, self.frame_size), isColor=False)

    # for i in range(len(images)):
    #     out.write(images[i])
    # out.release()
    # self.train_count += 1

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
      folder = 'balls_n4_t60_ex50000'
      data = np.load(os.path.join(data_dir, folder, 'dataset_info.npy'))
    else:
      dataset = np.load(os.path.join(data_dir, 'test_bouncing_balls.npz'),
                              allow_pickle=True)

      sequences = dataset['sequences']
      data = [sequences[:, i] for i in range(sequences.shape[1])]

    return cls(data, nx, seq_len, train)

