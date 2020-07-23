import os
from os.path import join

import numpy as np
from PIL import Image

from data.base import VideoDataset
import cv2

class Weather(VideoDataset):
    """
    Weather dataset.

    Attributes
    ----------
    data : list
        List containing the dataset data. For weather, it consists of a list of lists of image files, representing video
        frames.
    nx : int
        Width and height of the video frames.
    seq_len : int
        Number of frames to produce.
    train : bool
        Whether to use the training or testing dataset.
    """

    def __init__(self, data, nx, seq_len, train):
        """
        Parameters
        ----------
        data : list
            List containing the dataset data. For Weather, it consists of a list of lists of image files, representing
            video frames.
        nx : int
            Width and height of the video frames.
        seq_len : int
            Number of frames to produce.
        train : bool
            Whether to use the training or testing dataset.
        """
        self.data = data
        self.nx = nx
        self.nc = 3
        self.seq_len = seq_len
        self.train = train

    def _filter(self, data):
        return Weather(data, self.nx, self.seq_len, self.train)

    def __len__(self):
        if self.train:
            # Arbitrary number.
            # The number is a trade-off for max efficiency.
            # If too low, it is not good for batch size and multi-threaded dataloader.
            # If too high, it is not good for shuffling and sampling.
            return 500000
        return len(self.data)

    def __getitem__(self, index):
        if not self.train:
            # When testing, pick the selected video at its beginning (from the precomputed testing set)
            return self.data[index]
        # When training, pick a random video from the dataset, and extract a random sequence
        # Iterate until the selected video is long enough
        done = False
        while not done:
            vid_id = np.random.randint(len(self.data))
            vid = self.data[vid_id]
            vid_len = len(vid)
            if vid_len < self.seq_len:
                continue
            done = True
        # Random timestep for the beginning of the video
        t0 = np.random.randint(vid_len - self.seq_len + 1)
        # Extract the sequence from frame image files
        x = np.zeros((self.seq_len, self.nx, self.nx), dtype=np.uint8)
        # x = np.zeros((self.seq_len, self.nx, self.nx, self.nc), dtype=np.uint8)
        for t in range(self.seq_len):
            img_path = vid[t0 + t]
            # img = cv2.imread(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.nx, self.nx))
            x[t] += np.array(img)
        # Returned video is an uint8 NumPy array of shape (length, width, height)
        return x

    @classmethod
    def make_dataset(cls, data_dir, nx, seq_len, train):
        """
        Creates a dataset from the directory where the dataset is saved.

        Parameters
        ----------
        data_dir : str
            Path to the dataset.
        nx : int
            Width and height of the video frames.
        seq_len : int
            Number of frames to produce.
        train : bool
            Whether to use the training or testing dataset.

        Returns
        -------
        data.weather.Weather
        """
        # Select the right fold (train / test)
        if train:
            # Loads all preprocessed videos
            data_dir = join(data_dir, f'processed_{nx}')
            data = []
            for vid in os.listdir(data_dir):
                if not os.path.isdir(join(data_dir, vid)):
                    continue
                # removes March and use other months to be the training set
                month = vid[:3]
                if month == 'mar':
                    continue
                # Videos are lists of frame image files
                images = sorted([
                    join(data_dir, vid, img)
                    for img in os.listdir(join(data_dir, vid)) if os.path.splitext(img)[1] == '.png'
                ])
                data.append(images)
        else:
            # If testing, load the precomputed dataset
            fname = f'weather_test_set_{seq_len}.npz'
            dataset = np.load(join(data_dir, fname), allow_pickle=True)
            sequences = dataset['sequences']
            data = [sequences[i] for i in range(len(sequences))]
        # Create and return the dataset
        return cls(data, nx, seq_len, train)
