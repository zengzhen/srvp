# Copyright 2020 Mickael Chen, Edouard Delasalles, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os

import numpy as np

from os.path import join
from tqdm import trange
from torchvision import datasets
import sys

import json
import cv2

sys.path.append('/home/ubuntu/workspace/srvp')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Bouncing Balls testing set generation.',
        description='Generates the bouncing balls testing set. Videos are saved in \
                     an npz file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the testing set will be saved.')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=60,
                        help='Number of frames per testing sequences.')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    parser.add_argument('--frame_size', type=int, metavar='SIZE', default=64,
                        help='Size of generated frames.')
    args = parser.parse_args()

    initial_size = 128
    scale = initial_size / 800
    radius = int(60 * scale)

    # load pre-computed test set balls trajectories from npm js in DDPAE
    folder = 'balls_n4_t60_ex2000'
    traj_data = np.load(os.path.join(args.data_dir, folder, 'dataset_info.npy'))
    # Register videos
    test_videos = []

    # for i in range(len(traj_data)):
    for i in range(10):
        traj = traj_data[i]
        vid_len, n_balls = traj.shape[:2]

        images = np.zeros([args.seq_len, args.frame_size, args.frame_size], np.uint8)
        for fid in range(args.seq_len):
            temp = np.zeros([initial_size, initial_size], np.uint8)
            for bid in range(n_balls):
                # each ball:
                ball = traj[fid, bid]
                x, y = int(round(scale * ball[0])), int(round(scale * ball[1]))
                temp = cv2.circle(temp, (x, y), int(radius * ball[3]),
                                        255, -1)
            images[fid] = cv2.resize(temp, (args.frame_size, args.frame_size))

        # Register video and other information
        test_videos.append(images.astype(np.uint8))
        print(i)

    test_videos = np.array(test_videos, dtype=np.uint8).transpose(1, 0, 2, 3)

    # Save results at the given path
    fname = 'test_bouncing_balls.npz'
    print(f'Saving testset at {join(args.data_dir, fname)}')
    # Create the directory if needed
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    np.savez_compressed(join(args.data_dir, fname),
                        sequences=test_videos)
