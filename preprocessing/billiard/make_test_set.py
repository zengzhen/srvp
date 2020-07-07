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
from PIL import Image
import cv2

sys.path.append('/home/ubuntu/workspace/srvp')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('''
        Generates the billiard testing set. Videos are saved in an npz file.
        ''')
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the testing set will be saved.')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=100,
                        help='Number of frames per testing sequences.')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    parser.add_argument('--image_size', type=int, metavar='SIZE', default=64,
                        help='Size of generated frames.')
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(args.seed)
    
    # Register videos
    test_videos = []
    
    processed_dir = join(args.data_dir, f'processed_{args.image_size}')
    images_fnames = sorted(os.listdir(join(processed_dir, 'billiar')))

    # hand picked region to test
    fps = 25
    time = [17*fps, 39*fps, 93*fps, 112*fps, 138*fps, 183*fps, 4965, 225*fps, 6091, 7095]
    x_min_max = [[80, 190], [120, 280], [160,330], [160, 360], [350, 490], [290, 460], [70, 200], [315, 470], [130, 280], [450, 565], [265, 475]]
    y_min_max = [[160, 310], [80, 220], [110, 300], [70,240], [35, 170], [180,296], [45, 175], [80,230], [187, 297], [35, 195], [35, 195]]

    for i in range(len(time)):
        # Randomly choose the beginning of the video extract to be included in the testing set
        # t_0 = np.random.randint(len(images_fnames) - args.seq_len + 1)
        t_0 = time[i]
        images = []
        for t in range(args.seq_len):
            filename = join(processed_dir, 'billiar', images_fnames[t_0 + t])
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # crop image to hand picked region & resize
            img = img[y_min_max[i][0]:y_min_max[i][1], x_min_max[i][0]:x_min_max[i][1]]
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # all interploations lead to weird artifacts

            img_array = np.array(img)
            images.append(img_array)

        x = np.array(images)
        test_videos.append(x.astype(np.uint8))
        print(i)
    test_videos = np.array(test_videos, dtype=np.uint8).transpose(1, 0, 2, 3)

    # Save results at the given path
    fname = 'billiards.npz'
    print(f'Saving testset at {join(args.data_dir, fname)}')
    # Create the directory if needed
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    np.savez_compressed(join(args.data_dir, fname),
                        sequences=test_videos)
