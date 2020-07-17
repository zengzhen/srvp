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
from PIL import Image
from tqdm import trange
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Weather testing set generation.',
        description='Generates the weather testing set from the testing videos by extracting fixed-length sequences. \
                     Videos as well as action and subject information are saved in an npz file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = argparse.ArgumentParser('''
        Generates the weather testing set from the testing videos by extracting fixed-length sequences.
        Videos as well as action and subject information are saved in an .npz file.
        ''')
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the dataset is stored and the testing set will be saved.')
    parser.add_argument('--size', type=int, metavar='SIZE', default=10,
                        help='Number of sequences to extract (size of the testing set).')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=60,
                        help='Number of frames per testing sequences.')
    parser.add_argument('--image_size', type=int, metavar='SIZE', default=128,
                        help='Width and height of videos.')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    args = parser.parse_args()

    # Fix random seed
    # np.random.seed(args.seed)

    # Update data directory with processed videos directlory
    processed_dir = join(args.data_dir, f'processed_{args.image_size}')

    sequences = []
    # Randomly extract a determined number of videos
    # use March as the test month
    vid = 'mar_2018'
    print(vid)
    for i in trange(args.size):
        images_fnames = sorted(os.listdir(join(processed_dir, vid)))
        # Randomly choose the beginning of the video extract to be included in the testing set
        t_0 = np.random.randint(len(images_fnames) - args.seq_len + 1)
        images = []
        for t in range(args.seq_len):
            # img = cv2.imread(join(processed_dir, vid, images_fnames[t_0 + t]), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(join(processed_dir, vid, images_fnames[t_0 + t]))
            images.append(img)
        sequences.append(np.array(images))

        video_file = '/home/ubuntu/weather_dataset/test/' + str(i) + '.mp4'
        print(video_file)
        out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.image_size, args.image_size), isColor=True)

        for i in range(len(images)):
            out.write(images[i])
        out.release()
    sequences = np.array(sequences)

    # Save the dataset
    save_file = join(args.data_dir, f'weather_test_set_{args.seq_len}.npz')
    print(f'Saving testset at {save_file}')
    np.savez_compressed(save_file, sequences=sequences)
