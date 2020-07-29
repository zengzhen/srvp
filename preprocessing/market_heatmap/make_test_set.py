import argparse
import os

import numpy as np

from os.path import join
from tqdm import trange
from torchvision import datasets
import sys

import json
import cv2

import math
import csv
import matplotlib.pyplot as plt

sys.path.append('/home/ubuntu/workspace/srvp')

def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def sigmoid(x, shift=0.0, k=1.0):
    return 1.0 / (1.0 + math.exp(-k*(x-shift)))

def loadCSVData(data_dir, getCSVName=False):
    if getCSVName: # just get names
        names = []
        for index, market in enumerate(list_files(data_dir, 'csv')):
            # print(market)
            market = os.path.splitext(market)[0][:3]
            names.append(market)
        return names
    else: # get data
        data = []
        for index, market in enumerate(list_files(data_dir, 'csv')):
            # print(market)
            market_data = np.genfromtxt(os.path.join(data_dir, market), delimiter=',', skip_header=1)
            # column 4 - close values
            market_data = market_data[:,4]
            # calcualte gain/loss percentage based on close values
            step = 5
            market_data = (market_data[step:] - market_data[:-step])/market_data[:-step]*100

            # visualize percentage change with different step size
            # fig = plt.figure(figsize=(15,6))
            # step = 1
            # temp = (market_data[step:] - market_data[:-step])/market_data[:-step]*100
            # temp_annt = "step=1, mean={mean}, std={std}".format(mean=round(np.mean(temp),2), std=round(np.std(temp),2))
            # temp1, = plt.plot(temp, label=temp_annt)
            # step = 5
            # temp = (market_data[step:] - market_data[:-step])/market_data[:-step]*100
            # temp_annt = "step=5, mean={mean}, std={std}".format(mean=round(np.mean(temp),2), std=round(np.std(temp),2))
            # for temp_idx in range(3):
            #     temp = np.insert(temp, 0, 0)
            # temp2, = plt.plot(temp, label=temp_annt)
            # plt.legend(handles=[temp1, temp2])

            # plt.show()
            # plt.savefig('percentage_change_' + market + '.png')

            if index==0:
                data = market_data
            else:
                data = np.column_stack((data, market_data))

        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Market heatmap testing set generation.',
        description='Generates the market heatmap testing set. Videos are saved in \
                     an npz file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the testing set will be saved.')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=15,
                        help='Number of frames per testing sequences.')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    parser.add_argument('--frame_size', type=int, metavar='SIZE', default=64,
                        help='Size of generated frames.')
    args = parser.parse_args()

    # read in all CSV files: 01.01.2019 - 12.31.2019
    test_data_dir = args.data_dir + '/test/'
    data = loadCSVData(test_data_dir)

    # Register videos
    test_videos = []

    for i in range(10):
        # Random timestep for the beginning of the video
        t0 = np.random.randint(len(data) - args.seq_len + 1)
        # Extract the sequence from stock_hist files
        n_markets = data[0].shape[0]

        images = np.zeros([args.seq_len, args.frame_size, args.frame_size, 3], np.uint8)
        for t in range(args.seq_len):
            for id in range(n_markets):
                # 9 markets visualized in 3x3 grid
                tile_w = 21
                tile_h = 21
                perc = data[t0+t][id]
                x1, y1 = (int(math.floor(id/3.0)) * tile_h, id%3 * tile_w)
                x2, y2 = (x1+tile_h-2, y1+tile_w-2)
                # draw tiles: green v.s. red
                if perc >= 0: # green
                    images[t] = cv2.rectangle(images[t], (x1, y1), (x2, y2), (0, 255*sigmoid(perc, k=0.5), 0), -1)
                else: # red
                    images[t] = cv2.rectangle(images[t], (x1, y1), (x2, y2), (0, 0, 255*sigmoid(abs(perc), k=0.5)), -1)
                # draw tiles: light green v.s. dark green
                # images[t] = cv2.rectangle(images[t], (x1, y1), (x2, y2), (0, 255*sigmoid(perc, k=2), 0), -1)

        # Register video and other information
        test_videos.append(images.astype(np.uint8))
        print(i)

        video_file = '/home/ubuntu/workspace/srvp/results/market_heatmap/test_%04d.mp4' % (i)
        print(video_file)
        out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 1, (args.frame_size, args.frame_size), isColor=True)
        for vid in range(len(images)):
            out.write(images[vid])
        out.release()

    test_videos = np.array(test_videos, dtype=np.uint8).transpose(1, 0, 2, 3, 4)

    # Save results at the given path
    fname = 'test_market_heatmap.npz'
    print(f'Saving testset at {join(args.data_dir, fname)}')
    # Create the directory if needed
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    np.savez_compressed(join(args.data_dir, fname),
                        sequences=test_videos)
