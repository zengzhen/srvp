import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import argparse
import math
from preprocessing.market_heatmap import make_test_set

def addMarketLabel(data_dir, image):
    # names of markets from CSV files in data folder
    names = make_test_set.loadCSVData(data_dir, getCSVName=True)
    # overlay market labels on 3x3 grid image
    for id in range(len(names)):
        # 9 markets visualized in 3x3 grid
        tile_w = 21
        tile_h = 21
        x, y = (int(math.floor(id/3.0)) * tile_h + 2, id%3 * tile_w + 11)
        # put market label text
        cv2.putText(image, names[id], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser('''
#         Visualize test results in 3-row videos: GT, Pred_best, Pred_worst
#         ''')
    parser.add_argument('--dataset', type=str, required=True, choices=['KTH', 'MMNIST', 'Billiards','BouncingBalls', 'weather', 'market_heatmap', 'payments'],
                        help='Name of the dataset')
    parser.add_argument('--mode', type=str, required=False, choices=['s', 'd'],
                        help='Mode (s/d) for stochastic or deterministic MMNIST')
    parser.add_argument('--metric', type=str, default='lpips',
                        help='Metric to use for choosing best and worst samples')
    parser.add_argument('--plot', type=str, default=False,
                        help='Metric to use for choosing best and worst samples')
    args = parser.parse_args()

    if args.dataset == 'MMNIST' and args.mode is None:
        parser.error("dataset MMNIST requires --mode")

    plot = args.plot
    dataset = args.dataset

    if args.dataset == 'KTH':
        test_path = '/home/ubuntu/KTH_dataset/svg_test_set_40.npz'
        pred_path = '/home/ubuntu/workspace/srvp/models/kth/'
    elif args.dataset == "MMNIST":
        if args.mode == 's':
            test_path = '/home/ubuntu/MMNIST_dataset/smmnist_test_2digits_64.npz'
            pred_path = '/home/ubuntu/workspace/srvp/models/mmnist/stochastic/'
            dataset = dataset + '_Stoc'
        elif args.mode == 'd':
            test_path = '/home/ubuntu/MMNIST_dataset/mmnist_test_2digits_64.npz'
            pred_path = '/home/ubuntu/workspace/srvp/models/mmnist/deterministic/'
            dataset = dataset + '_Det'
    elif args.dataset == "Billiards":
        test_path = '/home/ubuntu/Billiards_dataset/billiards.npz'
        pred_path = '/home/ubuntu/workspace/srvp/models/billiard/'
    elif args.dataset == "BouncingBalls":
        test_path = '/home/ubuntu/BouncingBalls_dataset/test_bouncing_balls.npz'
        pred_path = '/home/ubuntu/workspace/srvp/models/bouncing_balls/'
    elif args.dataset == "weather":
        test_path = '/home/ubuntu/weather_dataset/weather_test_set_60.npz'
        pred_path = '/home/ubuntu/workspace/srvp/models/weather/'
    elif args.dataset == "market_heatmap":
        test_path = '/home/ubuntu/market_heatmap/test_market_heatmap.npz'
        pred_path = '/home/ubuntu/workspace/srvp/models/market_heatmap/'
        market_path = '/home/ubuntu/market_heatmap/test/'
    elif args.dataset == "payments":
        test_path = '/home/ubuntu/PYMT/test_payments.npz'
        pred_path = '/home/ubuntu/workspace/srvp/models/payments/'

    test_set = np.load(test_path)
    pred_best = np.load(pred_path + args.metric + '_best.npz')
    pred_worst = np.load(pred_path + args.metric + '_worst.npz')

    video_name_list = []
    plt_name_list = []
    color_convert = True
    show_residual = False
    # KTH: test_set:'samples', 'persons', 'actions', 'sequences'
    # MNIST: test_set: 'latents', 'labels', 'digits', 'sequences'
    if args.dataset == 'KTH':
        cond_num = 10
        pred_num = 30
        test_set_sequence = test_set['sequences']
        for j in range(len(test_set_sequence)):
            video_name_list.append(test_set['actions'][j] + '_' + str(test_set['persons'][j]))
            plt_name_list.append(test_set['actions'][j])
    elif args.dataset == 'MMNIST':
        if args.mode == 's':
            cond_num = 5
            pred_num = 20
        elif args.mode == 'd':
            cond_num = 5
            pred_num = 95
        test_set_sequence = test_set['sequences'].transpose(1, 0, 2, 3)
        for j in range(len(test_set_sequence)):
            digits_name = ''.join([str(i) for i in test_set['labels'][j]])
            video_name_list.append(digits_name)
            plt_name_list.append(digits_name)
    elif args.dataset == 'Billiards':
        cond_num = 5
        pred_num = 95
        test_set_sequence = test_set['sequences'].transpose(1, 0, 2, 3)
        for j in range(len(test_set_sequence)):
            name = str(j)
            video_name_list.append(name)
            plt_name_list.append(name)
    elif args.dataset == 'BouncingBalls':
        cond_num = 10
        pred_num = 50
        test_set_sequence = test_set['sequences'].transpose(1, 0, 2, 3)
        for j in range(len(test_set_sequence)):
            name = str(j)
            video_name_list.append(name)
            plt_name_list.append(name)
    elif args.dataset == 'weather':
        cond_num = 10
        pred_num = 50
        test_set_sequence = test_set['sequences']
        color_convert = True
        for j in range(len(test_set_sequence)):
            name = str(j)
            video_name_list.append(name)
            plt_name_list.append(name)
    elif args.dataset == 'market_heatmap':
        cond_num = 10
        pred_num = 5
        test_set_sequence = test_set['sequences'].transpose(1, 0, 2, 3, 4)
        color_convert = False
        show_residual = True
        for j in range(len(test_set_sequence)):
            name = str(j)
            video_name_list.append(name)
            plt_name_list.append(name)
    elif args.dataset == 'payments':
        cond_num = 8
        pred_num = 25-8
        test_set_sequence = test_set['sequences'].transpose(1, 0, 2, 3)
        for j in range(len(test_set_sequence)):
            name = str(j)
            video_name_list.append(name)
            plt_name_list.append(name)

    for j in range(len(test_set_sequence)):
        img_array = []
        video_name = video_name_list[j]
        plt_name = plt_name_list[j]
        sequence = test_set_sequence[j]
        # visualize input frames
        if plot:
            fig = plt.figure(figsize=(15,6))
        for i in range(cond_num):
            if color_convert:
                img = cv2.cvtColor(sequence[i], cv2.COLOR_BGR2RGB)
            else:
                img = sequence[i]
            cv2.rectangle(img,(0,0),(img.shape[0]-1,img.shape[1]-1),(0,255,0),2)
            # add labels of market if needed
            if args.dataset == 'market_heatmap':
                img = addMarketLabel(market_path, img)

            # three rows of images: upper gt, lower pred
            if show_residual:
                residual = np.zeros([img.shape[0], img.shape[1], 3], np.uint8)
                if args.dataset == 'market_heatmap':
                    residual = addMarketLabel(market_path, residual)
                conc_img = cv2.vconcat([img, img, img, residual])
            else:
                conc_img = cv2.vconcat([img, img, img])
            img_array.append(conc_img)

            if plot:
                plt.subplot(1, cond_num, i+1)
                plt.axis("off")
                plt.imshow(img)
                plt.title('%s-%d' % (plt_name, i+1), fontsize=10)
        # visualize predicted frames
        pred_seq_best = pred_best['samples'][j]
        pred_seq_worst = pred_worst['samples'][j]
        if plot:
            fig = plt.figure(figsize=(15,6))
        for i in range(pred_num):
            if color_convert:
                img_best = cv2.cvtColor(pred_seq_best[i], cv2.COLOR_BGR2RGB)
                img_worst = cv2.cvtColor(pred_seq_worst[i], cv2.COLOR_BGR2RGB)
            else:
                img_best = pred_seq_best[i]
                img_worst = pred_seq_worst[i]
            cv2.rectangle(img_best,(0,0),(img_best.shape[0]-1,img_best.shape[1]-1),(0,165,255),2)
            cv2.rectangle(img_worst,(0,0),(img_worst.shape[0]-1,img_worst.shape[1]-1),(0,165,255),2)
            if args.dataset == 'market_heatmap':
                img_best = addMarketLabel(market_path, img_best)
                img_worst = addMarketLabel(market_path, img_worst)
            # two rows of images: upper gt, lower pred
            if color_convert:
                gt_img = cv2.cvtColor(sequence[i+cond_num], cv2.COLOR_BGR2RGB)
            else:
                gt_img = sequence[i+cond_num]
            cv2.rectangle(gt_img,(0,0),(gt_img.shape[0]-1,gt_img.shape[1]-1),(0,255,0),2)
            if args.dataset == 'market_heatmap':
                gt_img = addMarketLabel(market_path, gt_img)

            if show_residual:
                residual = abs(gt_img.astype(np.int32) - img_best.astype(np.int32)).astype(np.uint8)
                if args.dataset == 'market_heatmap':
                    residual = addMarketLabel(market_path, residual)
                conc_img = cv2.vconcat([gt_img, img_best, img_worst, residual])
            else:
                conc_img = cv2.vconcat([gt_img, img_best, img_worst])
            img_array.append(conc_img)

            if plot:
                plt.subplot(int(math.ceil(pred_num/cond_num)), cond_num, i+1)
                plt.axis("off")
                plt.imshow(img)
                plt.title('%s-%d' % (plt_name, i+11), fontsize=10)

        video_file = 'results/' + dataset + '/' + video_name + '.mp4'
        print(video_file)
        if show_residual:
            out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 10, (img.shape[0], img.shape[1]*4), isColor=True)
        else:
            out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 10, (img.shape[0], img.shape[1]*3), isColor=True)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
