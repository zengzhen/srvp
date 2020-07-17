# Converted from https://github.com/edenton/svg/blob/master/data/convert_kth.lua

import argparse
import os
import subprocess
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='weather dataset preprocessing.',
        description='Generates training and testing videos for the weather dataset from the original videos, and stores \
                     them in folder `processed_${SIZE}` in the same directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where videos from the original dataset are stored.')
    parser.add_argument('--image_size', type=int, metavar='SIZE', default=128,
                        help='Width and height of resulting processed videos.')
    parser.add_argument('--frame_rate', type=int, metavar='RATE', default=25,
                        help='Frame rate at which videos are processed.')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f'Error with data directory: {args.data_dir}')

    video_path = os.path.join(args.data_dir, 'cut')
    # Process all videos
    for vid in os.listdir(video_path):
        print(vid)
        if os.path.splitext(vid)[1] != '.mp4':
            continue
        fname = vid[:-4]
        print(fname)
        data_dir = os.path.join(args.data_dir, f'processed_{args.image_size}', fname)
        if os.path.isdir(data_dir):
            continue
        os.makedirs(data_dir)
        # Process selected video: stop each video at 2:18 s, because after that are slides
        cmd = [
            'ffmpeg',
            '-i', os.path.join(args.data_dir, 'cut', vid),
            '-r', str(args.frame_rate),
            '-f', 'image2',
            os.path.join(args.data_dir, f'processed_{args.image_size}', fname, f'image-%05d.png')
        ]
        subprocess.call(cmd)

        img_path = os.path.join(args.data_dir, f'processed_{args.image_size}', fname)
        for img_name in os.listdir(img_path):
            full_name = os.path.join(img_path, img_name)
            img = cv2.imread(full_name, cv2.IMREAD_COLOR)
            # crop image to hand picked region 64x64
            img = img[188:316, 288:416]
            cv2.imwrite(full_name, img)
