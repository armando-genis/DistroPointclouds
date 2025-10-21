import time
import numpy as np
import sys
import random
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm
import cv2

# Set OpenCV to use non-GUI backend to avoid Qt issues in headless environments
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
cv2.setNumThreads(0)

# config loader
from config.train_config import parse_dataloader_configs
# data loader & and debbuger dataloader
from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
import data_process.kitti_bev_utils as bev_utils
from data_process import kitti_data_utils
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
import config.kitti_config as cnf

sys.path.append('./')

def main():
    print("Starting main function...")
    configs = parse_dataloader_configs()
    train_dataloader, train_sampler = create_train_dataloader(configs)
    print(f'number of batches in training set: {len(train_dataloader)}')

    for batch_i, (img_files, imgs, targets) in enumerate(train_dataloader):
        if not (configs.mosaic and configs.show_train_data):
            img_file = img_files[0]
            img_rgb = cv2.imread(img_file)
            calib = kitti_data_utils.Calibration(img_file.replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = invert_target(targets[:, 1:], calib, img_rgb.shape, RGB_Map=None)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        # print shape imgs
        print(f'imgs shape: {imgs.shape}')




def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)