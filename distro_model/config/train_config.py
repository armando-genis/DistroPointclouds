"""
# -*- coding: utf-8 -*-
# Description: The configurations of the project will be defined here
"""

import os
import argparse

import torch
from easydict import EasyDict as edict


def parse_dataloader_configs():
    parser = argparse.ArgumentParser(description='DataLoader Configuration')

    # Path to KITTI dataset (hardcoded default)
    parser.add_argument(
        '--dataset-dir', type=str,
        default='/workspace/DistroPointclouds/kitti',
        help='Full path to the KITTI dataset directory'
    )

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Mini-batch size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of worker processes for data loading'
    )
    parser.add_argument(
        '--pin_memory', action='store_true',
        help='Pin host memory for faster GPU transfers'
    )

    # Dataset augmentation flags
    parser.add_argument(
        '--multiscale_training', action='store_true',
        help='Enable random scaling of BEV inputs during training'
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Limit dataset to this many samples (for debugging)'
    )
    parser.add_argument(
        '--mosaic', action='store_true',
        help='Use mosaic augmentation'
    )
    parser.add_argument(
        '--random_padding', action='store_true',
        help='Apply random padding when using mosaic'
    )

    ####################################################################
    ##############  Augmentation probabilities and parameters    #######
    ####################################################################
    parser.add_argument(
        '--hflip_prob', type=float, default=0.5,
        help='Probability of horizontal flip'
    )
    parser.add_argument(
        '--cutout_prob', type=float, default=0.0,
        help='Probability of cutout augmentation'
    )
    parser.add_argument(
        '--cutout_nholes', type=int, default=1,
        help='Number of holes for cutout'
    )
    parser.add_argument(
        '--cutout_ratio', type=float, default=0.3,
        help='Max ratio of cutout area'
    )
    parser.add_argument(
        '--cutout_fill_value', type=float, default=0.0,
        help='Fill value for cutout regions'
    )

    args = parser.parse_args()
    configs = edict(vars(args))

    # No distributed sampling by default
    configs.distributed = False

    return configs