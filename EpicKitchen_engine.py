

from torch.utils.data import Dataset
from torchvision import transforms
import utils
from petrel_client.client import Client as CephClient
import torchvision
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import os.path as osp
import csv
import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist

import random
import torch.nn.functional as F
# traj_name: 2, start_idx: 6, end_idx: 7, language: 8


class EpicKitchen(Dataset):
    def __init__(
                 self, 
                 root = "bridgedata:s3://mydata/epic-kitchens-100/",
                 meta_file = "/mnt/lustre/zhengjinliang/PPT/EpicKitchen-100-train2.csv", 
                 img_size = 224, 
                 train = True,
                 num_frames=2):
        self.train = train
        with open(meta_file, "r") as f:
            self.meta_file = csv.reader(f)
            self.meta_file = list(self.meta_file)
        self.root = root
        self.img_size = img_size
        self.num_frames = num_frames
        self.client = CephClient()
        self.create_transform()
        self.check_list()


    def __len__(self):
        return len(self.language_based_list)

    def check_list(self):
        language_based_dict = {}
        min_length = 1000
        for line in tqdm(self.meta_file):
            name = line[2]
            start_frame = int(line[6])
            end_frame = int(line[7])
            language = line[8]
            min_length = min(end_frame - start_frame, min_length)
            if language not in language_based_dict.keys():
                language_based_dict[language] = []

            language_based_dict[language].append({
                'path': osp.join(self.root, name),
                "start_idx": start_frame,
                "end_idx": end_frame,
                "language": language
            })
        self.language_based_dict = language_based_dict
        self.language_based_list = list(language_based_dict.values())
        print(f"avalible text num is {self.__len__()}")
        print(f"min_length is {min_length}")



    def create_transform(self):
        if self.train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.2, 1.0)),
                ]
            )
            self.single_img_transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size)
                ]
            )
            self.single_img_transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
        
    
    def get_single_img(self, img_dict, cur_idx):
        _tmp = "0" * (10 - len(f"{cur_idx}"))
        frame_name = f"frame_{_tmp}{cur_idx}.jpg"
        img_path = osp.join(img_dict, frame_name)
        try:
            value = self.client.Get(img_path)
            img_bytes = np.frombuffer(value, np.uint8)
            buff = io.BytesIO(img_bytes)
            with Image.open(buff) as img:
                img = img.convert('RGB')
            img = self.single_img_transform(img)
        except:
            print(f"Error when get img {img_path}, try to get nearest one")
            if cur_idx > 1:
                return self.get_single_img(img_dict, cur_idx - 1)
            else:
                return self.get_single_img(img_dict, cur_idx + 1)
        return img
    

    def get_a_traj(self, traj_meta):
        sampled_numbers = random.sample(range(traj_meta['start_idx'], traj_meta['end_idx'] + 1), self.num_frames)
        sorted_numbers = sorted(sampled_numbers)
        frames = [
            self.get_single_img(traj_meta['path'], int(i))
            for i in sorted_numbers
        ]
        frames = torch.stack(frames, dim = 0)
        frames = self.transform(frames)
        return frames # shape-> F, 3, H, W
        
    def __getitem__(self, index):

        # determined the negetive traj
        img_training_data = random.choice(self.language_based_list[index])
        imgs = self.get_a_traj(img_training_data)
        return imgs, img_training_data['language']




def EpicKitchenDataLoader(root= "bridgedata:s3://mydata/epic-kitchens-100/",
                      img_size=224,
                      batch_size=2,
                      num_workers=8,
                      pin_mem=True,
                      num_frames = 2
                      ):
    train_dataset = EpicKitchen(root=root, 
        meta_file="/mnt/lustre/zhengjinliang/PPT/EpicKitchen-100_train2.csv", 
        img_size=img_size, 
        num_frames=num_frames,
        train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True
    )


    eval_dataset = EpicKitchen(root=root, 
            meta_file="/mnt/lustre/zhengjinliang/PPT/EpicKitchen-100_test2.csv", 
            img_size=img_size, 
            num_frames=min(num_frames * 2, 10),
            train=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, 
        sampler=torch.utils.data.SequentialSampler(eval_dataset),
        batch_size=256,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    return train_dataloader, eval_dataloader







import sys
import math
import clip
from typing import Iterable
def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int,
                    tb_logger=None, 
                    start_idx=0
                    ):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    torch.cuda.synchronize()

    for visual_input, text_input in metric_logger.log_every(data_loader, print_freq, header):

        B, F, C, H, W = visual_input.shape
        visual_input = visual_input.reshape(B*F, C, H, W).to(device, non_blocking=True)
        text_input = clip.tokenize(text_input).to(device, non_blocking=True)
        visual_features, text_features = model(visual_input, text_input)
        visual_features = visual_features.reshape(B, F, visual_features.shape[-1])
        ppt_loss = Loss_DecisionNCE(visual_features, text_features)
        loss = ppt_loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        metric_logger.update(ppt_loss=ppt_loss.item())
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tb_logger is not None and utils.get_rank() == 0 and start_idx % 50 == 0:
            for k, meter in metric_logger.meters.items():
                tb_logger.add_scalar('train/{}_avg'.format(k), meter.global_avg, start_idx)
                tb_logger.add_scalar('train/{}_val'.format(k), meter.value, start_idx)
        start_idx += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def eval(model: torch.nn.Module, 
                    data_loader: Iterable, 
                    device: torch.device, 
                    epoch: int, 
                    tb_logger=None
                    ):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    torch.cuda.synchronize()

    for visual_input, text_input in metric_logger.log_every(data_loader, print_freq, header):

        B, F, C, H, W = visual_input.shape
        visual_input = visual_input.reshape(B*F, C, H, W).to(device, non_blocking=True)
        text_input = clip.tokenize(text_input).to(device, non_blocking=True)
        visual_features, text_features = model(visual_input, text_input)
        visual_features = visual_features.reshape(B, F, visual_features.shape[-1])
        acc_avg , acc_top1, acc_avg_text, acc_top1_text, loss, reward = eval_metric(
            visual_features, text_features
        )
        batch_size = B
        torch.cuda.synchronize()
        metric_logger.meters['acc_avg'].update(acc_avg.item(), n=batch_size)
        metric_logger.meters['acc_top1'].update(acc_top1, n=batch_size)
        metric_logger.meters['acc_avg_text'].update(acc_avg_text.item(), n=batch_size)
        metric_logger.meters['acc_top1_text'].update(acc_top1_text, n=batch_size)
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['reward'].update(reward.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if tb_logger is not None and utils.get_rank() == 0:
        for k, meter in metric_logger.meters.items():
            tb_logger.add_scalar('eval/{}_avg'.format(k), meter.global_avg, epoch)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



    