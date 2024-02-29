

from torch.utils.data import Dataset
from torchvision import transforms
import utils
import mmcv
import torch
import numpy as np
from PIL import Image
import os.path as osp
import csv
import io
from tqdm import tqdm
import random
import torch.nn.functional as F

class EpicKitchen(Dataset):
    def __init__(self,  root,
                 meta_file = "assets/EpicKitchen-100-train.csv", 
                 img_size = 224, 
                 num_frames=2,
                 file_client_args=dict(backend='petrel')):
        """Define EpicKiten Dataset
        Args:
            root (str): path of images
            meta_file (str): Path of meta file
            img_size (int): input image resolution
            num_frames (int): frames number of input images sequences
            backend (str, optional): The storage backend type. Options are 'ceph','petrel'. Default: 'petrel'.
        .. warning::
            Meta file should be the required form
        """
        with open(meta_file, "r") as f:
            self.meta_file = csv.reader(f)
            self.meta_file = list(self.meta_file)[1:] # drop the head line
        self.root = root
        self.img_size = img_size
        self.num_frames = num_frames
        self.client = mmcv.FileClient(**file_client_args)
        self._create_transform()
        self._check()

    def __len__(self):
        return len(self.language_based_list)

    def _check(self):
        language_based_dict = {}
        for line in tqdm(self.meta_file):
            name = line[2]
            start_frame = int(line[6])
            end_frame = int(line[7])
            language = line[8]
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
        print(f"avalible language num is {self.__len__()}")


    def _create_transform(self):
        """Data Augmentation for epickitchen
        Training Default Settings:
            For a trajectory: RandomResizedCrop
            For each image: Colorjitter
        """
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.2, 1.0)),
            ]
        )
        self.single_img_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.5),
                transforms.ToTensor()
            ]
        )
    
    def _get_single_img(self, img_dict, cur_idx):
        _tmp = "0" * (10 - len(f"{cur_idx}"))
        frame_name = f"frame_{_tmp}{cur_idx}.jpg"
        img_path = osp.join(img_dict, frame_name)
        try:
            value = self.client.get(img_path)
            img_bytes = np.frombuffer(value, np.uint8)
            buff = io.BytesIO(img_bytes)
            with Image.open(buff) as img:
                img = img.convert('RGB')
            img = self.single_img_transform(img)
        except:
            print(f"Error when get img {img_path}, try to get nearest one")
            if cur_idx > 1:
                return self._get_single_img(img_dict, cur_idx - 1)
            else:
                return self._get_single_img(img_dict, cur_idx + 1)
        return img
    

    def _get_a_traj(self, traj_meta):
        sampled_numbers = random.sample(range(traj_meta['start_idx'], traj_meta['end_idx'] + 1), self.num_frames)
        sorted_numbers = sorted(sampled_numbers)
        frames = [
            self._get_single_img(traj_meta['path'], int(i))
            for i in sorted_numbers
        ]
        frames = torch.stack(frames, dim = 0)
        frames = self.transform(frames)
        return frames # shape-> F, 3, H, W
        
    def __getitem__(self, index):
        img_training_data = random.choice(self.language_based_list[index])
        imgs = self._get_a_traj(img_training_data)
        return imgs, img_training_data['language']




def EpicKitchenDataLoader(root,
                      train_meta_file,
                      img_size=224,
                      batch_size=2,
                      num_workers=8,
                      pin_mem=True,
                      num_frames = 2
                      ):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_dataset = EpicKitchen(root=root, 
                    meta_file=train_meta_file, 
                    img_size=img_size, 
                    num_frames=num_frames)
    sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True
    )
    return train_dataloader



import sys
import math
import clip
from typing import Iterable
def train_one_epoch(model: torch.nn.Module, 
                    loss_model: torch.nn.Module, ,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int,
                    tb_logger=None, 
                    start_idx=0,
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
        loss = loss_model(visual_features, text_features)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
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

    