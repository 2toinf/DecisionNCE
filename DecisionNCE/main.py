# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
from email.policy import default
import logging
import os
import random
from tabnanny import verbose
import numpy as np
import time
import utils
import json
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from tensorboardX import SummaryWriter
from datasets.EpicKitchen_engine import train_one_epoch, EpicKitchenDataLoader
from pathlib import Path
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from losses import DecisionNCELoss
from model import CLIPBasedEncoder
import clip

def get_args_parser():
    parser = argparse.ArgumentParser('DecisionNCE training script', add_help=False)
    
    # Data preparation
    parser.add_argument('--image_path',  type=str)
    parser.add_argument('--meta_file',  type=str)
    
    # Base Settings
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--num_frames', default=2, type=int, 
                        help='frame numbers for each sample')
    
    parser.add_argument('--model', default='RN50', type=str, 
                        metavar='MODEL', 
                        choices=clip.available_models(),
                        help='Name of model to train, \
                        please make sure your model is in list of available_models of clip')
    
    parser.add_argument('--loss-type', default="DecisionNCE-T", type=str,
                        help='training loss')
    parser.add_argument('--logit-scale', default=100, type=int,
                        help='logit scale for training loss')
    
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "fusedadamw"')
    parser.add_argument('--opt-eps', default=1e-6, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.001)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--seed', default=0, type=int)
    
    # Resume & Checkpoint Save & evaluation parameters
    parser.add_argument('--save_interval', default=50, type=int,
                        help='(default: 50ep)')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # DataLoader parameters
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29529, type=int, help='port')


    
    return parser

def main(args):
    utils.init_distributed_mode(args, verbose=True)
    output_dir = Path(args.output_dir)
    tb_logger = None
    if utils.get_rank() == 0:
        tensorboard_path = os.path.join(output_dir, 'events')
        tb_logger = SummaryWriter(tensorboard_path)
    utils.init_log(__name__, log_file=os.path.join(output_dir, 'full_log.txt'))
    logger = logging.getLogger(__name__)
    print = logger.info
    print(args)
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    train_dataloader = EpicKitchenDataLoader(
        root=args.image_path,
        train_meta_file=args.meta_file,
        img_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        num_frames=args.num_frames
    )

    model = CLIPBasedEncoder(modelid=args.model, device=device)
    loss = DecisionNCELoss(logit_scale=args.logit_scale, loss_type=args.loss_type)
    model.to(device)
    
    torch.distributed.barrier()
    model_without_ddp = model
    print(f'batch size {args.batch_size}, world size {utils.get_world_size()}')
    print(f'scaled lr {args.lr}')
    optimizer = create_optimizer(args, model_without_ddp)
    model = NativeDDP(model, device_ids=[args.gpu], find_unused_parameters=False)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params: {n_parameters}')
    torch.distributed.barrier()
    
    lr_scheduler, _ = create_scheduler(args, optimizer)
    try:
        checkpoint = utils.load_checkpoint(args.resume)
        print('>>>>>> resume from {}'.format(args.resume))
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    except:
        print('>>>>>> no resume')

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    start_idx = args.start_epoch * len(train_dataloader)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, 
            loss=loss,
            data_loader=train_dataloader,
            optimizer=optimizer, 
            device=device, 
            epoch=epoch, 
            tb_logger=tb_logger, 
            start_idx=start_idx
        )
        start_idx += len(train_dataloader)
        lr_scheduler.step(epoch)
        if args.output_dir and utils.is_main_process() and epoch % args.save_interval == 0:
            with open(os.path.join(output_dir, "log.txt"), 'a') as f:
                f.write(json.dumps({**{f'train_{k}': v for k, v in train_stats.items()},
                                        'epoch': epoch,
                                        'n_parameters': n_parameters}) + "\n")
            utils.save_checkpoint({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
            }, os.path.join(output_dir, f"ckpt_{epoch}ep.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DecisionNCE training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

