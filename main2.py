from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import wandb
from pylightning_mods import Lightning_Pct
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_API_KEY"] = "04a5d6fba030b76e5b620f5bd6509cf7dffebb8b"


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main2.py checkpoints'+'/'+args.exp_name+'/'+'main2.py.backup')
    os.system('cp pylightning_mods.py checkpoints'+'/'+args.exp_name+'/'+'pylightning_mods.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=2,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=2,
                            batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = "cuda" if args.cuda else "cpu"

    model = Lightning_Pct(args)
    print(str(model))
    #model = nn.DataParallel(model)

    wandb_logger = WandbLogger(project="PCT_Pytorch", name=args.exp_name, config=args)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=device, devices=1, logger=[wandb_logger], gradient_clip_val=2, default_root_dir="some/path/")
    trainer.fit(model, train_loader, test_loader)
    wandb.finish()

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = "cuda" if args.cuda else "cpu"

    model = Lightning_Pct(args)
    #ckpt_path = "checkpoints/"+args.exp_name+"/models/best.ckpt"
    ckpt_path = "PCT_Pytorch/6w3c4337/checkpoints/epoch=99-step=15300.ckpt"
    #model = model.load_from_checkpoint(ckpt_path, args)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    #model = nn.DataParallel(model) 
    wandb_logger = WandbLogger(project="PCT_Pytorch", name=args.exp_name, config=args)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=device, devices=1, logger=[wandb_logger], gradient_clip_val=2)
    trainer.test(model, test_loader)
    wandb.finish()

def visualize(args, io):
    
    obj = ModelNet40(partition='test', num_points=args.num_points)[3]

    device = "cuda" if args.cuda else "cpu"

    x = torch.from_numpy(obj[0]).unsqueeze(0).to(device)
    print(x.shape)
    y = torch.from_numpy(obj[1])
    model = Lightning_Pct(args).to(device)
    #ckpt_path = "checkpoints/"+args.exp_name+"/models/best.ckpt"
    ckpt_path = "PCT_Pytorch/0g02r60u/checkpoints/epoch=99-step=15300.ckpt"
    #model = model.load_from_checkpoint(ckpt_path, args)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.eval()
    #model = nn.DataParallel(model) 
    logits, masks, distrs = model.forward_with_mask([x,y])
    masks = [mask.detach().cpu().numpy() for mask in masks]
    distrs = [distr[0].detach().cpu().numpy() for distr in distrs]
    print(masks[1].shape)
    preds = logits.max(dim=1)[1]
    print(preds, y)
    print(masks[0])

    print(distrs[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  
    xs = x[0, :, 0].cpu().numpy()
    ys = x[0, :, 1].cpu().numpy()
    zs = x[0, :, 2].cpu().numpy()
    # creating the plot
    ax.scatter(xs, ys, zs, color='green')
    #ax.scatter(xs[masks[-1]], ys[masks[-1]], zs[masks[-1]], color='red')
    
    # setting title and labels
    ax.set_title("3D plot")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.autoscale()
    # displaying the plot
    plt.savefig("test.png")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--visualize', type=bool,  default=False,
                        help='visualize the point masking')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--adaptive', type=bool, default=False,
                        help='Adaptive point drop')
    parser.add_argument('--alpha', type=float, default=0.01, 
                        help="Weight for the drop regularization loss")
    parser.add_argument('--layers_to_drop', help='List of layers where points are dropped', 
                        type=lambda s: [int(item) for item in s.split(',')], default="-1")
    parser.add_argument('--drop_ratio', help='List of drop ratios for each layer', 
                        type=lambda s: [float(item) for item in s.split(',')], default="-1")
    parser.add_argument('--no_wandb', type=bool, default=False,
                        help='Use wandb')
    parser.add_argument('--drop_slow_start', type=int, default=0,
                        help='Drop warmup starting epoch')
    parser.add_argument('--drop_slow_end', type=int, default=10,
                        help='Drop warmup ending epoch')
    parser.add_argument('--drop_warmup', type=bool, default=False,
                        help='Enables drop warmup')
    args = parser.parse_args()
    if args.layers_to_drop == [-1]:
        args.layers_to_drop = []
    if args.drop_ratio == [-1]:
        args.drop_ratio = []

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    if not args.no_wandb:
        wandb.login()
        wandb.init(config=args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        if not args.visualize:
            test(args, io)
        else:
            visualize(args, io)