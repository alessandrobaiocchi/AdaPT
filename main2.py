from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ShapeNet_partseg
from datasets.ShapeNet55 import ShapeNet55DataModule
from datasets.ScanObjectNN import ScanObjectNNDataModule
from datasets.ModelNet40Ply2048 import ModelNet40Ply2048DataModule
from model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import wandb
from pylightning_mods import Lightning_pct_adaptive, Lightning_pct_merger, Lightning_pct
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hydra
from omegaconf import DictConfig

import time 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_API_KEY"] = "04a5d6fba030b76e5b620f5bd6509cf7dffebb8b"


torch.set_float32_matmul_precision('medium')


#@hydra.main(config_path=".", config_name="config", version_base=None)
def _init_(cfg):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+cfg.experiment.exp_name):
        os.makedirs('checkpoints/'+cfg.experiment.exp_name)
    if not os.path.exists('checkpoints/'+cfg.experiment.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+cfg.experiment.exp_name+'/'+'models')
    os.system('cp main2.py checkpoints'+'/'+cfg.experiment.exp_name+'/'+'main2.py.backup')
    os.system('cp pylightning_mods.py checkpoints'+'/'+cfg.experiment.exp_name+'/'+'pylightning_mods.py.backup')
    os.system('cp model.py checkpoints' + '/' + cfg.experiment.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + cfg.experiment.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + cfg.experiment.exp_name + '/' + 'data.py.backup')


#@hydra.main(config_path=".", config_name="config", version_base=None)
def train(cfg, train_loader, test_loader):

    

    device = "cuda" if cfg.cuda else "cpu"


    if cfg.teacher.use_teacher:
        teacher = Lightning_pct(cfg)



    if cfg.teacher.use_teacher:
        if cfg.wandb:
            wandb_logger = WandbLogger(project="PCT_Pytorch", name="teacher", config=cfg)
            teacher_trainer = pl.Trainer(max_epochs=cfg.train.epochs, accelerator=device, devices=1, logger=[wandb_logger], gradient_clip_val=2)
        else:
            teacher_trainer = pl.Trainer(max_epochs=cfg.teacher.epochs, accelerator=device, devices=1, gradient_clip_val=2)
        print("Training teacher model")
        if cfg.teacher.checkpoint_path is not None:
            teacher.load_state_dict(torch.load(cfg.teacher.checkpoint_path)["state_dict"])
        else:
            teacher_trainer.fit(teacher, train_loader, test_loader)

    if cfg.train.adaptive.is_adaptive:
        if cfg.teacher.use_teacher:
            model = Lightning_pct_adaptive(cfg, teacher)
        else:
            model = Lightning_pct_adaptive(cfg)
    elif cfg.train.merger.is_merger:
        model = Lightning_pct_merger(cfg)
    else:
        model = Lightning_pct(cfg)
        #"ERROR: No model selected."
    
    #print(str(model))
    #model = nn.DataParallel(model)

    if cfg.teacher.use_teacher:
        print("Training student model")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='best_models', monitor="val/acc")

    if cfg.wandb:
        wandb_logger = WandbLogger(project="PCT_Pytorch", name=cfg.experiment.exp_name, config=cfg)
        trainer = pl.Trainer(max_epochs=cfg.train.epochs, accelerator=device, devices=1, logger=[wandb_logger], gradient_clip_val=2, callbacks=[checkpoint_callback])
    else:
        trainer = pl.Trainer(max_epochs=cfg.train.epochs, accelerator=device, devices=1, gradient_clip_val=2, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, test_loader)
    if cfg.wandb:
        wandb.finish()

#@hydra.main(config_path=".", config_name="config", version_base=None)
def test(cfg, test_loader):
    
    device = "cuda" if cfg.cuda else "cpu"

    model = Lightning_Pct(cfg)
    #ckpt_path = "checkpoints/"+args.exp_name+"/models/best.ckpt"
    #ckpt_path = "PCT_Pytorch/6w3c4337/checkpoints/epoch=99-step=15300.ckpt"  #DETERMINISTIC DROP
    ckpt_path = "PCT_Pytorch/1nca3jm1/checkpoints/epoch=99-step=15300.ckpt"   #GUMBEL NOISE

    #model = model.load_from_checkpoint(ckpt_path, args)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    #model = nn.DataParallel(model) 
    if cfg.wandb:
        wandb_logger = WandbLogger(project="PCT_Pytorch", name=cfg.experiment.exp_name, config=cfg)
        trainer = pl.Trainer(max_epochs=cfg.test.epochs, accelerator=device, devices=1, logger=[wandb_logger], gradient_clip_val=2)
    else:
        trainer = pl.Trainer(max_epochs=cfg.test.epochs, accelerator=device, devices=1, gradient_clip_val=2)
    trainer.test(model, test_loader)
    if cfg.wandb:
        wandb.finish()

#@hydra.main(config_path=".", config_name="config", version_base=None)
def visualize(cfg):
    
    obj = ModelNet40(partition='test', num_points=cfg.test.num_points)[21]

    device = "cuda" if cfg.cuda else "cpu"

    x = torch.from_numpy(obj[0]).unsqueeze(0).to(device)
    print(x.shape)
    y = torch.from_numpy(obj[1])
    model = Lightning_pct_adaptive(cfg).to(device)
    ckpt_path = "PCT_Pytorch/fwnnazxt/checkpoints/epoch=499-step=76500.ckpt" #DETERMINISTIC DROP
    #model = model.load_from_checkpoint(ckpt_path, args)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.eval()
    #model = nn.DataParallel(model) 
    ret = model.forward_with_mask([x,y])
    logits = ret[0]
    masks = ret[1]
    distrs = ret[2]
    xyz = ret[3]
    new_xyz = ret[4]

    masks = [mask.detach().cpu().numpy() for mask in masks]
    distrs = [distr[0].detach().cpu().numpy() for distr in distrs]
    print(masks[1].shape)
    preds = logits.max(dim=1)[1]
    print(preds, y)
    print(masks[2])

    print(distrs[2])

    keepprob = [temp[1] for temp in distrs[2]]

    #print(keepprob)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(keepprob, bins=15)
    plt.savefig("histdrop2.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  
    print("xyzshape: ",xyz.shape)
    print("newxyzshape: ",new_xyz.shape)

    xs = xyz[0, :, 0].cpu().numpy()
    ys = xyz[0, :, 1].cpu().numpy()
    zs = xyz[0, :, 2].cpu().numpy()
    new_xs = new_xyz[0, :, 0].cpu().numpy()
    new_ys = new_xyz[0, :, 1].cpu().numpy()
    new_zs = new_xyz[0, :, 2].cpu().numpy()
    # creating the plot
    #ax.scatter(xs[masks[-1]], ys[masks[-1]], zs[masks[-1]], color='red')
    
    # setting title and labels
    ax.set_title("3D plot")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    
    

    ax.scatter(xs, ys, zs, color='green', alpha=0.2)
    ax.autoscale()
    # displaying the plot
    plt.savefig("points.png")

    #ax.clear()
    
    mask = masks[-1].squeeze().astype(bool)
    #print(mask.shape, mask)
    #print(new_xs.shape, new_ys.shape, new_zs.shape)

    #print(new_xs[mask].shape, new_ys[mask].shape, new_zs[mask].shape)

        
    ax.scatter(new_xs[np.logical_not(mask)], new_ys[np.logical_not(mask)], new_zs[np.logical_not(mask)], color='blue', alpha=1)
    ax.autoscale()
    plt.savefig("sampled.png")

    #ax.clear()

    ax.scatter(new_xs[mask], new_ys[mask], new_zs[mask], color='red', alpha=1)
    ax.autoscale()
    plt.savefig("sampled_kept.png")

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):

    _init_(cfg)

    io = IOStream('checkpoints/' + cfg.experiment.exp_name + '/run.log')

    if cfg.wandb:
        wandb.login()
        wandb.init(config=cfg)

    cfg.cuda = cfg.cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.experiment.seed)
    if cfg.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(cfg.experiment.seed)
    else:
        io.cprint('Using CPU')

    if cfg.dataset == "ModelNet40":
        dataset = ModelNet40Ply2048DataModule(batch_size=cfg.train.batch_size)
        #train_loader = DataLoader(ModelNet40(partition='train', num_points=cfg.train.num_points), num_workers=8,
        #                    batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
        #test_loader = DataLoader(ModelNet40(partition='test', num_points=cfg.test.num_points), num_workers=8,
        #                    batch_size=cfg.test.batch_size, shuffle=False, drop_last=False)
    elif cfg.dataset == "ShapeNet55":
        dataset = ShapeNet55DataModule(batch_size=cfg.train.batch_size)
    elif cfg.dataset == "ScanObjectNN":
        dataset = ScanObjectNNDataModule(batch_size=cfg.train.batch_size)
    else:
        raise Exception("Dataset not supported")
    dataset.setup()
    train_loader = dataset.train_dataloader()
    test_loader = dataset.val_dataloader()
    cfg.nclasses = dataset.num_classes

    if not cfg.eval:
        train(cfg, train_loader, test_loader)
    else:
        if not cfg.visualize_pc:
            test(cfg, test_loader)
        else:
            visualize(cfg)

if __name__ == "__main__":
    main()