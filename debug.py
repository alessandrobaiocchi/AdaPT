from model import CrossAttention, Pct_nogroup, Pct
from pylightning_mods import Lightning_pct, Lightning_pct_adaptive
import torch
import torch.nn.functional as F
from data import ShapeNet_partseg
from util import DiceLoss
from omegaconf import DictConfig
import hydra
from pytorch3d.ops import sample_farthest_points, knn_gather, knn_points
from util import sample_and_group, index_points, query_ball_point
#from datasets.ShapeNet55 import ShapeNet55DataModule
from datasets.ModelNet40Ply2048 import ModelNet40Ply2048DataModule
#from torch.utils.flop_counter import FlopCounterMode
from pthflops import count_ops
import matplotlib.pyplot as plt
import numpy as np


torch.random.manual_seed(0)
np.random.seed(0)

#from pointnet2_ops import pointnet2_utils
#from util import sample_and_group 
#from math import sqrt


#crossatt = CrossAttention(128,64,64,3)
#x = torch.randn(1, 10, 128)

#output, attn = crossatt(x)

#print(attn)
#hard = torch.argmax(attn, dim=-1)
#print(hard)
#print(output.shape)

#gumbatt = F.gumbel_softmax(torch.log(attn), hard=True, tau=10e-8)
#print(gumbatt+attn)

#shapenet = ShapeNet_partseg()
#print(shapenet[0][0].shape)


#diceloss = DiceLoss()
#target = torch.zeros(16, 2, 20)

#input = torch.randn(16, 2, 20)
#input = target+torch.rand_like(target)*1000

#print(diceloss(input, target))
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #shapenet = ShapeNet55DataModule()
    #shapenet.setup()    
    #print(shapenet.train_dataset[0][0].shape)
    device = 'cuda:0'
    modelnet = ModelNet40Ply2048DataModule()
    modelnet.setup()
    data = torch.Tensor(modelnet.modelnet_train[0][0]).to(device).unsqueeze(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0,:,0].cpu().numpy(), data[0,:,1].cpu().numpy(), data[0,:,2].cpu().numpy(), color='blue', alpha=0.1)
    plt.savefig('figures/pointcloud.png')
    #print(data.shape)
    #data = torch.randn(64, 3, 2048).to(device)
    print(data.shape)
    cfg.nclasses = 40
    cfg.train.adaptive.is_adaptive = False
    #model = Pct(cfg, output_channels=cfg.nclasses).to(device)
    sampled_points, sampled_indexes = sample_farthest_points(data, K=128, random_start_point=False)
    print(sampled_points.shape, sampled_indexes.shape)
    knn = knn_points(sampled_points, data, K=32, return_nn=True)
    print(knn.knn.shape)
    knn_pts = knn.knn
    #grouped_points = knn_gather(data, knn_idx)
    #print(grouped_points.shape)
    patch1 = knn_pts[0,1,:,:].cpu().numpy()
    patch2 = knn_pts[0,13,:,:].cpu().numpy()
    patch3 = knn_pts[0,22,:,:].cpu().numpy()
    #print(patch1.shape)
    ax.scatter(patch1[:,0], patch1[:,1], patch1[:,2], color='red')
    ax.scatter(patch2[:,0], patch2[:,1], patch2[:,2], color='green')
    ax.scatter(patch3[:,0], patch3[:,1], patch3[:,2], color='orange')

    plt.savefig('figures/patch.png')


    plt.clf()
    plt.cla()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sampled_points[0,:,0].cpu().numpy(), sampled_points[0,:,1].cpu().numpy(), sampled_points[0,:,2].cpu().numpy(), color='black', alpha=0.5)
    plt.savefig('figures/sampled.png')
    ax.scatter(patch1[:,0], patch1[:,1], patch1[:,2], color='red')
    ax.scatter(patch2[:,0], patch2[:,1], patch2[:,2], color='green')
    ax.scatter(patch3[:,0], patch3[:,1], patch3[:,2], color='orange')
    plt.savefig('figures/patch_sampled.png')
    


    #print(model(data))
    #count_ops(model, data)

    #cfg.train.adaptive.is_adaptive = True
    #model_ada = Pct(cfg, output_channels=cfg.nclasses).to(device)


    #count_ops(model_ada, data)




if __name__ == "__main__":

    torch.fx.wrap("sample_and_group")
    main()
