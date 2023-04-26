from model import CrossAttention, Pct_nogroup
import torch
import torch.nn.functional as F
from data import ShapeNet_partseg
from util import DiceLoss
from omegaconf import DictConfig
import hydra



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
    model = Pct_nogroup(cfg)
    #print(model)
    x = torch.randn(1, 3, 1024)
    out, refs = model(x)
    print(out.shape, refs.shape)

if __name__ == "__main__":
    main()
