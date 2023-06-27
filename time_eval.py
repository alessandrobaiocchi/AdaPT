import hydra
from omegaconf import DictConfig
import torch
from pylightning_mods import Lightning_pct_adaptive, Lightning_pct_merger, Lightning_pct
import pytorch_lightning as pl
from datasets.ModelNet40Ply2048 import ModelNet40Ply2048DataModule
from model import Pct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):

    ntokens = [16,32,64,128,192,256,384,512,768,1024,1536,2048,4096,8192]
    time = time_scaling(cfg, adaptive=False, ntokens=ntokens)
    time_ada = time_scaling(cfg, adaptive=True, ntokens=ntokens)
    

    fig = plt.figure()
    plt.plot(ntokens, time, label="fixed")
    plt.plot(ntokens, time_ada, label="adaptive")
    plt.legend()
    
    plt.xlabel("n_tokens")
    plt.ylabel("time (s)")
    plt.savefig("figures/time_scaling.png")
    plt.yscale("log")
    plt.savefig("figures/time_scaling_log.png")



def time_scaling(cfg, adaptive=False, ntokens=[]):
    device = torch.device("cuda")
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 2

    times = np.zeros(len(ntokens))

    cfg.train.adaptive.is_adaptive = adaptive
    
    for n_tokens in tqdm(ntokens):
        
        cfg.train.n_tokens = n_tokens
        #dummy_pc = torch.randn(1, 3, n_tokens).to(device)
        dummy_pc = torch.randn(1, 3, 8192).to(device)
        model = Pct(cfg, 40).to(device).eval()
        model.to(device)
        
        # INIT LOGGERS

        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(1):
            _ = model(dummy_pc)
        # MEASURE PERFORMANCE

        #print("n_tokens:",n_tokens)

        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_pc)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        #print("mean:",mean_syn, " std:",std_syn)
        times[ntokens.index(n_tokens)] = mean_syn
    return times



if __name__ == "__main__":

    main()
