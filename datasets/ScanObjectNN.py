from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ScanObjectNN(Dataset):
    def __init__(self, root, split="training", perturbed=True):
        assert split == "training" or split == "test"
        self.split = split
        file_name = (
            "_objectdataset_augmentedrot_scale75.h5"
            if perturbed
            else "_objectdataset.h5"
        )
        h5_name = Path(root) / (split + file_name)
        with h5py.File(h5_name, mode="r") as f:
            self.data = f["data"][:].astype(np.float32)  # type: ignore
            self.label = f["label"][:].astype(np.int64)  # type: ignore

    def __len__(self):
        return self.data.shape[0]  # type: ignore

    def __getitem__(self, index):
        points = self.data[index]
        label = self.label[index]
        if self.split == 'training':
            points = random_point_dropout(points) # open for dgcnn not for our idea  for all
            points = translate_pointcloud(points)
            np.random.shuffle(points)
        return points, label


class ScanObjectNNDataModule(pl.LightningDataModule):
    """
    size: 14298
    train: 11416
    test: 2882
    """

    def __init__(
        self,
        data_dir: str = "./data/ScanObjectNN",
        split: str = "main_split",
        perturbed: bool = True,
        batch_size: int = 32,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ScanObjectNN(Path(self.hparams.data_dir) / self.hparams.split, split="training", perturbed=self.hparams.perturbed)  # type: ignore
        self.test_dataset = ScanObjectNN(Path(self.hparams.data_dir) / self.hparams.split, split="test", perturbed=self.hparams.perturbed)  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=self.hparams.drop_last,  # type: ignore
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=4,
        )

    @property
    def num_classes(self):
        return 15

    @property
    def label_names(self) -> List[str]:
        return [
            "bag",
            "bin",
            "box",
            "cabinet",
            "chair",
            "desk",
            "display",
            "door",
            "shelf",
            "table",
            "bed",
            "pillow",
            "sink",
            "sofa",
            "toilet",
        ]
