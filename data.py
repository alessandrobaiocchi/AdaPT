import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch



def download(folder, url):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, folder)):
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_modelnet(partition):
    download('modelnet40_ply_hdf5_2048', 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

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

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ShapeNet_partseg(Dataset):
    def __init__(self, partition="train"):
        self.alldata = []
        self.alllabels = []
        self.datapath = "data/shapenet_partseg/%s_data/" % partition
        self.labelpath = "data/shapenet_partseg/%s_label/" % partition
        i = 0
        maxlab = 0
        for root, dirs, files in os.walk(self.datapath):
            for file in files:
                filepath = os.path.join(root,file)
                data = np.genfromtxt(filepath, delimiter=' ')
                self.alldata.append(torch.from_numpy(data))
                label = np.genfromtxt(filepath.replace("_data", "_label").replace("pts", "seg"), delimiter=' ')
                if np.max(label)>maxlab: maxlab = np.max(label)
                self.alllabels.append(torch.from_numpy(label)+i)
            i += maxlab
        print(i)
    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, item):
        return self.alldata[item], self.alllabels[item]

if __name__ == '__main__':
    train = ShapeNet_partseg()
    test = ShapeNet_partseg('test')
    
    print(train[5000][0].shape, train[5000][1].shape)
    print(train[5000][0], train[5000][1])
    
    #for data, label in train:
    #    print(data.shape)
    #    print(label.shape)
