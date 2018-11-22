import torch as t
from water_dataset import WaterDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
# from data import util
import numpy as np
from config import opt
from torch.utils.data import Dataset
import os

class Transform(object):
    """
    transform the data

    Args:
        in_data

    """
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, in_data):
        in_data = (in_data - self.mean) / self.std

        return in_data


class TrainDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.db = WaterDataset(config.data_dir, split=split)
        self.tsf = Transform(config.mean, config.std)

    def __getitem__(self, idx):
        label, datas = self.db.get_example(idx)
        label = t.from_numpy(np.array(label))
        datas = np.array(datas)
        datas = self.tsf(datas)
        datas = t.from_numpy(datas)
        # TODO: check whose stride is negative to fix this instead copy all
        

        return label, datas

    def __len__(self):
        return len(self.db)


class TestDataset(Dataset):
    def __init__(self, config, split='test'):
        self.config = config
        self.db = WaterDataset(config.data_dir, split=split)
        self.tsf = Transform(config.mean, config.std)

    def __getitem__(self, idx):
        label, datas = self.db.get_example(idx)
        label = t.from_numpy(np.array(label))
        datas = np.array(datas)
        datas = self.tsf(datas)
        datas = t.from_numpy(datas)
        # TODO: check whose stride is negative to fix this instead copy all

        return label, datas

    def __len__(self):
        return len(self.db)
