import os

import numpy as np
import logging
from config import opt
def read_sample(sample_file_path):
    """Returns the sample_file.

    Args:
        sample_file (file): The file of the sample.

    Returns:
        a data sample

    """ 
    with open(sample_file_path, 'r') as f:
        lines = f.readlines()
        # label = int(float(lines[0].split(',')[0].strip()))
        reallabel = int(float(lines[0].split(',')[0].strip()))
        label = opt.labels_dict.index(reallabel)
        datas = list()
        for line in lines[1:] :
            line = [float(item) for item in line.split(',')]
            datas.append(line)
    f.close()        
    return label, datas

def dataprocess(datas):
    """Returns the processed datas.  if input not be uniformed

    Args:
        datas (list): The list of the datas.

    Returns:
        processed datas

    """
    
    for line in datas:
        line[0] = (line[0] - 59.0) * 2.0
        line[1] = (line[1] - 64.0) * 1.9
        line[2] = (line[2] - 25.0) * 0.9
        line[3] = (line[3] - 23.0) * 0.5
    return datas

class WaterDataset:

    def __init__(self, data_dir, split='train'):
        """
        Args:
            split:
        
        """
        self.data_dir = os.path.join(data_dir, split)
        self.list_file = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.list_file)

    def get_example(self, i):
        """Returns the i-th sample.

        Args:
            i (int): The index of the sample_files.

        Returns:
            a data sample

        """
        # Load a sample
        sample_file = self.list_file[i]

        label, datas = read_sample(os.path.join(self.data_dir, sample_file))
        # datas = dataprocess(datas)
        return label, datas

    __getitem__ = get_example


