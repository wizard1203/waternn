import os

import numpy as np

def read_sample(sample_file):
    """Returns the sample_file.

    Args:
        sample_file (file): The file of the sample.

    Returns:
        a data sample

    """ 
    with open(file, 'r') as f:
        lines = f.readlines()

        label = int(float(lines[0].split(',')[0].strip()))
        datas = list()
        for line in lines[1:] :
            line = [float(item) for item in line.split(',')]
            datas.append(line)
    f.close()        
    return label, datas

class WaterDataset:

    def __init__(self, data_dir, split='train'):
        """
        Args:
            split:
        
        """
        self.id_list_file = os.listdir(os.path.join(data_dir, split))

    def __len__(self):
        return len(self.id_list_file)

    def get_example(self, i):
        """Returns the i-th sample.

        Args:
            i (int): The index of the sample_files.

        Returns:
            a data sample

        """
        # Load a sample
        sample_file = id_list_file[i]

        label, datas = read_sample(sample_file)

        return label, datas

    __getitem__ = get_example


