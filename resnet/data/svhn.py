from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import numpy as np
import tensorflow as tensorflow

from resnet.data import svhn_input
from resnet.utils import logger


log = logger.get()


class SVHNDataset():

    def __init__(self,
                 folder,
                 split,
                 num_fold=10,
                 fold_id=0,
                 data_aug=False,
                 whiten=False,
                 div255=False):

        self.split = split
        self.data = svhn_input.read_SVHN(folder)
        num_ex = 73257
        self.split_idx = np.arange(num_ex)
        rnd = np.random.RandomState(0)
        rnd.shuffle(self.split_idx)
        num_valid = int(np.ceil(num_ex / num_fold))
        valid_start = fold_id * num_valid
        valid_end = min((fold_id + 1) * num_valid, num_ex)
        self.valid_split_idx = self.split_idx[valid_start:valid_end]
        self.train_split_idx = np.concatenate(
            [self.split_idx[:valid_start], self.split_idx[valid_end:]])

    def get_size(self):
    	if self.split == "train":
    		return 73257
    	elif self.split == "traintrain":
    		return np.ceil(73257*0.9)
    	elif self.split == "trainval":
    		return np.ceil(73257*0.1)
    	else:
    		return 26032

    def get_batch_idx(self, idx):
        if self.split == "train":
            result = {
                "img": self.data["train_img"][idx],
                "label": self.data["train_label"][idx]
            }
        elif self.split == "traintrain":
            result = {
                "img": self.data["train_img"][self.train_split_idx[idx]],
                "label": self.data["train_label"][self.train_split_idx[idx]]
            }
        elif self.split == "trainval":
            result = {
                "img": self.data["train_img"][self.valid_split_idx[idx]],
                "label": self.data["train_label"][self.valid_split_idx[idx]]
            }
        else:
            result = {
                "img": self.data["test_img"][idx],
                "label": self.data["test_label"][idx]
            }
        return result
