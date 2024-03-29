import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from data.UCRArchive_2018.UCR_datasets_name import UNIVARIATE_DATASET_NAMES_2018 as UCR_names


#UCR共有129个子数据集，每个子数据集都是单变量，且子数据集之间无联系
#因此想要训练就需要在不同的子数据集上实例一个dataset
#并按照输入维度初始化模型，因此data_provider应该同时返回input_size这个值
class UCR_data_provider(Dataset):
    def __init__(self, args, dataset_type: str = 'train', znorm: bool = True):
        self.input_size = 0
        self.output_size = 0
        self.data_path = args.data_path
        self.sub_data = args.sub_data
        self.dataset_type = dataset_type
        self.znorm = znorm
        
        self.read_dataset()

    def read_dataset(self):
        if self.dataset_type == 'train':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TRAIN.tsv', sep='\t', header=None)
        elif self.dataset_type == 'test':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TEST.tsv', sep='\t', header=None)
        else:
            raise ValueError("Illegal dataset type.")
        
        ts = df_raw.drop(columns=[0])
        self.input_size = ts.shape[1]#shape指第index维度的维数，DataFrame的成员
        ts.columns = range(ts.shape[1])
        label = df_raw.values[:, 0]
        self.output_size = int(max(label)+1)

        ts = ts.values

        if self.znorm:
            std_ = ts.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            ts = (ts - ts.mean(axis=1, keepdims=True)) / std_

        self.ts = ts
        self.label = label

    def __getitem__(self, index):
        ts = self.ts[index]
        label = self.label[index]
        return ts, label

    def __len__(self):
        return self.label.shape[0]