import argparse
import os
import torch
import random
import numpy as np

from work_process import work_process

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Time series classification')

# basic config
parser.add_argument('--model', type=str, default='MLP', help='model name, options: [MLP, CNN]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# data loader
parser.add_argument('--data', type=str, default='UCR', help='dataset')
parser.add_argument('--data_path', type=str, default='./data/UCRArchive_2018', help='root path of the data file')
parser.add_argument('--sub_data', type=str, default='ACSF1', help='sub-data file name')

# evaluation
parser.add_argument('--eva_store_path', type=str, default='./evaluation/UCR', help='evaluation store root path')

# optimization
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--learning_decay', type=bool, default=True, help='Does learning rate decay?')

# device
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

args = parser.parse_args()

# device check
if args.use_gpu and not torch.cuda.is_available():
    raise ValueError("Requiring GPU, but this platform has no. Set --use_gpu False and try again.")
elif args.use_gpu and args.gpu_id >= torch.cuda.device_count():
    raise ValueError("Requires an unexisting GPU, check --gpu_id and try again.")

if args.use_gpu:
    args.device = torch.device('cuda:'+str(args.gpu_id))
else:
    args.device = torch.device('cpu')

print(args)

work_process(args)

torch.cuda.empty_cache()

print('Done')