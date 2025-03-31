  
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import sys
sys.path.append("./src/")
from dataset import train_val_test_split

from models.tsmixer import TSMixer
from models.moderntcn import ModernTCN

torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)

# data
n_batch, n_channels, n_steps = 2, 3, 5
X = torch.randn((n_batch, n_channels, n_steps))

# model
# model = nn.Linear(5, 4)
# model = TSMixer(d_input=n_steps, d_output=n_steps-1, d_hidden_temporal=6, d_hidden_channel=6, n_input=n_channels)
model = ModernTCN(d_input=n_steps, d_output=n_steps-1, d_hidden=8, n_input=n_channels)

# input/output
print("INPUT")
print(X.shape)

print("OUTPUT")
print(model(X).shape)