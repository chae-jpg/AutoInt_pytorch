import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from model.AutoInt import AutoInt
from data.dataset import CriteoDataset

# 9000 items for training, 1000 items for valid, of all 10000 items
# Note: The original main.py had Num_train = 9000, but the comment said 900000. 
# I will stick to the code's value of 9000 for now as per the file I read.
Num_train = 9000

# load data
print("Loading data...")
train_data = CriteoDataset('./data', train=True)
loader_train = DataLoader(train_data, batch_size=100,
                          sampler=sampler.SubsetRandomSampler(range(Num_train)))
val_data = CriteoDataset('./data', train=True)
loader_val = DataLoader(val_data, batch_size=100,
                        sampler=sampler.SubsetRandomSampler(range(Num_train, 10000)))

feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print("Feature sizes:", feature_sizes)

# Check for CUDA
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")

# Initialize AutoInt model
model = AutoInt(feature_sizes, 
                embedding_size=16, 
                att_layer_num=3, 
                att_head_num=2, 
                use_cuda=use_cuda, 
                device=device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

print("Starting training...")
model.fit(loader_train, loader_val, optimizer, epochs=50, verbose=True)
