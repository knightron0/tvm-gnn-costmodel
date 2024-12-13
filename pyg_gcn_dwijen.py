import os.path as osp
import os

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from tqdm import tqdm
import json
import time
from torch import autograd
from torch.profiler import profile, record_function, ProfilerActivity


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TVMDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.processed_dir = osp.join(root, 'processed')
        self.processed_files = [osp.join(self.processed_dir, f) 
                                for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
        self.num_graphs = len(self.processed_files)
        self.transform = transform
        super().__init__()

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        path = self.processed_files[idx]
        data = torch.load(path)
        if self.transform:
            data = self.transform(data)
        return data

def create_dataloaders(dataset, batch_size, test_split_ratio=0.1, num_workers=16):
    # Shuffle dataset indices
    indices = torch.randperm(len(dataset))
    n_test = int(len(dataset) * test_split_ratio)
    
    # Split dataset
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Create DataLoaders
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader

# Dataset path
dataset_path = "/scratch/gilbreth/mangla/gnn_dataset"

# Initialize dataset
tvm_dataset = TVMDataset(root=dataset_path)
print(f"Loaded dataset with {len(tvm_dataset)} graphs.")

# Batch size and workers
batch_size = 384
num_workers = min(96, os.cpu_count() // 2)  # Dynamically adjust workers based on available CPUs

# Create DataLoaders
print("Creating data loaders...")
train_loader, test_loader = create_dataloaders(tvm_dataset, batch_size, num_workers=num_workers)
print("Data loaders created.")


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(20, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()

print("Moving model to device...")
model = model.to(device)
print("Model moved to device.")
print("Initializing optimizer...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
print("Optimizer initialized.")

def train(model, epoch):
    model.train()

    loss_all = 0
    print(f"Starting training epoch {epoch}...")
    for data in tqdm(train_loader, desc=f'Training epoch'):
        data = data.cuda()
        #model = model.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.huber_loss(output.squeeze(), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    print(f"Training epoch {epoch} completed.")
    return loss_all / len(train_dataset)

def test(model, loader):
    model.eval()

    total_mse = 0
    total_mae = 0
    total_huber = 0
    total_y = 0
    total_y_squared = 0
    total_count = 0
    min_y = float('inf')
    max_y = float('-inf')

    i = 0
    print("Starting testing...")
    for data in tqdm(loader, desc="Testing"):
        data = data.cuda()
        #model.to(device)
        pred = model(data)
        pred = pred.squeeze()

        mse = F.mse_loss(pred, data.y)
        huber = F.huber_loss(pred, data.y)

        total_mse += mse.item() * data.num_graphs
        total_huber += huber.item() * data.num_graphs
        
        mae = F.l1_loss(pred, data.y)
        total_mae += mae.item() * data.num_graphs
        
        total_y += torch.sum(data.y)
        total_y_squared += torch.sum(data.y ** 2)
        total_count += data.num_graphs

        batch_min = torch.min(data.y).item()
        batch_max = torch.max(data.y).item()
        min_y = min(min_y, batch_min)
        max_y = max(max_y, batch_max)
        
        i += 1
    print("Testing completed.")
    n = len(loader.dataset)
    mse = total_mse / n
    mae = total_mae / n
    huber = total_huber / n

    y_mean = total_y / total_count
    y_var = (total_y_squared / total_count) - (y_mean ** 2)
    y_std = torch.sqrt(y_var)

    r2_den = total_y_squared - (total_y ** 2) / total_count
    r2 = 1 - (total_mse / r2_den)

    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'Huber': huber,
        'R2': r2,
        'Y_Mean': y_mean.item(),
        'Y_StdDev': y_std.item(),
        'Y_Min': min_y,
        'Y_Max': max_y,
        'Y_Range': max_y - min_y
    }

save_dir = "/scratch/gilbreth/mangla/gnn_models/model_checkpoints"
os.makedirs(save_dir, exist_ok=True)

print("Starting test evaluation...")
test_acc = test(model, test_loader)
print("Test evaluation completed.")
print("Original Test Acc", test_acc)

for epoch in range(1, 3):
    loss = train(model, epoch)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}')
    print(f'Test Acc: {test_acc}')
    
    checkpoint_path = os.path.join(save_dir, f'model_epoch_h100_norm_{epoch}.pt')
    print(f"Saving model checkpoint for epoch {epoch}...")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved model checkpoint to {checkpoint_path}')
