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


class TVMDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        
        processed_files = [f for f in os.listdir(osp.join(root, 'processed')) if f.endswith('.pt')]
        self.num_graphs = len(processed_files)
        
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return [f for f in os.listdir(osp.join(self.root, 'processed')) if f.endswith('.pt')]

    @property
    def processed_paths(self):
        return [osp.join(self.processed_dir, f) for f in self.processed_file_names]

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data

tvm_dataset = TVMDataset(root="/scratch/gilbreth/mangla/gnn_dataset")
print("Shuffling dataset...")
tvm_dataset = tvm_dataset.shuffle()
print("Dataset shuffled.")
print("len tvm", len(tvm_dataset))
n = len(tvm_dataset) // 10
test_dataset = tvm_dataset[:n]
train_dataset = tvm_dataset[n:]
print("Creating test loader...")
test_loader = DataLoader(test_dataset, batch_size=384, num_workers=96)
print("Test loader created.")
print("Creating train loader...")
train_loader = DataLoader(train_dataset, batch_size=384, num_workers=96)
print("Train loader created.")


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
        x = F.sigmoid(self.lin3(x))

        return x


# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = torch.nn.DataParallel(Net())
#     device = torch.device('cuda')
# else:
#     print("Using a single GPU or CPU for training.")
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
    for data in train_loader:
        print(f"Number of graphs in the batch: {data.num_graphs}")
        print(f"Size of feature matrix x: {data.x.size()}")
        print(f"Size of edge index: {data.edge_index.size()}")
        print(f"Size of batch vector: {data.batch.size()}")
        break

    
    for data in tqdm(train_loader, desc=f'Training epoch'):
        data = data.to(device)
        model = model.to(device)
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
    total_y = 0
    total_y_squared = 0
    total_count = 0

    i = 0
    print("Starting testing...")
    for data in tqdm(loader, desc="Testing"):
        data = data.to(device)
        model.to(device)
        pred = model(data)
        pred = pred.squeeze()

        mse = F.mse_loss(pred, data.y)

        total_mse += mse.item() * data.num_graphs
        
        mae = F.l1_loss(pred, data.y)
        total_mae += mae.item() * data.num_graphs
        
        total_y += torch.sum(data.y)
        total_y_squared += torch.sum(data.y ** 2)
        total_count += data.num_graphs
        
        i += 1
    print("Testing completed.")
    n = len(loader.dataset)
    mse = total_mse / n
    mae = total_mae / n

    y_mean = total_y / total_count
    r2_den = total_y_squared - (total_y ** 2) / total_count
    r2 = 1 - (total_mse / r2_den)

    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'R2': r2
    }


save_dir = "/scratch/gilbreth/mangla/gnn_models/model_checkpoints"
os.makedirs(save_dir, exist_ok=True)


# print("Starting test evaluation...")
# test_acc = test(test_loader)
# print("Test evaluation completed.")
# print("Original Test Acc", test_acc)
for epoch in range(1, 201):
    with autograd.detect_anomaly():
        loss = train(model, epoch)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}')
        print(f'Test Acc: {test_acc}')
    
    checkpoint_path = os.path.join(save_dir, f'model_epoch_h100_{epoch}.pt')
    print(f"Saving model checkpoint for epoch {epoch}...")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved model checkpoint to {checkpoint_path}')