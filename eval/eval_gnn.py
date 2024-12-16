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
import torch.multiprocessing as mp

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class TVMDataset(Dataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
    self.root = root
    self.files = [f for f in os.listdir(self.root) if f.endswith('.pt') and f.startswith('data_')]
    self.num_graphs = len(self.files)
    super().__init__(root, transform, pre_transform, pre_filter)

  def len(self):
    return self.num_graphs

  def get(self, idx):
    data = torch.load(osp.join(self.root, self.files[idx]))
    return data

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


def main():    
    print("Loading test dataset...")
    test_dataset = TVMDataset(root="/scratch/gilbreth/mangla/newdataset/test")

    test_dataset = test_dataset.shuffle()

    print("len test", len(test_dataset))

    print("Creating test loader...")
    test_loader = DataLoader(test_dataset, batch_size=40, num_workers=32, persistent_workers=True)
    print("Test loader created.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net()
    model.load_state_dict(torch.load("/scratch/gilbreth/mangla/gnn_models/model_checkpoints/model_h100new_newdata_norm_h100_69.pt"))

    print("Moving model to device...")
    model = model.to(device)
    print("Model moved to device.")

    total_huber = 0
    for data in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        model.to(device)
        pred = model(data)
        pred = pred.squeeze()

        huber = F.huber_loss(pred, data.y)
        print(pred, data.y)

        total_huber += huber.item() * data.num_graphs

    print("Total Huber Loss:", total_huber / len(test_dataset))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()