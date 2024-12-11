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
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return data.to(device)

tvm_dataset = TVMDataset(root="/scratch/gilbreth/mangla/gnn_dataset")
tvm_dataset = tvm_dataset.shuffle()
print("len tvm", len(tvm_dataset))
n = len(tvm_dataset) // 10
test_dataset = tvm_dataset[:n]
train_dataset = tvm_dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=192)
train_loader = DataLoader(train_dataset, batch_size=192)
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


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(Net())
    device = torch.device('cuda')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def train(epoch):
    model.train()

    loss_all = 0
    for data in tqdm(train_loader, desc=f'Training epoch'):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.huber_loss(output.squeeze(), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    total_mse = 0
    total_mae = 0
    total_r2_num = 0
    total_r2_den = 0
    y_mean = torch.mean(torch.cat([data.y for data in loader]))

    i = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        pred = pred.squeeze()

        mse = F.mse_loss(pred, data.y)

        total_mse += mse.item() * data.num_graphs
        
        mae = F.l1_loss(pred, data.y)
        total_mae += mae.item() * data.num_graphs
        
        r2_num = torch.sum((pred - data.y) ** 2)
        r2_den = torch.sum((data.y - y_mean) ** 2)
        total_r2_num += r2_num.item()
        total_r2_den += r2_den.item()
        i += 1

    n = len(loader.dataset)
    mse = total_mse / n
    mae = total_mae / n
    r2 = 1 - (total_r2_num / total_r2_den)

    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'R2': r2
    }


save_dir = "/scratch/gilbreth/mangla/gnn_models/model_checkpoints"
os.makedirs(save_dir, exist_ok=True)

test_acc = test(test_loader)
print("Original Test Acc", test_acc)
for epoch in range(1, 201):
    loss = train(epoch)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}')
    print(f'Test Acc: {test_acc}')
    
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved model checkpoint to {checkpoint_path}')