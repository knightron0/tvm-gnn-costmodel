import os.path as osp

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
from gensim.models import fasttext

model = fasttext.FastText.load("/scratch/gilbreth/mangla/ast-models/fasttext_embed.model")

def build_embedding(node):
  return model.wv[node[0]].tolist() + [node[2], node[3]]

class TVMDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.all_graphs = []
        for raw_file in self.raw_file_names:
            with open(osp.join(self.raw_dir, raw_file), 'r') as f:
                data = f.read().split('][')
                for i in range(len(data)):
                    info = '' if i == 0 else '['
                    info += data[i]
                    info += ']' if i != len(data) - 1 else ''
                    g = json.loads(info)
                    self.all_graphs.extend(g)
        self.num_graphs = len(self.all_graphs)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["testtuning_65.json.graph.json"]

    @property 
    def processed_file_names(self):
        return [f'data_{i+1}.pt' for i in range(self.num_graphs)]

    @property
    def processed_paths(self):
        return [osp.join(self.processed_dir, f) for f in self.processed_file_names]

    def process(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
        batch_size = 50
        for i in tqdm(range(0, len(self.all_graphs), batch_size)):
            batch_graphs = self.all_graphs[i:i+batch_size]
            
            batch_edge_indices = []
            batch_node_features = [] 
            batch_costs = []
            
            x = i
            for g_i in batch_graphs:
                if osp.exists(self.processed_paths[x]):
                    x += 1
                    continue

                g_info = g_i['graph']
                cost = g_i['cost']
                
                edge_index = []
                node_features = []
                
                num_nodes = len(g_info)
                
                for node_id, node_info in g_info.items():
                    node_id = int(node_id)
                    neighbors = node_info.get("neighbors", [])
                    embedding = build_embedding(node_info.get("data", "None"))
                    node_features.append(embedding)
                    for neighbor in neighbors:
                        edge_index.append([node_id, neighbor])
                
                node_features = torch.tensor(node_features, dtype=torch.float, device=device)
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
                
                data = Data(
                    x=node_features,
                    edge_index=edge_index.t().contiguous(),
                    y=torch.tensor([cost], dtype=torch.float, device=device)
                )
                
                data = data.cpu()
                torch.save(data, self.processed_paths[x])
                x += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return data.to(device)

tvm_dataset = TVMDataset(root="/scratch/gilbreth/mangla/gnn_dataset")
tvm_dataset = tvm_dataset.shuffle()
n = len(tvm_dataset) // 10
test_dataset = tvm_dataset[:n]
train_dataset = tvm_dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=5)
train_loader = DataLoader(train_dataset, batch_size=5)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(20, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 1)
        # self.lin2 = torch.nn.Linear(128, 64)
        # self.lin3 = torch.nn.Linear(64, 1)

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

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin2(x))
        x = self.lin1(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


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

    total_loss = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = F.huber_loss(pred.squeeze(), data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


for epoch in range(1, 201):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
          f'Test Acc: {test_acc:.5f}')