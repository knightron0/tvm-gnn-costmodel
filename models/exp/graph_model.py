
""" Cost model based on GNN """

import numpy as np
import time
import logging

# from ..graph_feature import get_graph_from_measure_pairs, get_graph_from_states, get_graph_from_file
# from .cost_model import PythonBasedModel

import dgl
import collections
import torch
import dgl.nn as dglnn
import torch.nn.functional as F
import dgl.function as fn
import math
import copy
import matplotlib.pyplot as plt
import torch as th
from itertools import chain
import json

logger = logging.getLogger('auto_scheduler')

Edge = collections.namedtuple("Edge", ["src", "dst", "feature"])
Node = collections.namedtuple("Node", ["node_type", "id", "feature"])


def compute_rmse(preds, labels):
    """Compute RMSE (Rooted mean square error)"""
    return np.sqrt(np.mean(np.square(preds - labels)))

class LambdaRankLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1., device=None):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cpu'
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss

class PythonBasedModel():
    """Base class for cost models implemented in python"""

    def __init__(self):
        def update_func(inputs, results):
            self.update(inputs, results)

        def predict_func(task, states, return_ptr):
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(len(states),))
            array_wrapper[:] = self.predict(task, states)

        def predict_stage_func(task, states, return_ptr):
            ret = self.predict_stages(task, states)
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=ret.shape)
            array_wrapper[:] = ret

        self.__init_handle_by_constructor__(
            _ffi_api.PythonBasedModel, update_func, predict_func, predict_stage_func
        )

    def update(self, inputs, results):
        """Update the cost model according to new measurement results (training data).

        Parameters
        ----------
        inputs : List[auto_scheduler.measure.MeasureInput]
            The measurement inputs
        results : List[auto_scheduler.measure.MeasureResult]
            The measurement results
        """
        raise NotImplementedError

    def predict(self, task, states):
        """Predict the scores of states

        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        states : List[State]
            The input states

        Returns
        -------
        scores: List[float]
            The predicted scores for all states
        """
        raise NotImplementedError

    def predict_stages(self, task, states):
        """Predict the scores of all stages in states. This is the breakdown version of `predict`.

        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        states : List[State]
            The input states

        Returns
        -------
        scores: List[float]
            The predicted scores for all stages in all states in the packed format

        Note
        ----
        For faster data copy between c++ and python, the python part returns scores in a
        single flatten array using a packed format. The c++ part then unpacks the flatten array.

        The packed format is:
        {
          float  scores[N];                 // scores[i] is the score for states[i].
          int    n_stage_0;                 // the number of stages in states[0]
          float  stage_scores_0[[n_stage_0] // the scores for all stages in states[0]
          int    n_stage_1;                 // the number of stages in states[1]
          float  stage_scores_1[n_stage_1]; // the scores for all stages in states[1]
          ...
          int    n_stage_i;                 // the number of stages in states[i]
          float  stage_scores_1[n_stage_i]; // the scores for all stages in states[i]
          ...  // until i == N - 1
        }
        To implement this format, we also store int as float, so we can store all numbers
        into a single float array.
        """
        raise NotImplementedError

class GNN(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super(GNN, self).__init__()
        # self.msg = torch.nn.Linear(node_dim + edge_dim, hidden_dim)
        self.conv1 = dglnn.TAGConv(node_dim, hidden_dim)
        self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.TAGConv(hidden_dim, hidden_dim)
        self.conv4 = dglnn.TAGConv(hidden_dim, hidden_dim)
        self.conv5 = dglnn.TAGConv(hidden_dim, hidden_dim)
        self.predict = torch.nn.Linear(hidden_dim, 1)

    # def message_func(self, edges):
    #     return {'mid': F.relu(self.msg(torch.cat([edges.src['fea'], edges.data['fea']], 1)))}

    def forward(self, g):
        with g.local_scope():
            nans = (torch.where(torch.isnan(g.ndata['fea']))[1])
            #print(len(nans))

            g.ndata['fea'] = torch.nan_to_num(g.ndata['fea'])

            # g.update_all(self.message_func, fn.sum('mid', 'h_neigh'))

            # g.ndata['h_neigh'] = torch.nan_to_num(g.ndata['h_neigh'])

            h1 = F.relu(self.conv1(g, g.ndata['fea']))
            #print(h.size())
            h2 = F.relu(self.conv2(g, h1))
            h3 = F.relu(self.conv3(g, h2))
            h4 = F.relu(self.conv4(g, h3))
            h5 = F.relu(self.conv5(g, h4))
            #print(h.size())
            #h = F.relu(self.conv3(g, h))
            #print(h.size())

            g.ndata['h'] = h5
            # pool mean over graph wrt embedding h
            hg = dgl.mean_nodes(g, 'h')
            # predict based on embedding
            hg = self.predict(hg)

            #print(torch.where(torch.isnan(hg)))

            #assert(False)
            return hg


class GraphModel(PythonBasedModel):

    def __init__(self):
        self.params = {
            'batch_size': 256,
            'itr_num': 500,
            'lr':  0.01,
            'hidden_dim': 64,
            'node_fea': 20
            # 'edge_fea': 4,
        }

        self.GNN = None

        super().__init__()

        # measurement input/result pairs
        self.inputs = []
        self.results = []
        self.few_shot_learning = "base_only"
        self.loss_func = torch.nn.MSELoss() 

    def register_new_task(self, task):
        pass
        #workload_key = str(task.workload_key)
        #self.workload_embed_dict[workload_key] = get_workload_embedding(workload_key)

    def fit_base(self, train_set, valid_set=None, valid_train_set=None):
        if self.few_shot_learning == "local_only":
            self.base_model = None
        else:
            self.base_model = self._fit_a_model(train_set, valid_set, valid_train_set)

    def _fit_a_model(self, train_set, valid_set=None, valid_train_set=None):
        print("Fit a GNN. Train size: %d" % len(train_set))

        pairs = list(train_set.features.values())

        # modify this based on structure of train_set
        idx = np.random.permutation(len(pairs))
        train_pairs = list(chain(*[[build_graph(x) for x in pairs[i]] for i in idx]))
        print("Sample Graph: ", train_pairs[0])
        train_batched_graphs, train_batched_labels = create_batch(train_pairs, self.params['batch_size'])

        self.GNN = GNN(self.params['node_fea'], self.params['hidden_dim']).float().cuda()
        opt = torch.optim.SGD(self.GNN.parameters(), lr=self.params['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
        
        n = len(train_batched_graphs)

        self.loss_func = torch.nn.MSELoss()

        print('Learning rate: {} batch size: {}'.format(self.params['lr'], self.params['batch_size']))

        for epoch in range(self.params['itr_num']):
            total_loss = 0
            preds = []
            labels = []
            tic = time.time()
            for i in range(n):
                opt.zero_grad()
                prediction = self.GNN.forward(train_batched_graphs[i])
                #loss = torch.sqrt(loss_func(prediction, train_batched_labels[i].unsqueeze(1).cuda())) #loss_func(prediction, torch.log(train_batched_labels[i].unsqueeze(1).cuda()))
                loss = self.loss_func(prediction.view(-1,1), train_batched_labels[i].view(-1,1).cuda())
                total_loss += loss.detach().item() * self.params['batch_size']
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.GNN.parameters(), 10)
                #print(loss)
                #print(prediction.size())
                #print(train_batched_labels[i].size())
                #print('=======')
                #print(list(self.GNN.conv1.parameters()))
                #for x in list(self.GNN.conv1.parameters()): print(x.grad)
                opt.step()
                #print(list(self.GNN.conv1.parameters()))
                pred = prediction.squeeze().cpu().tolist()
                label = train_batched_labels[i].squeeze().cpu().tolist()
                if type(pred) is float: pred = [pred]
                if type(label) is float: label = [label]
                preds = preds + pred
                labels = labels + label
            epoch_loss = compute_rmse(np.exp(np.array(preds)), np.array(labels))
            #print("Time spent in last epoch: %.2f" % (time.time() - tic))
            if epoch % 100 == 0:
                scheduler.step()
            if epoch % 10 == 0:
                print('Epoch {} | loss {:.4f}'.format(epoch, epoch_loss))

        return self.GNN

    # what is dataset 
    def predict(self, dataset):
        if self.few_shot_learning in ["base_only", "fine_tune_mix_task", "fine_tune_per_task", "MAML"]:
            return self._predict_a_dataset(self.base_model, dataset)
        elif self.few_shot_learning in ["local_only_mix_task", "local_only_per_task"]:
            ret = {}
            for task in dataset.tasks():
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = local_preds
            return ret
        elif self.few_shot_learning in ["plus_mix_task", "plus_per_task"]:
            base_preds = self._predict_a_dataset(self.base_model, dataset)
            ret = {}
            for task in dataset.tasks():
                if task not in self.local_model and self.few_shot_learning == "plus_mix_task":
                    self.local_model[task] = list(self.local_model.values())[0]
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = base_preds[task] + local_preds
            return ret
        else:
            raise ValueError("Invalid few show learing: " + self.few_shot_learning)

    def _predict_a_dataset(self, model, dataset):
        ret = {}
        for task, features in dataset.features.items():
            ret[task] = self._predict_a_task(model, task, features)
        return ret

    def build_graph(pair):  # (src_nodes, dest_nodes, node_fea)
        #print(len(pair))
        #print(len(pair[0]))
        if len(pair) == 3: (src_cur, dst_cur, edge_fea), node_fea, _ = pair
        else: (src_cur, dst_cur, edge_fea), node_fea = pair
        g = dgl.graph((th.tensor(src_cur).cuda(), th.tensor(dst_cur).cuda()))
        # g.edata['fea'] = th.tensor(edge_fea).float().cuda()
        g.ndata['fea'] = node_fea.cuda()
        return g

    def _predict_a_task(self, model, task, features):
        graphs = [build_graph(x) for x in features]
        tic = time.time()
        batched_graphs = dgl.batch(graphs)
        preds = model(batched_graphs).squeeze().tolist()
        print("prediction time: %.2f" % (time.time() - tic))
        return list(np.exp(np.array(preds)))

    def save(self, file_name: str):
        print("saving to: "+ file_name)
        torch.save(self.GNN.state_dict(), file_name)

    def load(self, file_name: str):
        print("loading from: "+file_name)
        self.GNN = GNN(self.params['node_fea'], self.params['hidden_dim']).float()
        self.GNN.load_state_dict(torch.load(file_name))


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def create_batch(graph_pairs, batch_size):
    batched_graphs = []
    batched_labels = []
    for i in range(0, len(graph_pairs), batch_size):
        if i+batch_size >= len(graph_pairs):
            g, l = collate(graph_pairs[i:])
            batched_graphs.append(g)
            batched_labels.append(l)
        else:
            g, l = collate(graph_pairs[i:i+batch_size])
            batched_graphs.append(g)
            batched_labels.append(l)
    return batched_graphs, batched_labels


def load_dataset(file_path):
    """
    Loads graphs from a JSON file, converts them to DGL format, and extracts node data.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple: (DGLGraph, node_data_dict), where:
            - DGLGraph is the constructed DGL graph.
            - node_data_dict is a dictionary mapping node IDs to their "data".
    """
    with open(file_path, 'r') as f:
        data = f.read().split('][')
        graphs = []
        for i in range(len(data)):
            info = '' if i == 0 else '['
            info += data[i]
            info += ']' if i != len(data) - 1 else ''
            # print(len(info), info[c - diff: c])
            # print('\n' * 5)
            # print(info[c:c + diff])
            try:
                g = json.loads(info)  # Load the list of graphs from the JSON file
            except:
                D = 50000
                print("Error occured while processing " + str(i) + "th index of data")
                print("First 50k chars:")
                print(info[:D])

            graphs.extend(g)

    print("Number of graphs:", len(graphs))

    for g_i in graphs:
        # Initialize the graph
        # print(g_i)
        # continue

        # Prepare node data storage
        node_data_dict = {}

        g_info = g_i['graph']
        cost = g_i['cost']
        num_nodes = len(g_info)
        
        g = dgl.graph(([], []), num_nodes=num_nodes)  # Empty graph with num_nodes nodes

        # Add edges and extract node features
        for node_id, node_info in g_info.items():
            node_id = int(node_id)  # Convert node_id from string to integer
            
            # print(node_info)
            # Extract neighbors and add edges
            neighbors = node_info.get("neighbors", [])
            for neighbor in neighbors:
                # print(node_id, neighbor)
                g.add_edges(node_id, neighbor)
            
            # Extract and store node "data"
            node_data_dict[node_id] = node_info.get("data", [])
        
        # g = dgl.graph((srcs, dests), num_nodes=num_nodes)
        # Convert node data to tensor and store in the graph
        # node_features = [torch.tensor(node_data_dict[node_id], dtype=torch.float32) for node_id in range(num_nodes)]
        # g.ndata['data'] = torch.stack(node_features)

        # Yield the DGL graph and node data dictionary
        yield (g, node_data_dict, cost)


file = '/scratch/gilbreth/mangla/gnn_dataset/testtuning_55.json.graph.json' # './testtuning_60.json.graph.json'
gen = load_dataset(file)

for (g, emb, cost) in gen:
    print("Graph:", g)

    # print(emb)
