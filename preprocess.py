import os.path as osp
import torch
import json
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from gensim.models import fasttext
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = fasttext.FastText.load("/scratch/gilbreth/mangla/ast-models/fasttext_embed.model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def build_embedding(node):
    return model.wv[node[0]].tolist() + [node[2], node[3]]

def process_batch_graphs(batch_graphs, cost_min, cost_max):
    try:
        batch_data = []
        
        for g_i in batch_graphs:
            g_info = g_i['graph']
            cost = g_i['cost']
            normalized_cost = (cost - cost_min) / (cost_max - cost_min)
            
            num_nodes = len(g_info)
            edge_index = []
            node_features = []
            
            for node_id, node_info in g_info.items():
                node_id = int(node_id)
                neighbors = node_info.get("neighbors", [])
                embedding = build_embedding(node_info.get("data", "None"))
                node_features.append(embedding)
                edge_index.extend([[node_id, neighbor] for neighbor in neighbors])
            
            node_features = torch.tensor(node_features, dtype=torch.float, device=device)
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
            
            data = Data(
                x=node_features,
                edge_index=edge_index.t().contiguous(),
                y=torch.tensor([normalized_cost], dtype=torch.float, device=device)
            )
            
            batch_data.append(data)
        
        return batch_data
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return None

def preprocess_graphs(raw_file_names, output_dir, batch_size=16):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # First pass: calculate cost statistics
    logger.info("Calculating cost statistics...")
    all_costs = []
    saved_data = []
    infoz = []
    for raw_file in tqdm(raw_file_names, desc="Reading costs"):
        with open(raw_file, 'r') as f:
            cost_data = f.read().split('][')
            saved_data = cost_data
            for i, chunk in tqdm(enumerate(cost_data), desc=f"Processing chunks in {raw_file}", leave=False):
                indiv_info = '' if i == 0 else '['
                indiv_info += chunk 
                indiv_info += ']' if i != len(cost_data) - 1 else ''
                infoz.append(indiv_info)
                json_g = json.loads(indiv_info)
                costs = [graph['cost'] for graph in json_g if graph['cost'] < 2]
                all_costs.extend(costs)
    
    costs = torch.tensor(all_costs, device=device).float()
    cost_min = costs.min().item()
    cost_max = costs.max().item()
    del all_costs, costs
    
    processed_count = 0
    current_batch = []
    
    logger.info("Processing and saving graphs...")
    for raw_file in tqdm(raw_file_names, desc="Processing files"):
        with open(raw_file, 'r') as f:
            f.seek(0)
            data = f.read().split('][')
            for i, chunk in tqdm(enumerate(data), desc=f"Processing chunks in {raw_file}", leave=False):
                info = '' if i == 0 else '['
                info += chunk
                info += ']' if i != len(data) - 1 else ''
                g = json.loads(info)

                filtered_graphs = [graph for graph in g if graph['cost'] < 2]
                
                for graph in filtered_graphs:
                    current_batch.append(graph)
                    
                    if len(current_batch) >= batch_size:
                        processed_batch = process_batch_graphs(current_batch, cost_min, cost_max)
                        if processed_batch:
                            for graph_data in processed_batch:
                                processed_count += 1
                                output_file = osp.join(output_dir, f'data_{processed_count}.pt')
                                graph_data = graph_data.cpu()
                                torch.save(graph_data, output_file)
                        current_batch = []
                        
                        if processed_count % (batch_size * 10) == 0:
                            torch.cuda.empty_cache()
    
    # Process remaining graphs
    if current_batch:
        processed_batch = process_batch_graphs(current_batch, cost_min, cost_max)
        if processed_batch:
            for data in processed_batch:
                processed_count += 1
                output_file = osp.join(output_dir, f'data_{processed_count}.pt')
                data = data.cpu()
                torch.save(data, output_file)
    
    logger.info("Preprocessing completed")
    return processed_count

if __name__ == "__main__":
    raw_files = [
        "testtuning_65.json.graph.json",
        "testtuning_27.json.graph.json",
        "testtuning_36.json.graph.json",
        # "testtuning_48.json.graph.json",
        # "testtuning_55.json.graph.json",
        # "testtuning_60.json.graph.json",
        # "testtuning_70.json.graph.json",
        # "finetuning_69.json.graph.json",
        # "finetuning_35.json.graph.json",
        # "finetuning_120.json.graph.json",
        # "0_test_67.json.graph.json",
        # "0_test_43.json.graph.json",
        # "0_test_11.json.graph.json"
    ]
    
    # Add full path to raw files
    raw_dir = "/scratch/gilbreth/mangla/gnn_dataset/raw"  # Update this path
    raw_files = [osp.join(raw_dir, f) for f in raw_files]
    
    output_dir = "/scratch/gilbreth/mangla/gnn_dataset/processed"  # Update this path
    
    # Process with GPU batching
    num_processed = preprocess_graphs(raw_files, output_dir, batch_size=32)
    print(f"Successfully processed {num_processed} graphs")
