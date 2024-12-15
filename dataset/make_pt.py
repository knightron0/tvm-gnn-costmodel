import os.path as osp
import torch
import json
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from gensim.models import fasttext
from pathlib import Path
import logging
import os
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = fasttext.FastText.load("/scratch/gilbreth/mangla/ast-models/fasttext_embed.model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def build_embedding(node):
    return model.wv[node[0]].tolist() + [node[2], node[3]]

def process_batch_graphs(batch_graphs):
    try:
        batch_data = []
        
        for g_i in batch_graphs:
            g_info = g_i['graph']
            cost = g_i['cost']
            
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
                y=torch.tensor([cost], dtype=torch.float, device=device)
            )
            
            batch_data.append(data)
        
        return batch_data
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return None

def preprocess_graphs(raw_file_names, output_dir, batch_size=384, num_workers=8):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Calculating cost statistics...")
    processed_count = 0
    current_batch = []
    metadata_file = "/scratch/gilbreth/mangla/bigdataset_metadata.txt"
    
    logger.info("Processing and saving graphs...")
    for raw_file in tqdm(raw_file_names, desc="Processing files"):
        with open(raw_file, 'r') as f:
            f.seek(0)
            data = f.readlines()
            for i, chunk in tqdm(enumerate(data), desc=f"Processing chunks in {raw_file}", leave=False):
                graph = json.loads(chunk)
                if graph['cost'] < 2:
                    current_batch.append(graph)
                else:
                    continue
                
                if len(current_batch) >= batch_size:
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        processed_batch = process_batch_graphs(current_batch)
                        if processed_batch:
                            idx = 0
                            metadata_entries = []
                            for graph_data in processed_batch:
                                processed_count += 1
                                output_file = osp.join(output_dir, f'data_{processed_count}.pt')
                                metadata_entries.append(f"{output_file},{current_batch[idx]['workload']}\n")
                                torch.save(graph_data, output_file)
                                idx += 1
                                
                            with open(metadata_file, 'a') as mf:
                                mf.writelines(metadata_entries)
                                
                    current_batch = []
                    
                    if processed_count % (batch_size * 50) == 0:
                        torch.cuda.empty_cache()

    if current_batch:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            processed_batch = process_batch_graphs(current_batch)
            if processed_batch:
                metadata_entries = []
                idx = 0
                for data in processed_batch:
                    processed_count += 1
                    output_file = osp.join(output_dir, f'data_{processed_count}.pt')
                    metadata_entries.append(f"{output_file},{current_batch[idx]['workload']}\n")
                    torch.save(data, output_file)
                    idx += 1
                
                with open(metadata_file, 'a') as mf:
                    mf.writelines(metadata_entries)
    
    logger.info("Preprocessing completed")
    return processed_count

if __name__ == "__main__":
    raw_files = os.listdir("/scratch/gilbreth/dchawra/bigdataset")
    
    raw_dir = "/scratch/gilbreth/dchawra/bigdataset" 
    raw_files = [osp.join(raw_dir, f) for f in raw_files]
    
    output_dir = "/scratch/gilbreth/mangla/bigdataset" 
    
    num_processed = preprocess_graphs(raw_files, output_dir, batch_size=1024, num_workers=8)
    print(f"Successfully processed {num_processed} graphs")
