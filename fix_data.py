import os
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

def process_file(filepath):
    try:
        import time
        import os.path
        
        mod_time = os.path.getmtime(filepath)
        current_time = time.time()
        
        if current_time - mod_time < 600: 
            return None
            
        data = torch.load(filepath)
        
        if hasattr(data, 'y'):
            # data.y = (data.y * (0.191615 - 0.00000212446)) + 0.00000212446
            data.y = (data.y - 0.00027766588) / 0.0022391088
            
        #     stats['min'] = min(stats['min'], data.y.min().item())
        #     stats['max'] = max(stats['max'], data.y.max().item())
        #     stats['sum'] += data.y.sum().item()
        #     stats['count'] += len(data.y)
        # print(data.y)
        torch.save(data, filepath)
    
    except Exception as e:
        return f"Error processing {filepath}: {str(e)}\n{traceback.format_exc()}"

def main():
    data_dir = "/scratch/gilbreth/mangla/newdataset/val"
    
    pt_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.pt')
    ]
    
    print(f"Found {len(pt_files)} .pt files to process")
    
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_file = {
            executor.submit(process_file, filepath): filepath 
            for filepath in pt_files
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(pt_files)):
            filepath = future_to_file[future]
            try:
                result = future.result()
                # if isinstance(result, dict):
                #     stats = result
            except Exception as e:
                print(f"Exception occurred while processing {filepath}: {str(e)}")
                
    # if stats['count'] > 0:
    #     stats['mean'] = stats['sum'] / stats['count']
        
    # print(f"After transformation statistics:")
    # print(f"Min: {stats['min']}")
    # print(f"Max: {stats['max']}")
    # print(f"Mean: {stats['mean']}")

if __name__ == "__main__":
    main()


# import os
# import torch
# from tqdm import tqdm
# import numpy as np

# def calculate_statistics(data_dir):
#     # Find all .pt files
#     pt_files = [
#         os.path.join(data_dir, f)
#         for f in os.listdir(data_dir)
#         if f.endswith('.pt')
#     ]
    
#     print(f"Found {len(pt_files)} .pt files")
    
#     # Initialize stats
#     all_y_values = []
    
#     # Process each file
#     for filepath in tqdm(pt_files, desc="Processing files"):
#         try:
#             data = torch.load(filepath)
#             y_values = data.y.cpu().numpy()
#             all_y_values.extend(y_values)
#         except Exception as e:
#             print(f"Error processing {filepath}: {str(e)}")
#             continue
    
#     # Convert to numpy array for calculations
#     all_y_values = np.array(all_y_values)
    
#     # Calculate statistics
#     stats = {
#         'min': np.min(all_y_values),
#         'max': np.max(all_y_values),
#         'mean': np.mean(all_y_values),
#         'median': np.median(all_y_values),
#         'std': np.std(all_y_values),
#         'count': len(all_y_values)
#     }
    
#     # Print statistics
#     print("\nDataset Statistics:")
#     print(f"Count: {stats['count']:,}")
#     print(f"Min: {stats['min']:.6f}")
#     print(f"Max: {stats['max']:.6f}") 
#     print(f"Mean: {stats['mean']:.6f}")
#     print(f"Median: {stats['median']:.6f}")
#     print(f"Std Dev: {stats['std']:.6f}")
    
#     return stats

# if __name__ == "__main__":
#     data_dir = "/scratch/gilbreth/mangla/gnn_dataset/processed"
#     calculate_statistics(data_dir)
