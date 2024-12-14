import os
import torch
from tqdm import tqdm
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_file_stats(filepath):
    try:
        data = torch.load(filepath)
        return data.x.shape[0]
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def calculate_statistics(data_dir):
    pt_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.pt')
    ]
    
    print(f"Found {len(pt_files)} .pt files")
    
    from multiprocessing import Pool, cpu_count
    num_processes = cpu_count()
    
    with Pool(num_processes) as pool:
        graph_sizes = list(tqdm(
            pool.imap(process_file_stats, pt_files),
            total=len(pt_files),
            desc=f"Processing files using {num_processes} processes"
        ))
    
    graph_sizes = [size for size in graph_sizes if size is not None]
    graph_sizes = np.array(graph_sizes)
    
    stats = {
        'min': np.min(graph_sizes),
        'max': np.max(graph_sizes),
        'mean': np.mean(graph_sizes),
        'median': np.median(graph_sizes),
        'std': np.std(graph_sizes),
        'count': len(graph_sizes)
    }
    
    print("\nDataset Statistics:")
    print(f"Count: {stats['count']:,}")
    print(f"Min: {stats['min']:.6f}")
    print(f"Max: {stats['max']:.6f}") 
    print(f"Mean: {stats['mean']:.6f}")
    print(f"Median: {stats['median']:.6f}")
    print(f"Std Dev: {stats['std']:.6f}")
    
    return stats
def process_file(filepath):
    try:
        data = torch.load(filepath)
        y_values = data.y.cpu().numpy()
        return y_values
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def plot_y_distribution(data_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from multiprocessing import Pool, cpu_count
    
    pt_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.pt')
    ]
            
    num_processes = cpu_count()
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file, pt_files),
            total=len(pt_files),
            desc=f"Processing files using {num_processes} processes"
        ))
    
    all_y_values = []
    current_idx = 0
    indices = []
    
    for result in results:
        if result is not None:
            all_y_values.extend(result)
            indices.extend(range(current_idx, current_idx + len(result)))
            current_idx += len(result)
            
    all_y_values = np.array(all_y_values)
    print(all_y_values.shape)
    print(len(all_y_values))
    print(all_y_values.min())
    print(all_y_values.max())
    print(all_y_values.mean())
    print(all_y_values.std())
    # np.save('y_values_new.npy', all_y_values)
    print("Y values saved to 'y_values.npy'")
    
    # fig = plt.figure(figsize=(15, 15))
    # gs = plt.GridSpec(3, 2)
    
    # ax1 = fig.add_subplot(gs[0, :])
    # sns.histplot(data=all_y_values, kde=True, ax=ax1)
    # ax1.set_title('Distribution of Target Values (y)')
    # ax1.set_xlabel('Target Value')
    # ax1.set_ylabel('Count')
    
    # ax2 = fig.add_subplot(gs[1, 0])
    # sns.boxplot(data=all_y_values, ax=ax2)
    # ax2.set_title('Box Plot of Target Values (y)')
    # ax2.set_xlabel('Target Value')
    
    # ax3 = fig.add_subplot(gs[1, 1])
    # ax3.scatter(indices[:10000], all_y_values[:10000], alpha=0.5, s=1)
    # ax3.set_title('Scatter Plot of First 10000 Target Values')
    # ax3.set_xlabel('Sample Index')
    # ax3.set_ylabel('Target Value')
    
    # ax4 = fig.add_subplot(gs[2, :])
    # stats.probplot(all_y_values, dist="norm", plot=ax4)
    # ax4.set_title("Q-Q Plot of Target Values")
    
    # plt.tight_layout()
    # plt.savefig('y_distribution_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # print("\nPlot has been saved as 'y_distribution_analysis.png'")
    
    # plt.figure(figsize=(10, 6))
    # sns.violinplot(data=all_y_values)
    # plt.title('Violin Plot of Target Values (y)')
    # plt.savefig('y_violin_plot.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # print("Additional violin plot saved as 'y_violin_plot.png'")



if __name__ == "__main__":
    data_dir = "/scratch/gilbreth/mangla/bigdataset"
    calculate_statistics(data_dir)
    # plot_y_distribution(data_dir)