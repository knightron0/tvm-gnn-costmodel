import os
import torch
from tqdm import tqdm
import numpy as np

def calculate_statistics(data_dir):
    pt_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.pt')
    ]
    
    print(f"Found {len(pt_files)} .pt files")
    
    graph_sizes = []
    
    for filepath in tqdm(pt_files, desc="Processing files"):
        try:
            data = torch.load(filepath)
            graph_sizes.extend([data.x.shape[0]])
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue
    
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

def plot_y_distribution(data_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    pt_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.pt')
    ]
    
    all_y_values = []
    indices = []
    current_idx = 0
    
    for filepath in tqdm(pt_files, desc="Processing files for plotting"):
        try:
            data = torch.load(filepath)
            y_values = data.y.cpu().numpy()
            all_y_values.extend(y_values)
            indices.extend(range(current_idx, current_idx + len(y_values)))
            current_idx += len(y_values)
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue
            
    all_y_values = np.array(all_y_values)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(3, 2)
    
    # Histogram with KDE
    ax1 = fig.add_subplot(gs[0, :])
    sns.histplot(data=all_y_values, kde=True, ax=ax1)
    ax1.set_title('Distribution of Target Values (y)')
    ax1.set_xlabel('Target Value')
    ax1.set_ylabel('Count')
    
    # Box plot
    ax2 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=all_y_values, ax=ax2)
    ax2.set_title('Box Plot of Target Values (y)')
    ax2.set_xlabel('Target Value')
    
    # Scatter plot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(indices[:10000], all_y_values[:10000], alpha=0.5, s=1)
    ax3.set_title('Scatter Plot of First 10000 Target Values')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Target Value')
    
    # Q-Q plot
    ax4 = fig.add_subplot(gs[2, :])
    stats.probplot(all_y_values, dist="norm", plot=ax4)
    ax4.set_title("Q-Q Plot of Target Values")
    
    plt.tight_layout()
    plt.savefig('y_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPlot has been saved as 'y_distribution_analysis.png'")
    
    # Additional violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=all_y_values)
    plt.title('Violin Plot of Target Values (y)')
    plt.savefig('y_violin_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Additional violin plot saved as 'y_violin_plot.png'")



if __name__ == "__main__":
    data_dir = "/scratch/gilbreth/mangla/gnn_dataset/processed"
    calculate_statistics(data_dir)
