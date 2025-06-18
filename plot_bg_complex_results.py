import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_background_complexity_results(csv_file='results_bg_complex.csv'):
    """
    Plot background complexity results from CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing results
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Extract data
    models = df['Model'].tolist()
    
    # Extract train and validation losses for each level
    train_losses = {
        'L0': df['Train L0'].tolist(),
        'L1': df['Train L1'].tolist(), 
        'L2': df['Train L2'].tolist(),
        'L3': df['Train L3'].tolist()
    }
    
    val_losses = {
        'L0': df['Val L0'].tolist(),
        'L1': df['Val L1'].tolist(),
        'L2': df['Val L2'].tolist(), 
        'L3': df['Val L3'].tolist()
    }
    
    # Prepare data for plotting
    levels = ['L0', 'L1', 'L2', 'L3']
    train_data = []
    val_data = []
    
    for model in models:
        model_idx = models.index(model)
        train_model_losses = [train_losses[level][model_idx] for level in levels]
        val_model_losses = [val_losses[level][model_idx] for level in levels]
        train_data.append(train_model_losses)
        val_data.append(val_model_losses)
    
    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Create separate plots for train and validation loss
    
    # Plot 1: Average Train Loss
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    for i, model in enumerate(models):
        if not all(np.isnan(train_data[i])):
            ax1.plot(range(len(levels)), train_data[i], 
                    marker='o', label=model.upper(), color=colors[i], 
                    linewidth=2, markersize=6)
    
    ax1.set_xlabel('Level', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Average Train Loss by Background Complexity Level', fontsize=14)
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels([f'{i}' for i in range(len(levels))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    train_output_file = 'train_loss_bg_complexity.png'
    plt.savefig(train_output_file, dpi=300, bbox_inches='tight')
    print(f"Train loss plot saved as: {train_output_file}")
    plt.show()
    
    # Plot 2: Average Val Loss
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    for i, model in enumerate(models):
        if not all(np.isnan(val_data[i])):
            ax2.plot(range(len(levels)), val_data[i], 
                    marker='o', label=model.upper(), color=colors[i], 
                    linewidth=2, markersize=6)
    
    ax2.set_xlabel('Level', fontsize=12)
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_title('MSE vs. Background Complexity Level', fontsize=14)
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels([f'{i}' for i in range(len(levels))])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    val_output_file = 'val_loss_bg_complexity.png'
    plt.savefig(val_output_file, dpi=300, bbox_inches='tight')
    print(f"Val loss plot saved as: {val_output_file}")
    plt.show()
    
    # Also create the combined plot for comparison
    fig_combined, (ax1_combined, ax2_combined) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Combined Plot 1: Average Train Loss
    for i, model in enumerate(models):
        if not all(np.isnan(train_data[i])):
            ax1_combined.plot(range(len(levels)), train_data[i], 
                    marker='o', label=model.upper(), color=colors[i], 
                    linewidth=2, markersize=6)
    
    ax1_combined.set_xlabel('Level', fontsize=12)
    ax1_combined.set_ylabel('Train Loss', fontsize=12)
    ax1_combined.set_title('Average Train Loss by Background Complexity Level', fontsize=14)
    ax1_combined.set_xticks(range(len(levels)))
    ax1_combined.set_xticklabels([f'{i}' for i in range(len(levels))])
    ax1_combined.legend()
    ax1_combined.grid(True, alpha=0.3)
    
    # Combined Plot 2: Average Val Loss
    for i, model in enumerate(models):
        if not all(np.isnan(val_data[i])):
            ax2_combined.plot(range(len(levels)), val_data[i], 
                    marker='o', label=model.upper(), color=colors[i], 
                    linewidth=2, markersize=6)
    
    ax2_combined.set_xlabel('Level', fontsize=12)
    ax2_combined.set_ylabel('Val Loss', fontsize=12)
    ax2_combined.set_title('Average Val Loss by Background Complexity Level', fontsize=14)
    ax2_combined.set_xticks(range(len(levels)))
    ax2_combined.set_xticklabels([f'{i}' for i in range(len(levels))])
    ax2_combined.legend()
    ax2_combined.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_output_file = 'combined_bg_complexity_results.png'
    plt.savefig(combined_output_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved as: {combined_output_file}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Print the data in a nice table format
    print(f"{'Model':<12} | {'Train L0':<10} | {'Train L1':<10} | {'Train L2':<10} | {'Train L3':<10} | {'Val L0':<10} | {'Val L1':<10} | {'Val L2':<10} | {'Val L3':<10}")
    print("-" * 120)
    
    for i, model in enumerate(models):
        train_str = " | ".join([f"{x:.4f}" if not np.isnan(x) else "  N/A  " for x in train_data[i]])
        val_str = " | ".join([f"{x:.4f}" if not np.isnan(x) else "  N/A  " for x in val_data[i]])
        print(f"{model.upper():<12} | {train_str} | {val_str}")
    
    # Calculate and print averages across all models for each level
    print("\n" + "="*80)
    print("LEVEL AVERAGES (across all models)")
    print("="*80)
    
    for j, level in enumerate(levels):
        train_avg = np.nanmean([train_data[i][j] for i in range(len(models))])
        val_avg = np.nanmean([val_data[i][j] for i in range(len(models))])
        print(f"{level}: Train Avg = {train_avg:.4f}, Val Avg = {val_avg:.4f}")

if __name__ == "__main__":
    # You can specify a different CSV file path here if needed
    csv_file = 'results_line_bg_complex.csv'
    plot_background_complexity_results(csv_file) 