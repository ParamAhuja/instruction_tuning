import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_merged_heatmaps():
    """
    Generates a single merged heatmap image for each layer, containing
    subplots for Q, K, V, and O modules.
    """
 
    num_layers = 28
    modules = ['q', 'k', 'v', 'o']
    csv_directory = 'EigenVectors data\downloaded_csvs'
    output_directory = 'heatmaps_merged'


    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    # --- Main Loop ---
    for layer_idx in range(num_layers):
        # Create a figure and a set of subplots in a 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        fig.suptitle(f'Dominant Eigenvector Heatmaps for Layer {layer_idx}', fontsize=20)
        
        # Flatten the 2x2 array of axes for easy iteration
        axes = axes.flatten()

        all_modules_found = True
        for i, module_name in enumerate(modules):
            filename = os.path.join(csv_directory, f"layer_{layer_idx}_{module_name}.csv")
            
            try:
                df = pd.read_csv(filename)
                heatmap_data = df.pivot(index='component_idx', columns='eigenvector_idx', values='value').sort_index()

                # Plot the heatmap on the appropriate subplot axis
                sns.heatmap(heatmap_data, ax=axes[i], cmap='viridis', cbar_kws={'label': 'Value'})
                axes[i].set_title(f'Module: {module_name.upper()}', fontsize=16)
                axes[i].set_xlabel('Eigenvector Index')
                axes[i].set_ylabel('Component Index')

            except FileNotFoundError:
                print(f"  File not found, cannot generate subplot for {filename}")
                axes[i].set_title(f'Module: {module_name.upper()} - FILE NOT FOUND', fontsize=12)
                axes[i].axis('off') # Turn off the empty subplot
                all_modules_found = False
            except Exception as e:
                print(f" Error processing {filename}: {e}")
                axes[i].set_title(f'Module: {module_name.upper()} - ERROR', fontsize=12)
                axes[i].axis('off')
                all_modules_found = False

        if all_modules_found:
            # Save the complete figure for the layer
            output_filename = os.path.join(output_directory, f"layer_{layer_idx}_merged_heatmap.png")
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
            plt.savefig(output_filename)
            print(f" Successfully generated merged heatmap for Layer {layer_idx}")
        else:
            print(f" Skipped saving merged heatmap for Layer {layer_idx} due to missing/error files.")

        # Close the figure to free up memory
        plt.close(fig)

if __name__ == '__main__':
    create_merged_heatmaps()
    print("\nAll merged heatmaps have been processed.")