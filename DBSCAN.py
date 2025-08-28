import os
import re
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pathlib import Path

# Load your prediction CSV
prediction_csv_path = r'EH-15740-040-masked_tile_predictions.csv'  # Update with your actual path
df = pd.read_csv(prediction_csv_path)

# Extract base name for output file
base_name = os.path.splitext(os.path.basename(prediction_csv_path))[0]
output_folder = "cluster_results"
output_filename = f"{base_name}_with_clusters.csv"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, output_filename)

# Function to extract coordinates from filename
def extract_coords(filename):
    # Handle both regular and masked tile names
    patterns = [
        r'tile_(\d+)_(\d+)_filled\.png',  # Pattern for masked files
        r'tile_(\d+)_(\d+)\.png'          # Pattern for unmasked files
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            return x, y
    return None, None

# Extract coordinates from tile_file column
coords_list = []
valid_indices = []  # To keep track of rows with valid coordinates

for idx, row in df.iterrows():
    x, y = extract_coords(row['tile_file'])
    if x is not None and y is not None:
        coords_list.append([x, y])
        valid_indices.append(idx)

# Convert to numpy array
coords = np.array(coords_list)

# Run DBSCAN clustering
db = DBSCAN(eps=2048, min_samples=4).fit(coords)
labels = db.labels_

# Add cluster labels to the DataFrame
# First, create a new column with NaN values
df['cluster'] = np.nan

# Then, assign cluster labels only to rows with valid coordinates
for i, idx in enumerate(valid_indices):
    df.at[idx, 'cluster'] = labels[i]

# Convert cluster column to integer (NaN values will become a special value, or keep as is)
df['cluster'] = df['cluster'].astype('Int64')  # Uses pandas' nullable integer type

# Count clusters
unique, counts = np.unique(labels, return_counts=True)
print("Cluster distribution:")
print(dict(zip(unique, counts)))

# Optional: visualize the clusters
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=30)
plt.gca().invert_yaxis()
plt.title("Tumor tile clusters from DBSCAN")
plt.colorbar(label='Cluster ID')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Save visualization to the output folder
visualization_path = os.path.join(output_folder, f"{base_name}_clusters_visualization.png")
plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
plt.show()

# Save the updated DataFrame with cluster information
df.to_csv(output_path, index=False)
print(f"Updated predictions saved to {output_path}")
print(f"Visualization saved to {visualization_path}")

# Additional analysis: Show how many tiles per cluster
cluster_counts = df['cluster'].value_counts().sort_index()
print("\nTiles per cluster:")
print(cluster_counts)

# Create a summary report
summary_path = os.path.join(output_folder, f"{base_name}_clustering_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"DBSCAN Clustering Results for {base_name}\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total tiles processed: {len(df)}\n")
    f.write(f"Tiles with valid coordinates: {len(valid_indices)}\n")
    f.write(f"Number of clusters identified: {len(np.unique(labels[labels >= 0]))}\n")
    f.write(f"Noise points (cluster = -1): {np.sum(labels == -1)}\n\n")
    f.write("Cluster sizes:\n")
    for cluster_id, count in dict(zip(unique, counts)).items():
        f.write(f"  Cluster {cluster_id}: {count} tiles\n")

print(f"Summary report saved to {summary_path}")