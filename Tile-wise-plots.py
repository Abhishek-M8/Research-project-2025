import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tommy_library import get_delong_pvalue_zscore
import re
import os
from sklearn.preprocessing import label_binarize
# Extract prefix from unmasked filename
unmasked_filename = 'EH-15740-040_tile_predictions_with_clusters.csv'
prefix_match = re.match(r'^EH-(.*?)_tile_predictions_with_clusters\.csv$', unmasked_filename)
prefix = prefix_match.group(1) if prefix_match else 'EH-15740-040'

# Create output directory with prefix
output_dir = f'{prefix}_results'
os.makedirs(output_dir, exist_ok=True)

# Load data
df_unmasked = pd.read_csv(unmasked_filename)
df_masked = pd.read_csv(r'EH-15740-040-masked_tile_predictions_with_clusters.csv')

# Preprocess tile names
df_unmasked['base_tile'] = df_unmasked['tile_file'].str.replace('.png', '')
df_masked['base_tile'] = df_masked['tile_file'].str.replace('_filled.png', '')

# Merge dataframes
df_merged = pd.merge(
    df_unmasked, 
    df_masked, 
    on='base_tile', 
    suffixes=('_unmasked', '_masked')
)

# ====== OVERALL TUMOR PREDICTION ======
def calculate_overall_tumor_prediction(df, suffix):
    """Calculate overall tumor prediction and entropy from tile-level predictions"""
    tumor_classes = ['CBT', 'CBTA', 'CBTP', 'CBT3', 'CBTPA', 'CBTP3']
    prob_cols = [f'prob_{cls}_{suffix}' for cls in tumor_classes]
    avg_probs = df[prob_cols].mean()
    total = avg_probs.sum()
    norm_probs = avg_probs / total
    predicted_class = norm_probs.idxmax().replace(f'prob_', '').replace(f'_{suffix}', '')
    # Check if max probability is below a threshold to classify as Unclassified
    max_prob = norm_probs.max()
    if max_prob < 0.5:  # Adjustable threshold
        predicted_class = 'Unclassified'
    epsilon = 1e-10
    entropy_val = -np.sum(norm_probs * np.log(norm_probs + epsilon))
    return predicted_class, entropy_val, norm_probs

# Calculate overall predictions
unmasked_pred, unmasked_entropy, unmasked_probs = calculate_overall_tumor_prediction(df_merged, 'unmasked')
masked_pred, masked_entropy, masked_probs = calculate_overall_tumor_prediction(df_merged, 'masked')

# ====== OUTPUT OVERALL RESULTS ======
print("===== OVERALL TUMOR PREDICTION =====")
print(f"Unmasked Prediction: {unmasked_pred}")
print(f"Unmasked Entropy: {unmasked_entropy:.6f}")
print(f"Masked Prediction: {masked_pred}")
print(f"Masked Entropy: {masked_entropy:.6f}")

# Plot probability distributions
plt.figure(figsize=(12, 6))
classes = ['CBT', 'CBTA', 'CBTP', 'CBT3', 'CBTPA', 'CBTP3', 'Unclassified']
x = np.arange(len(classes))

plt.subplot(1, 2, 1)
plt.bar(x, pd.concat([unmasked_probs, pd.Series(0, index=['Unclassified'])])[:len(classes)], color='skyblue')
plt.xticks(x, classes, rotation=45)
plt.title('Unmasked Overall Probabilities')
plt.ylabel('Probability')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.bar(x, pd.concat([masked_probs, pd.Series(0, index=['Unclassified'])])[:len(classes)], color='salmon')
plt.xticks(x, classes, rotation=45)
plt.title('Masked Overall Probabilities')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{prefix}_overall_probabilities.png'))
plt.close()
print(f"\nSaved overall probabilities plot as '{output_dir}/{prefix}_overall_probabilities.png'")

# ====== TILE-LEVEL ANALYSIS ======
# Use all tiles without filtering out Unclassified
valid_tiles = df_merged

# Calculate accuracy
accuracy = (valid_tiles['predicted_class_masked'] == valid_tiles['predicted_class_unmasked']).mean()
print(f"\nTile-level Accuracy: {accuracy:.4f}")

# Confusion matrix
conf_matrix = pd.crosstab(
    valid_tiles['predicted_class_unmasked'], 
    valid_tiles['predicted_class_masked'], 
    rownames=['Unmasked Tile Predictions'],
    colnames=['Masked Tile predictions']
).reindex(index=classes, columns=classes, fill_value=0)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: Masked vs Unmasked Predictions')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{prefix}_confusion_matrix.png'))
plt.close()
print(f"\nSaved confusion matrix as '{output_dir}/{prefix}_confusion_matrix.png'")


# ----- Class Shift Matrix -----
class_shift = pd.crosstab(
    df_merged['predicted_class_unmasked'],   # y-axis: ground truth
    df_merged['predicted_class_masked'],     # x-axis: predicted
).reindex(index=classes, columns=classes, fill_value=0)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(class_shift, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Class Shift: Masked vs Unmasked Predictions (in %)')
plt.ylabel('Unmasked tiles')
plt.xlabel('Masked tiles')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{prefix}_class_shift.png'))
plt.close()
print(f"Saved class shift plot as '{output_dir}/{prefix}_class_shift.png'")



# Classification report
print("\nUnique classes in valid_tiles:")
print("Unmasked (true):", valid_tiles['predicted_class_unmasked'].unique())
print("Masked (predicted):", valid_tiles['predicted_class_masked'].unique())
print("\nClassification Report:")
class_report = classification_report(
    valid_tiles['predicted_class_unmasked'],
    valid_tiles['predicted_class_masked'],
    labels=classes,
    target_names=classes,
    output_dict=False,
    zero_division=np.nan
)
print(class_report)

# Log classes with issues (optional diagnostic)
unique_unmasked = valid_tiles['predicted_class_unmasked'].unique()
unique_masked = valid_tiles['predicted_class_masked'].unique()
missing_unmasked = [c for c in classes if c not in unique_unmasked]
missing_masked = [c for c in classes if c not in unique_masked]
if missing_unmasked or missing_masked:
    print("\nWarning: Some classes are missing in the analysis:")
    if missing_unmasked:
        print(f"Missing in unmasked (true) labels: {missing_unmasked}")
    if missing_masked:
        print(f"Missing in masked (predicted) labels: {missing_masked}")

# Save classification report to file
with open(os.path.join(output_dir, f'{prefix}_classification_report.txt'), 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)
print(f"Saved classification report as '{output_dir}/{prefix}_classification_report.txt'")

results = []
for class_name in classes:
    y_true = (df_merged['predicted_class_unmasked'] == class_name).astype(int)
    n_positives = y_true.sum()
    
    if n_positives == 0:
        print(f"\nSkipping {class_name}: No positive examples")
        continue
    if len(np.unique(y_true)) == 1:
        print(f"\nSkipping {class_name}: Only one class present")
        continue

    # --- Masked AUC ---
    prob_col_masked = f'prob_{class_name}_masked' if class_name != 'Unclassified' else None
    y_score_masked = df_merged[prob_col_masked] if prob_col_masked else np.zeros(len(df_merged))
    try:
        auc_masked = roc_auc_score(y_true, y_score_masked) if class_name != 'Unclassified' else np.nan
    except ValueError:
        auc_masked = np.nan

    # --- Unmasked AUC ---
    prob_col_unmasked = f'prob_{class_name}_unmasked' if class_name != 'Unclassified' else None
    y_score_unmasked = df_merged[prob_col_unmasked] if prob_col_unmasked else np.zeros(len(df_merged))
    try:
        auc_unmasked = roc_auc_score(y_true, y_score_unmasked) if class_name != 'Unclassified' else np.nan
    except ValueError:
        auc_unmasked = np.nan

    # --- Δ AUC ---
    delta_auc = auc_masked - auc_unmasked if (not np.isnan(auc_masked) and not np.isnan(auc_unmasked)) else np.nan

    results.append({
        'class': class_name,
        'n_positives': n_positives,
        'auc_unmasked': auc_unmasked,
        'auc_masked': auc_masked,
        'delta_auc': delta_auc
    })

results_df = pd.DataFrame(results)
print("\n===== CLASS PERFORMANCE METRICS =====")
print(results_df.to_string(index=False))

# ====== SUMMARY STATS ======
uncertainty_rate = (df_merged['predicted_class_masked'] == 'Unclassified').mean()

# Add entropy + overall stats as an extra "Overall" row
overall_row = {
    'class': 'Overall',
    'n_positives': len(df_merged),
    'auc_masked': results_df['auc_masked'].mean(skipna=True),
    'delta_auc': results_df['delta_auc'].mean(skipna=True),
}
summary_df = pd.DataFrame([overall_row])

# Merge into one final table
final_df = pd.concat([results_df, summary_df], ignore_index=True)

# Add other columns
final_df['entropy_unmasked'] = unmasked_entropy
final_df['entropy_masked'] = masked_entropy
final_df['tile_accuracy'] = accuracy
final_df['uncertainty_rate'] = uncertainty_rate

print("\n===== FINAL SUMMARY TABLE =====")
print(final_df.to_string(index=False))

# Save to CSV
final_df.to_csv(os.path.join(output_dir, f'{prefix}_summary_table.csv'), index=False)
print(f"\nSaved final summary table as '{output_dir}/{prefix}_summary_table.csv'")
