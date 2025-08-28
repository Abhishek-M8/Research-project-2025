# aggregate.py (full refined version with prefix and other changes)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
import re

# ========== CONFIGURABLE PARAMETERS ==========
TILE_ENTROPY_THRESHOLD = 1   # Entropy threshold for tile-level uncertainty
SECTION_ENTROPY_THRESHOLD = 1  # Entropy threshold for section-level uncertainty
# =============================================

# Extract prefix from masked filename
masked_filename = r'EH-15740-040-masked_tile_predictions_with_clusters.csv'
prefix_match = re.match(r'^(.*?)-predictions-masked_with_clusters\.csv$', masked_filename)
prefix = prefix_match.group(1) if prefix_match else 'EH-15740-040-Section-Results'

# Create output directory with prefix
output_dir = f'{prefix}_analysis_results'
os.makedirs(output_dir, exist_ok=True)

# DEBUGGING: Load and verify columns
try:
    tumor_df = pd.read_csv(masked_filename)
    print("File loaded successfully. Columns:")
    print(tumor_df.columns.tolist())
    
    # Use 'cluster' column as the group identifier
    if 'cluster' not in tumor_df.columns:
        print("Error: No 'cluster' column found. Available columns:")
        print(tumor_df.columns.tolist())
        exit()
    
    # Rename to standardized 'group' for consistency in code
    tumor_df = tumor_df.rename(columns={'cluster': 'group'})
    
except Exception as e:
    print(f"File loading error: {e}")
    exit()

# Define tumor classes with correct labels
tumor_classes = ['CBTA', 'CBT3', 'CBTPA', 'CBT', 'CBTP', 'CBTP3']
prob_cols = ['prob_CBTA', 'prob_CBT3', 'prob_CBTPA', 'prob_CBT', 'prob_CBTP', 'prob_CBTP3']

# Filter tumor tiles and apply entropy threshold
tumor_df = tumor_df[
    tumor_df['predicted_class'].isin(tumor_classes) & 
    (tumor_df['entropy'] <= TILE_ENTROPY_THRESHOLD)  # Use configurable threshold
].copy()

if tumor_df.empty:
    print("No certain tumor tiles found")
    exit()

# Verify group column exists
if 'group' not in tumor_df.columns:
    print("'group' column still missing after rename")
    print("Current columns:", tumor_df.columns.tolist())
    exit()

print("\nGroups detected:", tumor_df['group'].unique())

# 1. AGGREGATE PROBABILITIES TO SECTION LEVEL
section_probs = tumor_df.groupby('group', as_index=False)[prob_cols].sum()

# Normalize to probability distributions
row_sums = section_probs[prob_cols].sum(axis=1)
section_probs[prob_cols] = section_probs[prob_cols].div(row_sums, axis=0)

# 2. CALCULATE SECTION-LEVEL ENTROPY
section_entropy = []
for _, row in section_probs.iterrows():
    probs = row[prob_cols].values.astype(float)
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
    ent = entropy(probs, base=2)
    section_entropy.append(ent)
section_probs['entropy'] = section_entropy

# 3. ASSIGN SECTION CLASSIFICATIONS WITH CONFIGURABLE THRESHOLD
section_class = []
for _, row in section_probs.iterrows():
    if row['entropy'] <= SECTION_ENTROPY_THRESHOLD:
        max_class = row[prob_cols].astype(float).idxmax()
        section_class.append(max_class.replace('prob_', ''))
    else:
        section_class.append('Uncertain')
section_probs['predicted_class'] = section_class

# 4. CREATE RESULTS DATAFRAME
section_results = section_probs.copy()
print("\nSection-Level Predictions:")
print(section_results[['group', 'entropy', 'predicted_class'] + prob_cols])

# Save results to CSV
section_results.to_csv(os.path.join(output_dir, f'{prefix}_section_predictions.csv'), index=False)
print(f"\nSection predictions saved to {output_dir}/{prefix}_section_predictions.csv")

# 5. VISUALIZE SECTION PROBABILITIES
plt.figure(figsize=(14, 8))
x = np.arange(len(section_results))
width = 0.12  # Adjusted for 6 classes

for i, cls in enumerate(prob_cols):
    plt.bar(x + i*width, section_results[cls], width, label=cls.replace('prob_', ''))

plt.axhline(y=SECTION_ENTROPY_THRESHOLD, color='r', linestyle='--', 
            label=f'Uncertainty Threshold ({SECTION_ENTROPY_THRESHOLD})')
plt.xlabel('Tumor Section Group')
plt.ylabel('Probability')
plt.title('Section-Level Class Probabilities')
plt.xticks(x + width*2.5, section_results['group'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{prefix}_section_probabilities.png'), dpi=300)
plt.close()

# NEW: ENTROPY DISTRIBUTION ANALYSIS
plt.figure(figsize=(14, 6))

# Tile-level entropy distribution
plt.subplot(1, 2, 1)
plt.hist(tumor_df['entropy'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(TILE_ENTROPY_THRESHOLD, color='r', linestyle='--', 
            label=f'Threshold ({TILE_ENTROPY_THRESHOLD})')
plt.title('Tile-Level Entropy Distribution')
plt.xlabel('Entropy')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)

# Section-level entropy distribution
plt.subplot(1, 2, 2)
plt.hist(section_results['entropy'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.axvline(SECTION_ENTROPY_THRESHOLD, color='r', linestyle='--', 
            label=f'Threshold ({SECTION_ENTROPY_THRESHOLD})')
plt.title('Section-Level Entropy Distribution')
plt.xlabel('Entropy')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{prefix}_entropy_distributions.png'), dpi=300)
plt.close()

# NEW: UNCERTAINTY ANALYSIS
tile_uncertain_rate = (tumor_df['entropy'] > TILE_ENTROPY_THRESHOLD).mean()
section_uncertain_rate = (section_results['entropy'] > SECTION_ENTROPY_THRESHOLD).mean()

print("\nUncertainty Analysis:")
print(f"Tile-Level Uncertainty Rate: {tile_uncertain_rate:.2%} (>{TILE_ENTROPY_THRESHOLD})")
print(f"Section-Level Uncertainty Rate: {section_uncertain_rate:.2%} (>{SECTION_ENTROPY_THRESHOLD})")

# Save uncertainty analysis to file
with open(os.path.join(output_dir, f'{prefix}_uncertainty_analysis.txt'), 'w') as f:
    f.write("Uncertainty Analysis:\n")
    f.write(f"Tile-Level Uncertainty Rate: {tile_uncertain_rate:.2%} (>{TILE_ENTROPY_THRESHOLD})\n")
    f.write(f"Section-Level Uncertainty Rate: {section_uncertain_rate:.2%} (>{SECTION_ENTROPY_THRESHOLD})\n")

# NEW: THRESHOLD SENSITIVITY ANALYSIS
thresholds = np.linspace(0, 2, 21)  # 0.0 to 2.0 in 0.1 steps
uncertain_rates = []

for thresh in thresholds:
    # Tile-level uncertainty
    tile_uncertain = (tumor_df['entropy'] > thresh).mean()
    
    # Section-level uncertainty
    section_uncertain = (section_results['entropy'] > thresh).mean()
    
    uncertain_rates.append({
        'threshold': thresh,
        'tile_uncertain': tile_uncertain,
        'section_uncertain': section_uncertain
    })

uncertain_df = pd.DataFrame(uncertain_rates)
uncertain_df.to_csv(os.path.join(output_dir, f'{prefix}_threshold_sensitivity.csv'), index=False)

plt.figure(figsize=(10, 6))
plt.plot(uncertain_df['threshold'], uncertain_df['tile_uncertain'], 
         'b-o', label='Tile-Level', linewidth=2, markersize=6)
plt.plot(uncertain_df['threshold'], uncertain_df['section_uncertain'], 
         'g-s', label='Section-Level', linewidth=2, markersize=6)
plt.axvline(TILE_ENTROPY_THRESHOLD, color='r', linestyle='--', 
            label=f'Current Threshold ({TILE_ENTROPY_THRESHOLD})')
plt.title('Uncertainty Rate vs. Entropy Threshold')
plt.xlabel('Entropy Threshold')
plt.ylabel('Uncertainty Rate')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{prefix}_threshold_sensitivity.png'), dpi=300)
plt.close()

# ===========================
# 6. COMPARE MASKED AND UNMASKED SECTION ENTROPY
# ===========================

try:
    # Load unmasked predictions
    unmasked_filename = r'EH-15740-040_tile_predictions_with_clusters.csv'
    df_unmasked = pd.read_csv(unmasked_filename)
    
    # Create base_tile identifiers
    tumor_df['base_tile'] = tumor_df['tile_file'].str.replace('_filled\.png$', '', regex=True)
    df_unmasked['base_tile'] = df_unmasked['tile_file'].str.replace('\.png$', '', regex=True)
    
    # Merge masked tiles with unmasked predictions on base_tile
    merged = pd.merge(
        tumor_df[['base_tile', 'group']], 
        df_unmasked[['base_tile'] + prob_cols],
        on='base_tile',
        how='inner'
    )
    
    if merged.empty:
        print("Warning: No matching tiles found for unmasked comparison")
        unmasked_section_probs = pd.DataFrame()
    else:
        # Aggregate unmasked probabilities per section
        unmasked_section_probs = merged.groupby('group', as_index=False)[prob_cols].sum()
        # Normalize
        row_sums = unmasked_section_probs[prob_cols].sum(axis=1)
        unmasked_section_probs[prob_cols] = unmasked_section_probs[prob_cols].div(row_sums, axis=0)
        
        # Calculate section-level entropy for unmasked
        unmasked_section_probs['entropy'] = unmasked_section_probs[prob_cols].apply(
            lambda row: entropy(np.clip(row.values.astype(float), 1e-10, 1.0), base=2),
            axis=1
        )
        
        # Assign section-level predicted class
        unmasked_section_probs['predicted_class'] = unmasked_section_probs[prob_cols].idxmax(axis=1).str.replace('prob_', '')
        
except Exception as e:
    print(f"Error during unmasked aggregation: {str(e)}")
    unmasked_section_probs = pd.DataFrame()

# ===========================
# Merge masked and unmasked entropies for comparison
# ===========================
if not unmasked_section_probs.empty:
    entropy_comparison = pd.merge(
        section_results[['group', 'entropy']].rename(columns={'entropy':'masked_entropy'}),
        unmasked_section_probs[['group', 'entropy']].rename(columns={'entropy':'unmasked_entropy'}),
        on='group',
        how='outer'
    )
    
    # Calculate entropy change
    entropy_comparison['delta_entropy'] = entropy_comparison['masked_entropy'] - entropy_comparison['unmasked_entropy']
    
    # Save table
    entropy_comparison.to_csv(os.path.join(output_dir, f'{prefix}_section_entropy_comparison.csv'), index=False)
    
    # Visualize masked vs unmasked entropy
    plt.figure(figsize=(12,6))
    x = np.arange(len(entropy_comparison))
    width = 0.35
    plt.bar(x - width/2, entropy_comparison['masked_entropy'], width, label='Masked', color='skyblue')
    plt.bar(x + width/2, entropy_comparison['unmasked_entropy'], width, label='Unmasked', color='lightgreen')
    plt.xlabel('Tumor Section Group')
    plt.ylabel('Entropy')
    plt.title('Section-Level Entropy: Masked vs Unmasked')
    plt.xticks(x, entropy_comparison['group'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_section_entropy_masked_vs_unmasked.png'), dpi=300)
    plt.close()
    
    print("\nSection-level entropy comparison saved and visualized.")
else:
    print("Skipping entropy comparison: no unmasked section data available.")


print(f"\nAll analysis results have been saved to the '{output_dir}' directory.")