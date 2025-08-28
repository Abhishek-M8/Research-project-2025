import pandas as pd

# Load the CSV file
file_path = r'C:\Users\Abhishek\Documents\GitHub\Research-project-Final\raw_csv\EH-15740-039-tile-predictions.csv'
df = pd.read_csv(file_path)

# Define ground truth (WSI label)
ground_truth = 'CBTPA'

# Calculate accuracy: count tiles where predicted_class matches ground_truth and is_uncertain is 0
correct_tiles = len(df[(df['predicted_class'] == ground_truth) & (df['is_uncertain'] == 0)])
total_tiles = len(df)
accuracy = correct_tiles / total_tiles

# Print results
print(f"Total tiles: {total_tiles}")
print(f"Correct tiles (predicted as {ground_truth} and not uncertain): {correct_tiles}")
print(f"Unmasked Accuracy: {accuracy:.4f}")

# Optional: Save to file
with open('unmasked_accuracy_results.txt', 'w') as f:
    f.write(f"Total tiles: {total_tiles}\n")
    f.write(f"Correct tiles (predicted as {ground_truth} and not uncertain): {correct_tiles}\n")
    f.write(f"Unmasked Accuracy: {accuracy:.4f}\n")