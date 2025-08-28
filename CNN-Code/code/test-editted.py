import os
import csv
import numpy as np
from scipy.stats import entropy
from src import *
import torch
import pandas as pd

# Define class labels and entropy threshold
CLASS_LABELS = ['CBT', 'CBTA', 'CBTP', 'CBT3', 'CBTPA', 'CBTP3']
ENTROPY_THRESHOLD = 0.2  # From paper

def save_tile_predictions(results, all_probs, output_file='tile_predictions.csv'):
    """Save tile predictions with WSI folder name, probabilities, entropy, and unclassified marking"""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Enhanced CSV header with WSI folder name
        writer.writerow([
            'wsi_folder', 'tile_file', 'predicted_class', 'entropy', 'is_uncertain',
            'prob_CBT', 'prob_CBTA', 'prob_CBTP',
            'prob_CBT3', 'prob_CBTPA', 'prob_CBTP3'
        ])
        
        # Process each tile result
        for res, probs in zip(results, all_probs):
            tile_path = res[2]
            # Extract WSI folder name (parent directory of tile)
            wsi_folder = os.path.basename(os.path.dirname(tile_path))
            # Get tile filename
            tile_file = os.path.basename(tile_path)
            
            # Convert to 1D probability array
            tile_probs = probs.squeeze()
            
            # Calculate entropy
            ent = entropy(tile_probs)
            
            # Get predicted class index
            pred_class_idx = res[0]
            
            # Determine if tile is uncertain
            is_uncertain = ent >= ENTROPY_THRESHOLD
            
            # Get class label - mark as "Unclassified" if uncertain
            if is_uncertain:
                class_label = "Unclassified"
            else:
                class_label = CLASS_LABELS[pred_class_idx]
            
            # Write to CSV
            writer.writerow([
                wsi_folder,        # WSI folder name (e.g., '15740-002')
                tile_file,         # Tile filename
                class_label,       # Either class name or "Unclassified"
                ent,
                int(is_uncertain),  # 1 if uncertain, 0 otherwise
                tile_probs[0], tile_probs[1], tile_probs[2],
                tile_probs[3], tile_probs[4], tile_probs[5]
            ])

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Initialize networks
pw_network = PatchWiseNetwork(args.channels)
iw_network = ImageWiseNetwork(args.channels)

# Handle testset path selection
if args.testset_path == '':
    import tkinter.filedialog as fdialog
    args.testset_path = fdialog.askopenfilename(
        initialdir=r"./dataset/test", 
        title="Choose your file", 
        filetypes=(("tiff files", "*.tif"), ("all files", "*.*"))
    )

# Verify checkpoint path exists
if not os.path.exists(args.checkpoints_path):
    raise FileNotFoundError(f"Checkpoint path {args.checkpoints_path} does not exist")

if args.network == '1':
    # Patch-wise model branch (unchanged)
    pw_checkpoint = os.path.join(args.checkpoints_path, 'weights_pw1.pth')
    if not os.path.exists(pw_checkpoint):
        raise FileNotFoundError(f"PatchWiseModel checkpoint {pw_checkpoint} does not exist")
    
    # Load checkpoint
    state_dict = torch.load(pw_checkpoint, map_location='cuda' if args.cuda else 'cpu')
    pw_network.load_state_dict(state_dict)
    print(f"Loaded PatchWiseModel weights from {pw_checkpoint}")
    
    pw_model = PatchWiseModel(args, pw_network)
    pw_model.test(args.testset_path)  # Original functionality
    
else:
    # Image-wise model branch with uncertainty marking and WSI folder
    # Load weights for image-wise model
    iw_checkpoint = os.path.join(args.checkpoints_path, 'weights_iw_h2gnew_trn1.pth')
    if not os.path.exists(iw_checkpoint):
        raise FileNotFoundError(f"ImageWiseModel checkpoint {iw_checkpoint} does not exist")
    
    state_dict = torch.load(iw_checkpoint, map_location='cuda' if args.cuda else 'cpu')
    iw_network.load_state_dict(state_dict)
    print(f"Loaded ImageWiseModel weights from {iw_checkpoint}")
    
    # Try to load patch-wise weights (optional)
    pw_checkpoint = os.path.join(args.checkpoints_path, 'weights_pw1.pth')
    if os.path.exists(pw_checkpoint):
        try:
            state_dict = torch.load(pw_checkpoint, map_location='cuda' if args.cuda else 'cpu')
            pw_network.load_state_dict(state_dict)
            print(f"Loaded PatchWiseModel weights from {pw_checkpoint}")
        except Exception as e:
            print(f"Warning: Failed to load PatchWiseModel weights: {str(e)}")
    else:
        print("Warning: PatchWiseModel weights not found, using random initialization")
    
    # Create model
    im_model = ImageWiseModel(args, iw_network, pw_network)
    
    # Run test with return_probs=True to get probabilities
    results, all_probs = im_model.test(
        args.testset_path, 
        ensemble=False,  # Disable ensemble for tile-level predictions
        verbose=False,
        return_probs=True  # Request probabilities
    )
    
    # Save predictions to CSV with WSI folder and uncertainty marking
    save_tile_predictions(results, all_probs)
    print(f"\n✅ Saved per-tile predictions to tile_predictions.csv")
    print(f"   - WSI folder name included for each tile")
    print(f"   - Tiles with entropy ≥ {ENTROPY_THRESHOLD:.2f} are marked as 'Unclassified'")