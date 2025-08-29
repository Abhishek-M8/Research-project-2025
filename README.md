Repository Structure:
CNN-Code/                # Folder containing the CNN model that was used in Hodis et al. (2022) with the edits in test.py and model.py. This was used to generate the CNN's CSV predictions
CSV-Predictions/         # Model prediction outputs per WSI in CSV format, before clustering
DB-Scan-Clusters/        # CSV Results and scatterplots of DBSCAN clustering
Final-Results/           # Final classification reports, confusion matrices, entropy data per WSI
__pycache__/             # Python cache files (can be ignored)

DBSCAN.py                # Script to perform DBSCAN clustering on tile-level predictions
FinalOpenslide.py        # Openslide-based tiling and preprocessing of WSIs
QuPath_filling_script.txt# QuPath batch script for nuclei detection + filling
requirements.txt         # Text file containing all libraries used
Section-wise-plots.py    # Generates section-level summary plots
Tile-wise-plots.py       # Generates tile-level prediction plots
accuracy_calulation.py   # Computes tile/slide-level accuracy metrics
tommy_library.py         # Utility functions for I/O, plotting, and analysis

Libraries/package installation: This project requires Python 3.8 and the following libraries
matplotlib==3.10.5
numpy==2.3.2
openslide_python==1.4.1
pandas==2.3.2
Pillow==11.3.0
scikit_learn==1.7.1
scipy==1.16.1
seaborn==0.13.2
skimage==0.0
torch==2.7.1
torchvision==0.22.1

Set up:
git clone https://github.com/Abhishek-M8/Research-project-2025.git
cd Research-project-2025
pip install -r requirements.txt

Workflow:
1) Slide tiling- Run FinalOpenslide.py with WSIs to convert them into tiles compatible with the CNN
2) Masking - Install QuPath and run the QuPath_filling_script.txt to fill the nuclei of the tiles
3) CNN predictions - Run test-editted.py with the tiles to get CSV predictions. 
   Make sure to use --checkpoints-path [Path-to-model-weights] --network 2 --testset-path [Path-to-tiles]
4) DBSCAN clustering - Run DBSCAN.py to segment the tumour slides (Needed for the section-wise predictions)
5) Model performance - Run Section-wise-plots.py for performance evaluation based on tumour sections
                       Run Tile-wise-plots.py for performance metrics on the model based on tile-wise predictions

