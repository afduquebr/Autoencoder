#!/bin/bash

# Define variables
LHCO_REPO_URL="https://gitlab.cern.ch/idinu/clustering-lhco.git" # URL of the Git repository to clone
LHCO_DIR="../LHCO-Dataset"  # Directory where the repository will be cloned
TEXT_FILE="requirements.txt"  # The requirements text file to modify in the cloned repo

# Check if the directory already exists
if [ -d "$LHCO_DIR" ]; then
  echo "Directory $LHCO_DIR already exists. Skipping repository clone."
else
  # Clone the remote repository into the specified directory
  echo "Cloning repository from $LHCO_REPO_URL into $LHCO_DIR"
  git clone "$LHCO_REPO_URL" "$LHCO_DIR"
fi

# Change directory to the cloned repository
cd "$LHCO_DIR" || { echo "Failed to change directory to $LHCO_DIR"; exit 1; }

# Confirm the current directory
echo "Current directory: $(pwd)"

# Set up and configure the virtual environment and dependencies

# Check if the virtual environment already exists
if [ -d ".venv" ]; then
  echo "Virtual environment already exists. Activating..."
else
  # Create a Python virtual environment
  echo "Creating virtual environment..."
  python3.9 -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Update the `requirements.txt` file if it exists
if [ -f "$TEXT_FILE" ]; then
  echo "Modifying $TEXT_FILE..."
  
  # Replace the fifth line of the file with `h5py==3.10.0`
  sed -i '' '5s/.*/h5py==3.10.0/' "$TEXT_FILE"
  
  echo "$TEXT_FILE has been modified."
else
  echo "$TEXT_FILE not found in the cloned repository."
fi

# Install the required Python packages
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories if they don't exist
echo "Creating data directories..."
mkdir -p data_raw data  # Create data directories if they don't exist

# Execute the clustering scripts for various datasets
echo "Clustering RnD datasets..."
./LHCO.py cluster -D RnD ./data_raw/ --out-dir ./data/ --out-prefix RnD --njets 2
echo "Clustering Black Box 1 dataset..."
./LHCO.py cluster -K -D BBOX1 ./data_raw/ --out-dir ./data/ --out-prefix BBOX1 --njets 2
echo "Clustering Black Box 2 dataset..."
./LHCO.py cluster -D BBOX2 ./data_raw/ --out-dir ./data/ --out-prefix BBOX2 --njets 2

# Deactivate the virtual environment and return to the original directory
deactivate
cd - || { echo "Failed to return to the original directory"; exit 1; }

# Confirm we're back in the original directory
echo "Back in the original directory: $(pwd)"
