# AutoCombat with radiomics

## Requirements
Create cond environment
```bash
ENVNAME="ComScan"
conda create -n $ENVNAME python==3.7.7 -y
conda activate $ENVNAME
```
Then install the following packages:

### pyradiomics
_Install via pip:_
```bash
python -m pip install pyradiomics
```

### comscan
_Development installation:_
```bash
git clone https://github.com/Alxaline/ComScan.git
cd ComScan
pip install -e .
```
then correct deprecated warnings as descibed in logs

## Project Structure
### Python code
base_.py  
base_autocombat.py - harmonizing data using autocombat   
base_batch_input_images.py - extract features from images using pyradiomics  
base_fine_tune_csv.py - remove not needed entries from extracted features  
base_net.py - find best model for classification problem  
base_net_training.py - train and save model  
base_run_single.py - run trained model on custom input image  
loader.py - functions for loading specific file types  
utils.py

### Pyradiomics feature extraction config
params.yaml

### Resources
r_autoCombat.pck  
r_svm.pck  
r_xTransformed.pck  
r_y.pck  
radiomics_features.csv  
radiomics_features_selected.csv

## Usage
Simple use:
```
base_run_single.py <file_name.jpg>
```
As a result, the predicted and ground truth value (if present) will be displayed.
