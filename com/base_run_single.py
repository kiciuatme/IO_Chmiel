import os.path
import sys
import pandas as pd
import csv

from radiomics import featureextractor

import utils
from loader import load_image, load_dummy_mask

PROJECT_ROOT = utils.get_project_root(__file__)
COM_ROOT = PROJECT_ROOT + r"\com"
PARAMS_FILE = COM_ROOT + r'\params.yaml'
BASE_DIR = PROJECT_ROOT + r"\ISIC2018_Task3_Training_Input"
inputCSV = PROJECT_ROOT + r"\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"
file_path_svm = COM_ROOT + r'\r_svm.pck'

###############################################################
# process input
if len(sys.argv) == 1:
    print("Usage: <script_name> <image_file_name>")
    exit(-1)

file_name = sys.argv[1]
file_path_img = os.path.join(BASE_DIR, file_name)

# print(file_name)
# print(file_path_img)
###############################################################
# load data
image3d = load_image(file_path_img)

mask_name = PROJECT_ROOT + r"\mask-1.png"
mask3d = load_dummy_mask(mask_name)

params = COM_ROOT + "\params.yaml"

extractor = featureextractor.RadiomicsFeatureExtractor(params, verbose=True)

result = extractor.execute(image3d, mask3d)
# print(result)

result = utils.pop_obsolete_entries(result)
# print(result)

# convert to pd
features = pd.DataFrame([result]).astype(float)
# print(features)

###############################################################
file_path_ac = COM_ROOT + r'\r_autoCombat.pck'
ac = utils.load_pickled(file_path_ac)

x = ac.transform(features)
###############################################################
# load svm
svm = utils.load_pickled(file_path_svm)
y = svm.predict(x)[0]

###############################################################
with open(inputCSV, 'r') as inFile:
    cr = csv.DictReader(inFile, lineterminator='\n')
    gt = [row for row in cr if file_name.startswith(row['image'])]
    if len(gt) == 0:
        print("Image not found in ground truth! GT value below will be made up.")
        gt = [{'MEL': '1.0'}]

    gt = gt[0]

y_true = int(float(gt['MEL']) == 1.)
labels = ['NV', 'MEL']
print(f'\n"{file_name} -> {labels[y]}({y}) | GT: {labels[y_true]}({y_true})')

if y == y_true:
    print('Successful prediction')
