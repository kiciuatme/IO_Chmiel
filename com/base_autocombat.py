import os
import pandas as pd

from loader import read_file_list_from_csv
import utils
from utils import save_pickled

###############################################################
PROJECT_ROOT = utils.get_project_root(__file__)

COM_ROOT = PROJECT_ROOT + r"\com"
csv_res = COM_ROOT + r"\radiomics_features_selected.csv"
###############################################################

# load preselected data
rows = read_file_list_from_csv(csv_res)

# check correctness
print("Len in:", len(rows))
print(rows[0])

# convert to pd
features = pd.DataFrame(rows)

# ensure type
all_cols = features.select_dtypes(include=object).columns
except_one = all_cols[all_cols != 'Image']
features[except_one] = features[except_one].astype(float)
features[['MEL', 'NV']] = features[['MEL', 'NV']].astype(int)

# check correctness
# print('\n', features)

# Print the count
print('Original lengths:')
print('MEL: ', features[features['MEL'] == 1].shape[0])
print('NV:  ', features[features['NV'] == 1].shape[0])
print('Cropping. After it both will be the same.')

# equalize number
desired_rows = 1259
rows_to_delete_indices = features[features['NV'] == 1].sample(features.shape[0] - 2 * desired_rows).index
features = features.drop(rows_to_delete_indices)

assert features[features['MEL'] == 1].shape[0] == features[features['NV'] == 1].shape[0]


###############################################################
def test_elbow_visualizer():
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.cluster import KMeans

    # y = features['MEL'].values
    X = features.drop('MEL', axis=1).drop('Image', axis=1).drop('NV', axis=1).values

    model = KElbowVisualizer(KMeans(), k=(4, 12))
    model.fit(X)
    model.show()  # looks fine


# test_elbow_visualizer()
###############################################################
from ComScan import AutoCombat  # why it loads so long?

limit_input_to = 600
data = features.drop('MEL', axis=1)\
               .drop('Image', axis=1)\
               .drop('NV', axis=1)\
               .head(limit_input_to)  # uncomment this line to limit input data # todo: is this good?

ft = [key for key in data.keys() if key.startswith('original_glcm')]
sf = [key for key in data.keys() if key.startswith('original_firstorder')]
ccf = [key for key in data.keys() if key not in ft + sf]

print('Preparing AutoCombat')
ac = AutoCombat(
    features=ft,
    sites_features=sf,
    continuous_cluster_features=ccf,
    size_min=2,
    n_jobs=10
)

print('Processing AutoCombat.fit')
print(ac.fit(data))

print('Processing AutoCombat.transform')
x_trans = ac.transform(data)
# print(x_trans)
print(x_trans.shape)

###############################################################
# save results
file_path_ac = COM_ROOT + r'\r_autoCombat.pck'
file_path_xt = COM_ROOT + r'\r_xTransformed.pck'
file_path_y = COM_ROOT + r'\r_y.pck'

y = features['MEL'].head(limit_input_to).values
save_pickled(file_path_y, y)
save_pickled(file_path_xt, x_trans)
save_pickled(file_path_ac, ac)

print("Files saved to:", file_path_ac, file_path_xt, file_path_y, sep='\n')
