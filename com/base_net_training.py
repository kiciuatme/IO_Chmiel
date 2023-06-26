from sklearn.model_selection import train_test_split

from base_net import create_best_svm
import utils

PROJECT_ROOT = utils.get_project_root(__file__)
COM_ROOT = PROJECT_ROOT + r"\com"
file_path_x = COM_ROOT + r'\r_xTransformed.pck'
file_path_y = COM_ROOT + r'\r_y.pck'
file_path_svm = COM_ROOT + r'\r_svm.pck'

############################################################
# load data
X = utils.load_pickled(file_path_x)
y = utils.load_pickled(file_path_y)

############################################################
# prepare data split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=41)

############################################################
# define model
svm = create_best_svm(X_train, X_test, y_train, y_test)

############################################################
# train model
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

############################################################
# evaluate model
utils.summary(y_test, y_pred)

############################################################
# save model
utils.save_pickled(file_path_svm, svm)
