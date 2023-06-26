import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import confusion_matrix
from sklearn import metrics


def get_project_root(file_path):
    return os.path.abspath(os.path.join(os.path.dirname(file_path), '..'))


def save_pickled(file_path, val):
    with open(file_path, 'wb') as file:
        pickle.dump(val, file)


def load_pickled(file_path):
    # print(f'Reading from {file_path}')  # logging
    with open(file_path, 'rb') as file:
        x = pickle.load(file)

    return x


def pop_obsolete_entries(entry):
    # Get a list of keys to delete - we don't need diagnostics for ML task
    keys_to_delete = [key for key in entry.keys() if key.startswith('diagnostics')]
    for key in keys_to_delete:
        del entry[key]

    # similar values - wont be useful
    keys_to_delete = [key for key in entry.keys() if key.startswith('original_shape')]
    for key in keys_to_delete:
        del entry[key]

    return entry


def summary(y_got, y_pred):
    juxta, confusion_mx_sample, confusion_mx_percent, references_mx, labels = sumup(y_got, y_pred)

    print("confusion_mx_sample:")
    print(confusion_mx_sample)

    classes = np.unique(y_got)

    plot_confusion_matrix(classes, confusion_mx_sample)

    print("confusion_mx_percent:")
    print(confusion_mx_percent)

    ############################################################
    accuracy = metrics.accuracy_score(y_got, y_pred)
    precision = metrics.precision_score(y_got, y_pred)
    recall = metrics.recall_score(y_got, y_pred)
    f1 = metrics.f1_score(y_got, y_pred)

    # Print individual metric scores
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    # Generate a classification report
    report = metrics.classification_report(y_got, y_pred)
    print("Classification Report:\n", report)


def plot_confusion_matrix(classes, confusion_mx_sample):
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_mx_sample,
                annot=True,
                cmap='Blues',
                fmt='d',
                xticklabels=classes, yticklabels=classes)
    labels = ['NV', 'MEL']
    plt.gca().set_xticklabels(labels)
    plt.gca().set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def sumup(y_got, y_pred, references=None, labels=None):
    """copied from KS"""
    juxta_col_reference = "reference"
    juxta_col_cat = ["GoT", "predicted"]
    juxta_col_sparse = ["#GoT", "#predicted"]

    # convert y_got, y_pred to sparse
    if type(y_got[0]) == str:  # categorical
        dl = labels or list(set(y_got).union(list(y_pred)))
        y_got = np.asarray([dl.index(y_s) for y_s in y_got])
        y_pred = np.asarray([dl.index(y_s) for y_s in y_pred])
    elif np.asarray(y_got).shape.__len__() == 1:  # sparse
        dl = labels or list(set(y_got).union(list(y_pred)))
    elif np.asarray(y_got).shape.__len__() == 2:  # full
        dl = labels or np.arange(y_got[0].__len__())
        y_got = y_got.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
    references = references or ["?"] * y_got.__len__()

    juxta = pd.DataFrame(zip(
        references,
        [dl[catix] for catix in y_got],
        [dl[catix] for catix in y_pred],
        y_got,
        y_pred,
    ), columns=[juxta_col_reference, *juxta_col_cat, *juxta_col_sparse])

    confusion_mx_sample = confusion_matrix(y_got, y_pred, sample_weight=None).astype(int)

    # weights by y_got
    sample_weight = [(1 / (y_got == catix).sum()) for catix in range(dl.__len__())]
    sample_weight = np.asarray([sample_weight[catix] for catix in y_got]) * 100
    confusion_mx_percent = confusion_matrix(y_got, y_pred, sample_weight=sample_weight).astype(int)

    references_mx = [[juxta.loc[np.logical_and(
        juxta[juxta_col_sparse[0]] == catix_got,
        juxta[juxta_col_sparse[1]] == catix_pred
    ), juxta_col_reference].tolist()
                      for catix_pred in range(dl.__len__())] for catix_got in range(dl.__len__())]

    return juxta, confusion_mx_sample, confusion_mx_percent, references_mx, labels


def plot_parameter_choosing(accs, gammas):
    def plot_c_values(accs, gamma=''):
        c_values = [item[1] for item in accs if item[0] == gamma]
        accuracies = [item[2] for item in accs if item[0] == gamma]
        plt.plot(c_values, accuracies, marker='o')

    plot_c_values(accs, gamma='auto')
    plot_c_values(accs, gamma='scale')
    plt.xlabel('C values')
    plt.ylabel('Accuracy')
    plt.title('Accuracy scores for different gammas and C values')
    plt.xticks(rotation=45)
    plt.legend(gammas)
    plt.grid(True)
    plt.show()
