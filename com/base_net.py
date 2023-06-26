import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def create_best_svm(X_train, X_test, y_train, y_test):
    gamma, c, acc = find_best_c_value(X_train, y_train, X_test, y_test)
    print(f"['{gamma}'][{c:.2f}]: {acc}")
    svm = SVC(C=c, gamma=gamma, random_state=2)

    return svm


def find_best_c_value(X_train, y_train, X_test, y_test):
    accs = []
    gammas = ['auto', 'scale']
    for gamma in gammas:
        for c in np.logspace(-3, 2, num=50):
            c_ = c / 10
            model = SVC(C=c_, gamma=gamma, random_state=2)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # print(f"[{c_}]: {accuracy}")  # logging

            accs.append((gamma, c_, accuracy))

    best_gamma, best_c, best_accuracy = max(accs, key=lambda x: x[2])

    # Plot c values
    # plot_parameter_choosing(accs, gammas)

    return best_gamma, best_c, best_accuracy
