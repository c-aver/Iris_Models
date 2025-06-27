import itertools
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from kNN import KNNClassifier

# requested k and p values for testing
ks: list[int] = [1, 3, 5, 7, 9]
ps: list[int | Literal['∞']] = [1, 2, '∞']


def error_rate(classifier: KNNClassifier, X, y):
    errors = sum(1 for point, label in zip(X, y) if (classifier.predict(point) != label))
    return errors / len(X)


# run one experiment with filtered data
def experiment(X, y):


    print(f"Starting experiment with classes: {set(y)}")
    # store results for each (k, p) pair
    results: dict[tuple[int, int | Literal['∞']], list[float]] = {}
    for k, p in itertools.product(ks, ps):
        # 0: train error, 1: test error - before condensing
        # 2: train error, 3: test error - after condensing
        results[(k, p)] = [0, 0, 0, 0]

    # repeat 100 times
    for i in range(100):
        # split data with different random state each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)

        # for each (k, p) pair
        for k, p in itertools.product(ks, ps):
            # create a classifier with those parameters
            classifier = KNNClassifier(k, p)
            # fit it to the data
            classifier.fit(X_train, y_train)
            # calculate error rates on train and test and add to results
            results[(k, p)][0] += error_rate(classifier, X_train, y_train)
            results[(k, p)][1] += error_rate(classifier, X_test, y_test)
            # condense sets
            classifier.condense()
            # again calculate error rates on train and test and add to results
            results[(k, p)][2] += error_rate(classifier, X_train, y_train)
            results[(k, p)][3] += error_rate(classifier, X_test, y_test)

    # divide all results by 100 to average over repeated tests
    for k, p in itertools.product(ks, ps):
        results[(k, p)][0] /= 100
        results[(k, p)][1] /= 100
        results[(k, p)][2] /= 100
        results[(k, p)][3] /= 100
        # optional prints
        # print(f"For k = {k}, p = {p}:")
        # print("Before condensing:")
        # print(f"\tTrain error rate = {results[(k, p)][0]:.4f}")
        # print(f"\tTest error rate = {results[(k, p)][1]:.4f}")
        # print(f"\tDifference is {results[(k, p)][1] - results[(k, p)][0]:.4f}")
        # print("After condensing:")
        # print(f"\tTrain error rate = {results[(k, p)][2]:.4f}")
        # print(f"\tTest error rate = {results[(k, p)][3]:.4f}")
        # print(f"\tDifference is {results[(k, p)][3] - results[(k, p)][2]:.4f}")

    # prepare data for plotting
    _x = ks
    _y = [3 if p == '∞' else p for p in ps]
    _xx, _yy = np.meshgrid(_x, _y)
    x_coords, y_coords = _xx.ravel(), _yy.ravel()
    z_coords = [[], [], [], []]
    for k, p in itertools.product(ks, ps):
        for i in range(4):
            z_coords[i].append(results[(k, p)][i])

    # create plots
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
    axes = [ax for line in axes for ax in line]
    fig.suptitle(f'Results for labels {set(y)}', fontsize=16)

    axes[0].set_title('Train error rate', fontsize=10)
    axes[1].set_title('Test error rate', fontsize=10)
    axes[2].set_title('Train error rate (condensed)', fontsize=10)
    axes[3].set_title('Test error rate (condensed)', fontsize=10)

    for i in range(4):
        axes[i].bar3d(x_coords, y_coords, np.zeros_like(z_coords[i]), 0.5, 0.5, z_coords[i], shade=True)
        ax = axes[i]
        ax.set_xlabel('k')
        ax.set_xticks(_x)
        ax.set_ylabel('p')
        ax.set_yticks(_y)
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels[2] = '∞'
        ax.set_yticklabels(labels)
        ax.set_zlabel('error')
        ax.set_zlim3d(bottom=0, top=None, auto=True)
    plt.show()

    # find the best parameters for test error rate before and after condensing
    best_uncondensed = min(results, key=lambda tup: results[tup][1])
    print(f"Best parameters when uncondensed are k = {best_uncondensed[0]}, p = {best_uncondensed[1]}")
    best_condensed = min(results, key=lambda tup: results[tup][3])
    print(f"Best parameters when condensed are k = {best_condensed[0]}, p = {best_condensed[1]}")

def filter_label_and_extract_X_y(data, label1, label2):
    filtered_data = data.where((data['label'] == label1) | (data['label'] == label2)).dropna()
    X: list[tuple[float, float]] = list((x['f2'], x['f3']) for (_, x) in filtered_data.iterrows())
    y: list[str] = list(x['label'] for (_, x) in filtered_data.iterrows())
    return X, y

if __name__ == '__main__':
    # read data from text file, keep only feature 1, feature 2, label
    data = (pd.read_csv("iris.txt", sep=' ', names=["f1", "f2", "f3", "f4", "label"])
            .filter(items=["f2", "f3", "label"]))

    # for first experiment, filter only virginica and versicolor
    experiment(*filter_label_and_extract_X_y(data, 'Iris-virginica', 'Iris-versicolor'))
    # for second experiment, filter only virginica and setosa
    experiment(*filter_label_and_extract_X_y(data, 'Iris-virginica', 'Iris-setosa'))
