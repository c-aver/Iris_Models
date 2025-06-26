import itertools
import math
from typing import Literal
from sklearn.model_selection import train_test_split
import pandas as pd

from kNN import KNNClassifier

ks: list[int] = [1, 3, 5, 7, 9]
ps: list[int | Literal['∞']] = [1, 2, '∞']


def error_rate(classifier: KNNClassifier, X, y):
    errors = sum(1 for point, label in zip(X, y) if (classifier.predict(point) != label))
    return errors / len(X)


def experiment(data):
    X: list[tuple[float, float]] = list((x['f2'], x['f3']) for (_, x) in data.iterrows())
    y: list[str] = list(x['label'] for (_, x) in data.iterrows())

    print(f"Starting experiment with classes: {set(y)}")
    results: dict[tuple[int, int | Literal['∞']], list[float]] = {}
    for k, p in itertools.product(ks, ps):
        results[(k, p)] = [0, 0, 0, 0]

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)

        for k, p in itertools.product(ks, ps):
            classifier = KNNClassifier(k, p)
            classifier.fit(X_train, y_train)
            results[(k, p)][0] += error_rate(classifier, X_train, y_train)
            results[(k, p)][1] += error_rate(classifier, X_test, y_test)
            classifier.condense()
            results[(k, p)][2] += error_rate(classifier, X_train, y_train)
            results[(k, p)][3] += error_rate(classifier, X_test, y_test)

    for k, p in itertools.product(ks, ps):
        results[(k, p)][0] /= 100
        results[(k, p)][1] /= 100
        results[(k, p)][2] /= 100
        results[(k, p)][3] /= 100
        print(f"For k = {k}, p = {p}:")
        print("Before condensing:")
        print(f"\tTrain error rate = {results[(k, p)][0]:.4f}")
        print(f"\tTest error rate = {results[(k, p)][1]:.4f}")
        print(f"\tDifference is {results[(k, p)][1] - results[(k, p)][0]:.4f}")
        print("After condensing:")
        print(f"\tTrain error rate = {results[(k, p)][2]:.4f}")
        print(f"\tTest error rate = {results[(k, p)][3]:.4f}")
        print(f"\tDifference is {results[(k, p)][3] - results[(k, p)][2]:.4f}")

    best = min(results, key=lambda tup: results[tup][1])
    print(f"Best parameters are k = {best[0]}, p = {best[1]}")


if __name__ == '__main__':
    data = (pd.read_csv("iris.txt", sep=' ', names=["f1", "f2", "f3", "f4", "label"])
            .filter(items=["f2", "f3", "label"]))
    filtered_data = (data.where((data['label'] == 'Iris-virginica') | (data['label'] == 'Iris-versicolor'))
                     .dropna())
    experiment(filtered_data)
    filtered_data = (data.where((data['label'] == 'Iris-virginica') | (data['label'] == 'Iris-setosa'))
                     .dropna())
    experiment(filtered_data)
