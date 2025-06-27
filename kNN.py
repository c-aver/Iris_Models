import itertools
from typing import Literal


Point = tuple[float, ...]


class KNNClassifier:
    def __init__(self, k: int, p: int | Literal['∞']):
        self.d: int | None = None
        self.labelled_points: list[tuple[Point, str]] | None = None
        self.k: int = k
        self.p: int | Literal["∞"] = p

    def __dist(self, p1: Point, p2: Point) -> float:
        if self.p == '∞':
            return max(map(lambda tup: abs(tup[0] - tup[1]), zip(p1, p2)))
        return sum(map(lambda tup: abs(tup[0] - tup[1])**self.p, zip(p1, p2)))**(1/self.p)

    def condense(self) -> int:
        label1, label2 = tuple(list(set((p[1] for p in self.labelled_points))))
        labelled1 = list(point for (point, label) in self.labelled_points if label == label1)
        labelled2 = list(point for (point, label) in self.labelled_points if label == label2)
        closest_pair = min(itertools.product(labelled1, labelled2), key=lambda pair: self.__dist(pair[0], pair[1]))
        eps = self.__dist(closest_pair[0], closest_pair[1])
        t = [self.labelled_points[0]]
        for p in self.labelled_points:
            dist = self.__dist(min(t, key=lambda lp: self.__dist(lp[0], p[0]))[0], p[0])
            if dist > eps:
                t.append(p)
        self.labelled_points = t
        return len(t)

    def fit(self, points: list[Point], labels: list[str]):
        self.d = len(points[0])
        assert (all(map(lambda d: d == self.d, map(len, points))))
        self.labelled_points = list(zip(points, labels))

    def predict(self, point: Point) -> str:
        if self.labelled_points is None:
            raise Exception("Cannot predict before fitting, call fit first")
        self.labelled_points.sort(key=lambda p: self.__dist(p[0], point))
        return max(set(label for label in self.labelled_points[0:self.k]), key=lambda p: p[0])[1]

