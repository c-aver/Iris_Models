import math
from abc import ABC, abstractmethod
from typing import Type

Point = tuple[float, ...]


def entropy(s: list[tuple[Point, int]] | list[int]):
    if len(s) == 0:
        return 0
    if isinstance(s[0], tuple):
        s = [l for p, l in s]
    res = 0
    for v in set(s):
        p = len([l for l in s if l == v]) / len(s)
        res -= p * math.log2(p)
    return res


def most_common(l: list):
    res = None
    max_c = 0
    for v in set(l):
        c = l.count(v)
        if c > max_c:
            max_c = c
            res = v
    return res


class DecisionTreeClassifier:
    class TreeNode(ABC):
        @abstractmethod
        def predict(self, p: Point) -> int:
            pass

    class DecisionNode(TreeNode):
        def __init__(self, d: int, thresh: float, t, f):
            self.d: int = d
            self.thresh: float = thresh
            self.t = t
            self.f = f

        def split(self, ps: list[tuple[Point, int]]):
            return ([(p, l) for (p, l) in ps if p[self.d] >= self.thresh],
                    [(p, l) for (p, l) in ps if not p[self.d] >= self.thresh])

        def predict(self, p: Point) -> int:
            if p[self.d] >= self.thresh:
                return self.t.predict(p)
            else:
                return self.f.predict(p)

        def set_t(self, node):
            self.t = node

        def set_f(self, node):
            self.f = node

    class LeafNode(TreeNode):
        def __init__(self, label: int):
            self.label: int = label

        def predict(self, _: Point) -> int:
            return self.label

    def __init__(self, max_depth: int | None = None):
        self.max_depth: int | None = max_depth
        self.root = self.LeafNode(0)
        self.d: int | None = None

    def set_root(self, node):
        self.root = node

    def fit_once(self, train: list[tuple[Point, int]]) -> bool:
        self.d = len(train[0][0])
        leaves = []
        queue = [(self.root, 0, train, self.set_root)]
        while len(queue) > 0:
            node, depth, s, pos = queue.pop()
            if depth >= self.max_depth - 1:  # shouldn't split if children would be too deep
                continue
            if isinstance(node, self.DecisionNode):
                s_t, s_f = node.split(s)
                queue.append((node.t, depth + 1, s_t, node.set_t))
                queue.append((node.f, depth + 1, s_f, node.set_f))
            else:
                leaves.append((node, s, pos))

        max_information_gain = -math.inf
        max_gain_replacement = None
        for leaf, s, pos in leaves:
            s_ent = entropy(s)
            for d in range(self.d):
                critical_values = []
                last_label = None
                for p, l in sorted(s, key=lambda p: p[d]):
                    if l != last_label:
                        critical_values.append(p[d])
                    last_label = l
                for v in critical_values:
                    candidate_node = self.DecisionNode(d, v, self.LeafNode(0), self.LeafNode(0))
                    s_t, s_f = candidate_node.split(s)
                    s_cond_ent = (entropy(s_t) * (len(s_t) / len(s))) + (entropy(s_f) * (len(s_f) / len(s)))
                    information_gain = s_ent - s_cond_ent
                    if information_gain > max_information_gain:
                        max_information_gain = information_gain
                        t_common = most_common([l for p, l in s_t])
                        if t_common is None:
                            t_common = 0
                        f_common = most_common([l for p, l in s_f])
                        if f_common is None:
                            f_common = 0
                        candidate_node.t = self.LeafNode(t_common)
                        candidate_node.f = self.LeafNode(f_common)
                        max_gain_replacement = (pos, candidate_node)
        if max_gain_replacement is not None:
            set_function, new = max_gain_replacement
            set_function(new)
            return True
        return False

    def fit(self, train: list[tuple[Point, int]]):
        while self.fit_once(train):
            pass

    def predict(self, p: Point) -> int:
        assert len(p) == self.d
        return self.root.predict(p)
