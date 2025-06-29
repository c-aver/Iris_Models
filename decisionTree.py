
import numpy as np
import pandas as pd

# loading the text file
df = pd.read_csv('iris.txt', header=None, names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class'],sep=r'\s+'
)

df = df[df['class'].isin(['Iris-versicolor', 'Iris-virginica'])]

# col 1 - sepal_wid, col 2 - petal_len
X = df[['sepal_wid', 'petal_len']].to_numpy()
y = (df['class'] == 'Iris-virginica').astype(int).to_numpy()   # 0/1
# ----------------------------------------------------------

from sklearn import tree
from sklearn.model_selection import train_test_split

def run_once(seed=None):
    #spliting the data 50-50, loading a tree with 2 levels, computing error
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed)

    clf = tree.DecisionTreeClassifier(max_depth=2,
                                      criterion='entropy',
                                      random_state=seed)
    clf.fit(X_tr, y_tr)
    return 1 - clf.score(X_tr, y_tr), 1 - clf.score(X_te, y_te), clf


# 50 runs
rng = np.random.default_rng(2025)
train_errs, test_errs = [], []
best_clf = None

for _ in range(50):
    tr, te, clf = run_once(seed=int(rng.integers(1e9)))
    train_errs.append(tr)
    test_errs.append(te)
    if best_clf is None:       #save the first tree
        best_clf = clf

print(f"Empirical error (train) mean ± std: "
      f"{np.mean(train_errs):.3f} ± {np.std(train_errs):.3f}")
print(f"True error      (test)  mean ± std: "
      f"{np.mean(test_errs):.3f} ± {np.std(test_errs):.3f}")

# creating the tree
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
tree.plot_tree(best_clf,
               feature_names=['Sepal width', 'Petal length'],
               class_names=['Versicolor', 'Virginica'],
               filled=True, rounded=True)
plt.tight_layout()
plt.savefig("tree_q2.png", dpi=300)
plt.show()
