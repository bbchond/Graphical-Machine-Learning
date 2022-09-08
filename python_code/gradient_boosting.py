import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn.model_selection import train_test_split

X, y = datasets.make_hastie_10_2(n_samples=4000, random_state=1)
labels, y = np.unique(y, return_inverse=True)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.8, random_state=0)
origin_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
}
plt.figure()
for label, color, setting in [
    ("No shrinkage", "orange", {"learning_rate": 1.0, "subsample": 1.0}),
    ("learning_rate=0.2", "turquoise", {"learning_rate": 0.2, "subsample": 1.0}),
    ("subsample=0.5", "blue", {"learning_rate": 1.0, "subsample": 0.5}),
    ("learning_rate=0.2, subsample=0.5", "gray", {"learning_rate": 0.2, "subsample": 0.5}),
    ("learning_rate=0.2, max_features=2", "magenta", {"learning_rate": 0.2, "max_features": 2}),
]:
    params = dict(origin_params)
    params.update(setting)
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    test_deviance = np.zeros((params["n_estimators"],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        test_deviance[i] = clf.loss_(y_test, y_pred)
    plt.plot(
        (np.arange(test_deviance.shape[0]) + 1)[:: 5],
        test_deviance[:: 5],
        "-",
        color=color,
        label=label
    )
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Test Set Deviance')
plt.show()
