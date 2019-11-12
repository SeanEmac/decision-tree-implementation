# scikit_impl.py
import numpy as np

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def run_knn(df):
    """
        Standard Scikit implementation of a decision tree,
        return the cross validated accuracy and confusion matrix
        so that we can compare it with my implementation.
    """
    print("\nRunning scikit-learn implementation:")
    X = df.drop(df.columns[len(df.columns) - 1], axis=1)
    y = df.iloc[:, len(df.columns) - 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=8)
    dtree.fit(X_train, y_train)
    dtree_predict = dtree.predict(X_test)

    cv_scores = cross_val_score(dtree, X, y, cv=10)

    return {
        'accuracy': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'confusion': confusion_matrix(y_test, dtree_predict)
    }
