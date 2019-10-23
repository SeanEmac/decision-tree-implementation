import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def run_knn(df):
    print("\n-- Running scikit-learn implementation --")
    X = df.drop(columns=['variety'])
    y = df['variety'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    knn_predict = knn.predict(X_test)
    cv_scores_knn = cross_val_score(knn, X, y, cv=10)

    print('10 fold score: {}'.format(np.mean(cv_scores_knn)))
    print(confusion_matrix(y_test, knn_predict))