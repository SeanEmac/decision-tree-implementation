import pandas as pd
import sys

from project.c45 import C45
from sklearn.model_selection import train_test_split


def run_classifier(df):
    """
        Run calculate_results 10 times and print the results
    """
    try:
        X = df.drop(df.columns[len(df.columns) - 1], axis=1)
        y = df.iloc[:, len(df.columns) - 1]

    except:
        sys.exit("There was a problem processing the file")

    results = []
    for i in range(10):
        results.append(calculate_results(i, X, y))

    accuracies = []
    for result in results:
        accuracies.append(sum(result['score']) / len(result['predicted']))

    return accuracies


def calculate_results(i, X, y):
    """
        Use train_test_split from scikit to shuffle and split the data
        into training and test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    classifier = C45()
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    actual = y_test.tolist()
    score = []
    for pred, corr in zip(predicted, actual):
        score.append(1) if pred == corr else score.append(0)

    results = {'actual': actual, 'predicted': predicted, 'score': score}
    export_results(i, results)
    return results


def export_results(i, results):
    """
        Print each of the 10 runs to csv files
    """
    filepath = 'data/results/predictions{0}.csv'.format(i + 1)
    df = pd.DataFrame(results)
    df.to_csv(filepath)
