import pandas as pd
import sys

from sklearn.model_selection import train_test_split


def run_classifier(df):
    try:
        X = df.drop(df.columns[len(df.columns) - 1], axis=1)
        y = df.iloc[:, len(df.columns) - 1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    except:
        sys.exit("There was a problem processing the file")

    classifier = MyClassifier()
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    correct = y_test.tolist()
    score = []
    for pred, corr in zip(predicted, correct):
        if pred == corr:
            score.append(1)
        else:
            score.append(0)

    export_results(correct, predicted, score)
    accuracy = sum(score) / len(predicted)
    return {
        'accuracy': accuracy
    }


def export_results(correct, predicted, score):
    data = {
        'actual': correct,
        'predicted': predicted,
        'correct': score
    }
    df = pd.DataFrame(data)
    df.to_csv(r'data/predictions.csv')


class MyClassifier:

    def __init__(self, name="test"):
        self.name = name

    def fit(self, X_train, y_train):
        print("Training a dumb predictor")

    def predict(self, X_test):
        print("Predicting on test data")
        r, c = X_test.shape
        predictions = []
        for x in range(0, r):
            predictions.append(self.predict_class(1))

        return predictions

    def predict_class(self, value):
        return "c_avellana"
