from project import utils


class C45:

    def __init__(self, name="test"):
        self.name = name

    def fit(self, X_train, y_train):
        print("Training a dumb predictor")
        print(utils.calc_entropy())

    def predict(self, X_test):
        print("Predicting on test data")
        r, c = X_test.shape
        predictions = []
        for x in range(0, r):
            predictions.append(self.predict_class(1))

        return predictions

    def predict_class(self, value):
        return "c_avellana"