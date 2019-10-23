from sklearn.model_selection import train_test_split

def run_classifier(df):
    print("\n-- Running my implementation --")
    X = df.drop(columns=['variety'])
    y = df['variety'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    classifier = MyClassifier()
    classifier.fit(X_train, y_train)
    predicted_classes = classifier.predict(X_test)

    correct_classes = y_test.tolist()
    score = 0
    for pred, correct in zip(predicted_classes, correct_classes):
        if pred == correct:
            score += 1

    print("Accuracy is: ")
    print(score / len(predicted_classes))



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
