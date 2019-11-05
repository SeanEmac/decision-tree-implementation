from project import utils


class TreeNode:

    def __init__(self, data, children=[]):
        self.data = data
        self.children = children
        self.threshold = 15.00


class C4point5:

    def __init__(self, name="test"):
        self.name = name
        self.tree = None

    def fit(self, X_train, y_train):
        self.tree = self.build_tree(X_train, y_train)


    def build_tree(self, X_train, y_train):
        root = TreeNode('One level tree', ['c_avellana', 'c_americana'])
        return root


    def predict(self, X_test):
        predictions = []
        for index, row in X_test.iterrows():
            predictions.append(self.predict_class(row))

        return predictions


    def predict_class(self, row):
        thresh = self.tree.threshold
        val = float(row.iloc[0])
        if val < thresh:
            predicted = self.tree.children[0]
        else:
            predicted = self.tree.children[1]

        return predicted
