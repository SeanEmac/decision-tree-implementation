from project import utils

class TreeNode:

    def __init__(self, data=None, children=[], threshold=None, is_leaf=False):
        self.data = data
        self.children = children
        self.threshold = threshold
        self.is_leaf = is_leaf


class C4point5:

    def __init__(self, name="test"):
        self.name = name
        self.tree = None

    def fit(self, X_train, y_train):
        """
            For now, lets only consider the first attribute 'length'
        """

        length = X_train.iloc[:, 0]
        length = length.values.tolist()
        y_train = y_train.values.tolist()

        self.tree = self.build_tree(length, y_train)


    def build_tree(self, data, classes):
        class_counts = utils.count_classes(classes)
        majority = utils.majority_class(classes)
        return TreeNode('Root Node', ['c_americana', 'c_cornuta'], 15.00, True)


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
