from project import utils


class TreeNode:

    def __init__(self, data=None, children=[], threshold=None, is_leaf=False, column=0):
        self.prediction = data
        self.children = children
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.column = column


class C4point5:

    def __init__(self, name="test"):
        self.name = name
        self.tree = None

    def fit(self, X_train, y_train):
        training_data = X_train.values.tolist()
        y_train = y_train.values.tolist()
        # Join the lists and transpose them
        for idx, feat in enumerate(training_data):
            feat.append(y_train[idx])

        training_data = list(map(list, zip(*training_data)))

        self.tree = self.build_tree(training_data)

    def build_tree(self, training_data):
        """
            Just get the best once, no recursion yet
        """
        gain, threshold, column, splits = utils.get_best_gain(training_data)
        output = dict([(0, 'length'), (1, 'width'), (2, 'thick'), (3, 'surface'), (4, 'mass'),
                       (5, 'compact'), (6, 'hardness'), (7, 'shell top'), (8, 'water'), (9, 'carb')])

        print('Best gain is: {}'.format(gain))
        print('Best threshold is: {}'.format(threshold))
        print('Best column is: {}'.format(output[column]))
        return TreeNode('root', ['c_americana','c_cornuta'], threshold, True, column)

    def predict(self, X_test):
        predictions = []
        for index, row in X_test.iterrows():
            predictions.append(self.predict_class(row))

        return predictions

    def predict_class(self, row):
        thresh = self.tree.threshold
        val = float(row.iloc[self.tree.column])
        if val < thresh:
            predicted = self.tree.children[0]
        else:
            predicted = self.tree.children[1]

        return predicted
