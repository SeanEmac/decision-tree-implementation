from project import utils


class TreeNode:

    def __init__(self, prediction=None, children=[], threshold=None, column=0, is_leaf=False):
        self.prediction = prediction
        self.children = children
        self.threshold = threshold
        self.column = column
        self.is_leaf = is_leaf


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
        class_counts = utils.count_classes(training_data)
        if len(class_counts) == 1:
            cls = list(class_counts.keys())[0]
            return TreeNode(cls, [], None, 0, True)
        else:
            gain, threshold, column, splits = utils.get_best_gain(training_data)
            if gain < 0.3:
                cls = self.get_majority(splits)
                return TreeNode(cls, [], None, 0, True)
            else:
                return TreeNode('not leaf', [self.build_tree(splits[0]), self.build_tree(splits[1])],
                            threshold, column, False)

    def predict(self, X_test):
        # self.printTree()
        predictions = []
        for index, row in X_test.iterrows():
            predict = self.predict_class(row, self.tree)
            predictions.append(predict)

        return predictions

    def predict_class(self, row, node):
        if node.is_leaf:
            return node.prediction
        else:
            a = float(row[node.column + 1])
            tresh = node.threshold
            if a < tresh:
                return self.predict_class(row, node.children[0])
            else:
                return self.predict_class(row, node.children[1])

    def get_majority(self, splits):
        cls = utils.count_classes(splits[0])
        cls.update(utils.count_classes(splits[1]))
        a = max(cls, key=cls.get)
        return a


    def printTree(self):
        self.printNode(self.tree)

    def printNode(self, node, indent=""):
        output = dict([(0, 'length'), (1, 'width'), (2, 'thick'), (3, 'surface'), (4, 'mass'),
                       (5, 'compact'), (6, 'hardness'), (7, 'shell top'), (8, 'water'), (9, 'carb')])
        if not node.is_leaf:
            leftChild = node.children[0]
            rightChild = node.children[1]
            if leftChild.is_leaf:
                print(indent + output[node.column] + " <= " + str(node.threshold) + " : " + leftChild.prediction)
            else:
                print(indent + output[node.column] + " <= " + str(node.threshold))
                self.printNode(leftChild, indent + "	")

            if rightChild.is_leaf:
                print(indent + output[node.column] + " > " + str(node.threshold) + " : " + rightChild.prediction)
            else:
                print(indent + output[node.column] + " > " + str(node.threshold))
                self.printNode(rightChild, indent + "	")
