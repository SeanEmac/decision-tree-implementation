from project import utils


class TreeNode:
    def __init__(self, is_leaf=False, prediction=None,  threshold=None, column=0, less=None, greater=None):
        """
            This TreeNode represents a decision node in my Tree. It can be a leaf that predicts
            a type of nut, or it can hold the information needed to traverse.
        """
        # If it's a leaf we can make a prediction
        self.is_leaf = is_leaf
        self.prediction = prediction

        # Data needed to traverse the tree
        self.threshold = threshold
        self.column = column

        # Children of thi TreeNode
        self.less = less
        self.greater = greater


class C45:
    def __init__(self):
        self.tree = None

    def fit(self, X_train, y_train):
        """
            Data frames became awkward to sort and cast to float,
            here I just join the training X & y training sets into one big list
            of 10 * 201 inner lists, then build the tree.
        """
        training_data = X_train.values.tolist()
        y_train = y_train.values.tolist()
        for idx, feat in enumerate(training_data):
            feat.append(y_train[idx])

        training_data = list(map(list, zip(*training_data)))
        self.tree = self.build_tree(training_data)

    def build_tree(self, training_data):
        """
            Count the number of each type of nut left in the data set,
            If we only have 1 type of nut left, then we can make a decision,
            Create a leaf node predicting the only nut left.

            If there is more than one type of nut left, then we need to perform another split.
            Get the best feature and threshold to split on, then recursively
            call build_tree on the less and greater than subsets.
        """
        class_counts = utils.count_classes(training_data)
        if len(class_counts) == 1:
            cls = list(class_counts.keys())[0]
            return TreeNode(True, cls)
        else:
            gain, threshold, column, splits = utils.get_best_gain(training_data)
            return TreeNode(False, None, threshold, column,
                            self.build_tree(splits[0]), self.build_tree(splits[1]))

    def predict(self, X_test):
        """
            For each row in our X_test data set, we call predict_class on that row.
            We return a list of predictions that can be compared against the y_test list.
        """
        predictions = []
        for _, row in X_test.iterrows():
            predictions.append(self.predict_class(row, self.tree))

        return predictions

    def predict_class(self, row, tree_node):
        """
            If we are at a leaf node, then return the predicted type of nut.

            Else, traverse the tree until we get to a leaf. We traverse the tree by
            going left if the value to predict is less than the threshold, or right
            if it's greater.
        """
        if tree_node.is_leaf:
            return tree_node.prediction
        else:
            if float(row[tree_node.column + 1]) < tree_node.threshold:
                return self.predict_class(row, tree_node.less)
            else:
                return self.predict_class(row, tree_node.greater)
