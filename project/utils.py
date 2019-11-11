# utils.py
import math
"""
    This file holds helper methods needed to implement C4.5
    - Calculate Entropy
    - Calculate Information Gain
    - Count the classes in the data set
    - Find the attribute & value that give the best information gain 
"""


def calc_entropy(data):
    """
        Entropy is a measure of the uncertainty of the variable.
        Basically, for each attribute class we are trying to predict (3 nuts),
        we sum their - (proportion) * log2(proportion)
        where proportion is (number of occurrences / total number of values)
    """
    total_count = len(data[0])
    class_counts = count_classes(data)
    ent_S = 0

    for class_i in class_counts.values():
        prop_i = class_i / total_count
        ent_S = - (prop_i * math.log(prop_i, 2))

    return ent_S


def calc_info_gain(full, less, greater):
    """
        Reduction in entropy from partitioning at this attribute
        total entropy of the set - sum of entropy for that value
        entropy(full) - (prop(v) * Ent(v)) for each value v
        In this case, because all attributes are continuous, we only have 2 v's (< or >)
        If it was discrete (cold, warm, hot) then we would need a loop for each v
    """
    v1_proportion = len(less[0]) / len(full[0])
    v2_proportion = len(greater[0]) / len(full[0])
    S_entropy = calc_entropy(full)
    v1_entropy = calc_entropy(less)
    v2_entropy = calc_entropy(greater)

    gain_S_A = S_entropy - (v1_proportion * v1_entropy) - (v2_proportion * v2_entropy)

    return gain_S_A


def count_classes(data):
    """
        Function to count the number of occurrences of items in a list.
        Loops through classes and adds them to a dict
        if already in dict then increment the counter
    """
    class_counts = {}
    class_column = data[-1]
    for cls in class_column:
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1

    return class_counts


def get_best_gain(dataset_S):
    """
        For each attribute, for each value, calculate the information gain if we did split on this value
        if it is higher than the previous best, this value now becomes the best value
        Return the one that gives the highest information gain.
    """
    best_split = []
    best_gain = -999
    best_value = -999
    best_attribute = -999

    # For each attribute
    for col_idx in range(len(dataset_S) - 1):
        dataset_S = sort_by_column(dataset_S, col_idx)
        attribute = [float(i) for i in dataset_S[col_idx]]

        # Find the best value to split on
        for row_idx in range(len(attribute) - 1):
            if attribute[row_idx] != attribute[row_idx + 1]:
                # Mid point between 2 values
                value = (attribute[row_idx] + attribute[row_idx + 1]) / 2
                split = split_data_set(attribute, value, dataset_S)
                # See what gain we would get if we split on this attribute & value
                gain = calc_info_gain(dataset_S, split[0], split[1])

                if gain > best_gain:
                    # If this new gain is the best, save its information
                    best_gain = gain
                    best_value = value
                    best_attribute = col_idx
                    best_split = split

    return best_gain, best_value, best_attribute, best_split


def sort_by_column(dataset_S, column_idx):
    """
        Messy but it works, transpose the data so its easier to sort,
        sort the whole data set based on attribute at column_idx.
        Transpose again when we return it.
    """
    dataset_S = list(map(list, zip(*dataset_S)))
    dataset_S.sort(key=lambda x: x[column_idx])
    return list(map(list, zip(*dataset_S)))


def split_data_set(attribute, value, dataset_S):
    """
        Given a column index of an attribute and a value to split on,
        partition the data into 2 parts, less than and greater than the value
    """
    less_than = []
    greater_than = []
    for rowIdx in range(len(attribute)):
        if attribute[rowIdx] > value:
            temp = []
            for ft in dataset_S:
                temp.append(ft[rowIdx])
            greater_than.append(temp)
        else:
            temp = []
            for ft in dataset_S:
                temp.append(ft[rowIdx])
            less_than.append(temp)
    return [list(map(list, zip(*less_than))), list(map(list, zip(*greater_than)))]
