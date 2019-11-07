import math


def calc_entropy(data):
    """
        Entropy is a measure of the uncertainty of the variable.
        Basically, for each attribute class we are trying to predict,
        we sum their (- proportion) * log(proportion)
        where proportion is (number of occurrences / total number of values)
    """
    total_count = len(data[0])
    class_counts = count_classes(data)
    entropy = 0

    for item in class_counts.values():
        proportion = item / total_count
        entropy -= proportion * math.log(proportion, 2)

    return entropy


def calc_info_gain(full, left, right):
    """
        Reduction in entropy from partitioning at this attribute
        total entropy of the set - sum of entropy for that value
        entropy(full) - (prop(v) * Ent(v)) for each value v
    """
    propLeft = len(left[0]) / len(full[0])
    propRight = len(right[0]) / len(full[0])
    full_entropy = calc_entropy(full)
    info_gain = full_entropy - propLeft * calc_entropy(left) - propRight * calc_entropy(right)

    return info_gain


def count_classes(data):
    """
        Loops through classes and adds them to a dict
        if already in dict then increment the counter
    """
    class_counts = {}
    val = data[-1]
    for c in val:
        if c in class_counts:
            class_counts[c] = class_counts[c] + 1
        else:
            class_counts[c] = 1

    return class_counts


def get_best_gain(training_data):
    """
        Very messy for now transposing the data in order to sort and iterate easily.
        For each feature, for each value, calculate the information gain if we did split on this value
        if it is higher than the previous best, this value now becomes the threshold
        Return the one that gives the highest information gain.
    """
    splits = []
    best_gain = -999
    best_threshold = -999
    best_column = -999

    for idx in range(len(training_data)):
        if idx < (len(training_data) - 1):
            # Hacked way of sorting
            training_data = list(map(list, zip(*training_data)))
            training_data.sort(key=lambda x: float(x[idx]))
            training_data = list(map(list, zip(*training_data)))
            feature = [float(i) for i in training_data[idx]]
            for i in range(len(feature) - 1): # -1 because we check + 1
                if feature[i] != feature[i + 1]:
                    threshold = (feature[i] + feature[i + 1]) / 2
                    left = []
                    right = []
                    for j in range(len(feature)):
                        # Split the data into 2 subsets
                        if feature[j] > threshold:
                            temp = []
                            for ft in training_data:
                                temp.append(ft[j])
                            right.append(temp)
                        else:
                            temp = []
                            for ft in training_data:
                                temp.append(ft[j])
                            left.append(temp)

                    gain = calc_info_gain(training_data, list(map(list, zip(*left))), list(map(list, zip(*right))))
                    if gain >= best_gain:
                        splits = [list(map(list, zip(*left))), list(map(list, zip(*right)))]
                        best_gain = gain
                        best_threshold = threshold
                        best_column = idx

    return best_gain, best_threshold, best_column, splits
