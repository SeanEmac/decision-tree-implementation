import math


def calc_entropy(y):
    """
        Entropy is a measure of the uncertainty of the variable.
        Basically, for each attribute class we are trying to predict,
        we sum their (- proportion) * log(proportion)
        where proportion is (number of occurrences / total number of values)
    """

    total_count = len(y)
    class_counts = count_classes(y)
    entropy = 0

    for item in class_counts:
        proportion = item / total_count
        entropy += -proportion * math.log(proportion, 2)

    return entropy


def calc_info_gain(full, left, right):
    """
        Reduction in entropy from partitioning at this attribute
        total entropy of the set - sum of entropy for that value
        entropy(full) - (prop(v) * Ent(v)) for each value v
    """
    propLeft = len(left) / len(full)
    propRight = len(right) / len(full)
    full_entropy = calc_entropy(full)
    info_gain = full_entropy - (propLeft * calc_entropy(left)
                                - propRight * calc_entropy(right))

    return info_gain


def majority_class(y):
    classes = count_classes(y)
    return max(classes, key=classes.get)


def count_classes(y):
    class_counts = {}

    for c in y:
        if c in class_counts:
            class_counts[c] = class_counts[c] + 1
        else:
            class_counts[c] = 1

    return class_counts
