import math


def calc_entropy(y):
    """
        Entropy is a measure of the uncertainty of the variable.
        Basically, for each attribute class we are trying to predict,
        we sum their (- proportion) * log(proportion)
        where proportion is (number of occurrences / total number of values)
    """

    total_count = y.count()
    class_counts = y.value_counts()
    entropy = 0

    for item in class_counts:
        proportion = item / total_count
        entropy += -proportion * math.log(proportion, 2)

    return entropy


def calc_info_gain(attribute):
    # Reduction in entropy from partitioning at this attribute
    # total entropy of the set - sum of entropy for that value

    info_gain = 1
    return info_gain
