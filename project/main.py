# main.py
import easygui
import pandas as pd
import sys

from project import my_impl, scikit_impl


def main():
    """
        Driver method for the application, create a simple GUI to select the file,
        parse the data and pass the same data frame to mine and scikit's implementation.
        Compare the 2 results.

    """
    file = easygui.fileopenbox("Please choose a file eg. hazelnuts.txt")
    df = prepare_data(file)
    my_results = my_impl.run_classifier(df)
    scikit_results = scikit_impl.run_knn(df)
    compare_results(file.split('/')[-1], my_results, scikit_results)


def compare_results(file, my_results, scikit_results):
    """
        Format the accuracies to 2 decimal places
        Open a GUI so print the results of both algorithms
    """
    formatted_results = ""
    for accuracy in my_results:
        formatted_results += ("%.2f " % accuracy)

    my_mean = (sum(my_results) / len(my_results))

    easygui.msgbox('The learning algorithm has successfully run on {a}!\n\n'
                   'Scores after 10 runs of my implementation:\n{b:}\n\n'
                   'Mean Accuracy of my implementation is: {c:.2f}\n'
                   'Mean Accuracy of scikit implementation is: {d:.2f}\n\n'
                   'The predictions are available in the /data/results directory'
                   .format(a=file, b=formatted_results, c=my_mean, d=scikit_results['mean_accuracy']),
                   'Results')


def prepare_data(file):
    """
        Read in the text data file, if there are any problems it should
        gracefully exit
    """
    try:
        df = pd.read_csv(file, sep='\t', header=None)
        df = df.transpose()
        df.drop(df.columns[0], axis=1, inplace=True)
        return df

    except:
        sys.exit("File: %s cannot be found \nExiting program" % file)


if __name__ == '__main__':
    main()
