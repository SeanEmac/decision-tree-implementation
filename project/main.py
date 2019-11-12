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
    my_formatted_results = ""
    for accuracy in my_results:
        my_formatted_results += ("%.2f " % accuracy)

    my_mean = (sum(my_results) / len(my_results))
    my_high = max(my_results)
    my_low = min(my_results)
    my_diff = my_high - my_low

    sci_formatted_results = ""
    for accuracy in scikit_results['accuracy']:
        sci_formatted_results += ("%.2f " % accuracy)

    sci_high = max(scikit_results['accuracy'])
    sci_low = min(scikit_results['accuracy'])
    sci_diff = sci_high - sci_low

    easygui.msgbox('The learning algorithm has successfully run on {a}!\n\n'
                   'Mean Accuracy of my implementation is: {c:.2f}\n'
                   'Scores: {b:}\n'
                   'Highest Accuracy: {d:.2f}\nLowest Accuracy: {e:.2f}\n'
                   'Range: {f:.2f}\n\n'
                   'Mean Accuracy of scikit implementation is: {g:.2f}\n'
                   'Scores: {z:}\n'
                   'Highest Accuracy: {h:.2f}\nLowest Accuracy: {i:.2f}\n'
                   'Range: {j:.2f}\n\n'
                   'The predictions are available in the /data/results directory'
                   .format(a=file, b=my_formatted_results, c=my_mean, d=my_high, e=my_low, f=my_diff,
                           g=scikit_results['mean_accuracy'], h=sci_high, i=sci_low, j=sci_diff, z=sci_formatted_results),
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
