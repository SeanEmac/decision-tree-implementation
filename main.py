import pandas as pd

import easygui
import my_impl
import scikit_impl
import sys


def main():
    file = easygui.fileopenbox("Please choose a file eg. hazelnuts.txt")
    df = prepare_data(file)
    my_results = my_impl.run_classifier(df)
    scikit_results = scikit_impl.run_knn(df)
    compare_results(file.split('/')[-1], my_results, scikit_results)


def compare_results(file, my, sci):
    easygui.msgbox('The learning algorithm has successfully run on {a}!\n'
                   'Accuracy of my implementation is: {b:.2f}\n'
                   'Accuracy of scikit implementation is: {c:.2f}\n'
                   'The predictions are available at data/predictions.csv.'
                   .format(a=file, b=my['accuracy'], c=sci['accuracy']),
                   'Results')

def prepare_data(file):
    try:
        df = pd.read_csv(file, sep='\t', header=None)
        df = df.transpose()
        df.drop(df.columns[0], axis=1, inplace=True)
        return df

    except:
        sys.exit("File: %s cannot be found \nExiting program" % file)


if __name__ == '__main__':
    main()
