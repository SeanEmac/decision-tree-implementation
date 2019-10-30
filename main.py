import pandas as pd

import argparse
import my_impl
import scikit_impl
import sys


def main():
    file = setup_cli()
    print("\nRunning the program on the file %s:" % file)
    df = prepare_data(file)
    my_impl.run_classifier(df)
    scikit_impl.run_knn(df)


def setup_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        help="The path to the csv file, default is data/hazelnuts.txt",
                        default="data/hazelnuts.txt")
    args = parser.parse_args()
    return args.input


def prepare_data(file):
    print("Preparing the data")

    try:
        df = pd.read_csv(file, sep='\t', header=None)
        df = df.transpose()
        df.drop(df.columns[0], axis=1, inplace=True)
        print("Finished preparing data")
        return df
    except:
        sys.exit("File: %s cannot be found \nExiting program" % file)


if __name__ == '__main__':
    main()
