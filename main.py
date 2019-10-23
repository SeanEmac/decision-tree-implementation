import pandas as pd

import my_impl
import scikit_impl


def main():
    print("\n-- Running the main file --")
    df = prepare_data()
    my_impl.run_classifier(df)
    scikit_impl.run_knn(df)


def prepare_data():
    print("\n-- Preparing the data --")

    headers = ['sample_id', 'length', 'width', 'thickness',
               'surface_area', 'mass', 'compactness', 'hardness',
               'shell_top_radius', 'water_content', 'carbohydrate_content', 'variety']

    df = pd.read_csv('data/hazelnuts.txt', sep='\t', header=None)
    df = df.transpose()
    df.columns = headers
    df = df.drop(columns=['sample_id'])

    print("Finished preparing")
    return df


if __name__ == '__main__':
    main()
