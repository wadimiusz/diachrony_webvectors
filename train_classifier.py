import argparse
import os

import joblib
import pandas as pd

from classifier import ShiftClassifier
from utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", dest='embeddings_dir', type=str,
                        help="Path to an existing directory that contains "
                             "trained embeddings for each year, e. g.: 2000.bin.gz",
                        default='embeddings')
    parser.add_argument('--first-year', type=int, dest='first_year',
                        help="The classifier will load embeddings for years from "
                             "--first-year to --last-year.", default=2000)
    parser.add_argument('--last-year', type=int, dest='last_year',
                        help="The classifier will load embeddings for years from "
                             "--first-year to --last-year.",
                        default=2020)
    parser.add_argument('--embeddings-extension', type=str, dest="embeddings_extension",
                        help='extension of the files with trained embeddings, e. g bin.gz',
                        default="bin.gz")
    parser.add_argument('--dataset', type=str,
                        help="Path of or link to the dataset",
                        default="https://raw.githubusercontent.com/wadimiusz/diachrony_for_russian/master/datasets/micro.csv")
    parser.add_argument('--save-to', type=str, dest='save_to', default="clf.pkl",
                        help="Name of file where the resulting classifier will be saved")
    args = parser.parse_args()

    models = dict()
    for year in range(args.first_year, args.last_year+1):
        models[year] = \
            load_model(os.path.join(args.embeddings_dir,
                                    f"{year}.{args.embeddings_extension}"))

    df = pd.read_csv(args.dataset)
    X_train = ((row.WORD, models[row.BASE_YEAR], models[row.BASE_YEAR+1]) for row in df.itertuples())
    y_train = df.GROUND_TRUTH

    clf = ShiftClassifier().fit(X_train, y_train)
    joblib.dump(clf, args.save_to)


if __name__ == "__main__":
    main()