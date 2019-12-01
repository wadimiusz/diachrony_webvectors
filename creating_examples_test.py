from creating_examples import GetExamples
import gensim
import logging
from os import path
from argparse import ArgumentParser
import pickle

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--word', '-w', required=True, help='query word')
    parser.add_argument('--models', '-m', required=True, help='path to the word embeddings models')
    parser.add_argument('--pickle', '-p', required=True, help='path to the pickle')
    parser.add_argument('--year0', '-y', required=True, type=int,
                        help='First year of the comparison')

    args = parser.parse_args()

    year0 = args.year0
    year1 = year0 + 1

    if 2018 < year0 < 2015:
        print('Year out of bounds! Exiting...')
        exit()

    years = [year0, year1]

    models = {}
    for year in years:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            path.join(args.models, '{year}_0_5.bin'.format(year=year)),
            binary=True, unicode_errors='replace')
        model.init_sims(replace=True)

        models.update({year: model})

    word = args.word

    with open(path.join(args.pickle, '{word}.pickle'.format(word=word)), 'rb') as f:
        pickle_data = pickle.load(f)

    GetExamples(word, pickle_data, years).create_examples(models)
