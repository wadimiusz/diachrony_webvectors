from creating_examples import GetExamples
import gensim
import logging
from os import path
from argparse import ArgumentParser

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--word', '-w', required=True, help='query word')
    parser.add_argument('--models', '-m', required=True, help='path to the word embeddings models')
    parser.add_argument('--corpora', '-c', required=True, help='path to the corpora')
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
    corpora = args.corpora

    GetExamples(word, corpora, years).create_examples(models)
