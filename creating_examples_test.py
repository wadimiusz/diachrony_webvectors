from creating_examples import GetExamples
import gensim
import logging
import random
from os import path
from argparse import ArgumentParser

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--word', '-w', required=True, help='query word')
    parser.add_argument('--models', '-m', required=True, help='path to the word embeddings models')
    parser.add_argument('--corpora', '-c', required=True, help='path to the corpora')
    args = parser.parse_args()

    models = {}
    for year in range(2015, 2020):
        model = gensim.models.KeyedVectors.load_word2vec_format(
            path.join(args.models, '{year}_0_5.bin'.format(year=year)),
            binary=True, unicode_errors='replace')
        model.init_sims(replace=True)

        models.update({year: model})

    vocabs = [model.vocab for model in list(models.values())]
    intersected_vocab = set.intersection(*map(set, vocabs))

    word = args.word
    corpora = args.corpora
    # word = random.sample(list(intersected_vocab), 1)[0]

    GetExamples(word, corpora).create_examples(models, intersected_vocab)
