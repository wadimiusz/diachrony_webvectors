import sys
import gensim
import logging
import pickle
import pandas as pd
from argparse import ArgumentParser
from os import path
from smart_open import open
from functools import partial
from multiprocessing import Pool


def generate_example(query, vocabularies=None):
    print('Saving word:', query, file=sys.stderr)
    out_dict = {}
    for cur_year in vocabularies:
        if query in vocabularies[cur_year]:
            corpus = corpuses.get(cur_year)
            samples = []
            for idx, lemmas, raw in corpus[['LEMMAS', 'RAW']].itertuples():
                lemmas_split = lemmas.split()
                if query in lemmas_split:
                    samples.append([lemmas_split, raw])
            out_dict.update({cur_year: samples})
    with open('pickles/{word}.pickle.gz'.format(word=query), 'wb') as f:
        pickle.dump(out_dict, f)
    return out_dict


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = ArgumentParser()
parser.add_argument('--models', '-m', required=True, help='path to the word embeddings models')
parser.add_argument('--corpora', '-c', required=True, help='path to the corpora')
parser.add_argument('--mincount', '-f', type=int, help='Minimal word count to consider',
                    default=1000)
parser.add_argument('--parallel', '-p', type=int, help='Number of parallel processes',
                    default=10)
args = parser.parse_args()

vocabs = {}
frequencies = {}
for year in range(2010, 2020):
    model = gensim.models.KeyedVectors.load(
        path.join(args.models, '{year}_rnc_incremental.model'.format(year=year)))
    vocabs[year] = set([w for w in model.index2word if w.endswith('_NOUN')
                        or w.endswith('_PROPN') or w.endswith('_ADJ')])
    for word in vocabs[year]:
        if word not in frequencies:
            frequencies[word] = 0
        frequencies[word] += model.vocab[word].count

united_vocab = set.union(*map(set, vocabs.values()))
united_vocab = set([w for w in united_vocab if frequencies[w] > args.mincount])
print('Total words:', len(united_vocab), file=sys.stderr)

corpuses = {}
for year in range(2010, 2020):
    print('Loading corpus:', year, file=sys.stderr)
    df = pd.read_csv(
        path.join(args.corpora, '{year}_contexts.csv.gz'.format(year=year)),
        index_col='ID')
    corpuses.update({year: df})

with Pool(args.parallel) as p:
    func = partial(generate_example, vocabularies=vocabs)
    for i in p.imap_unordered(func, united_vocab, chunksize=100):
        pass
