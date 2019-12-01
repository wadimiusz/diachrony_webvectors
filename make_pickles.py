import gensim
import pickle
import pandas as pd

corpuses = {}
for year in range(2015, 2020):
    df = pd.read_csv('corpora/tables/{year}_contexts.csv.gz'.format(year=year),
                     index_col='ID')
    corpuses.update({year: df})

models = {}
for year in range(2015, 2020):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'models/{year}_0_5.bin'.format(year=year),
        binary=True, unicode_errors='replace')
    model.init_sims(replace=True)

    models.update({year: model})

vocabs = [model.vocab for model in list(models.values())]
united_vocab = set.union(*map(set, vocabs))
print(len(united_vocab))

for word in united_vocab:
    dict = {}
    for year, model in models.items():
        if word in model.vocab:
            corpus = corpuses.get(year)
            samples = []
            for idx, lemmas, raw in corpus[['LEMMAS', 'RAW']].itertuples():
                lemmas_split = lemmas.split()
                if word in lemmas.split():
                    samples.append([lemmas.split(), raw])
            dict.update({year: samples})
    with open('pickles/{word}.pickle'.format(word=word), 'wb') as f:
        pickle.dump(dict, f)
