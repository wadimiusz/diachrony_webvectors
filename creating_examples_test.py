from creating_examples_oop import GetExamples
from tqdm import tqdm
import gensim
import random

models = {}
for year in tqdm(range(2015, 2020)):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'models/{year}_0_5.bin'.format(year=year), binary=True, unicode_errors='replace')
    model.init_sims(replace=True)

    models.update({year: model})

vocabs = [model.vocab for model in list(models.values())]
intersected_vocab = set.intersection(*map(set, vocabs))

word = random.sample(list(intersected_vocab), 1)[0]

GetExamples(word).create_examples()
