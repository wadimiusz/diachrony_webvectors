import pandas as pd
import numpy as np
from utils import log, format_time, intersection_align_gensim
import random
import sys
import time
import os
import gensim
import logging
from scipy import spatial
from tqdm import tqdm
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder


class GetExamplesElmo:
    def __init__(self):
        self.elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz")

    def create_examples(self, word, pickle, years):
        old_contexts = list()
        new_contexts = list()

        base_years = list()
        new_years = list()

        start = time.time()
        log("Finding samples...")

        try:
            old_samples = pickle.get(years[0])
            new_samples = pickle.get(years[1])

        except KeyError:
            raise KeyError("Problem with", word, "because not enough samples found")

        old_samples_vec = np.zeros((len(old_samples), 1024), dtype='float32')
        new_samples_vec = np.zeros((len(new_samples), 1024), dtype='float32')

        for nr, old_sample in enumerate(old_samples):
            old_sample_vec = self.elmo([old_sample[0]])
            if old_sample_vec is not None:
                old_samples_vec[nr, :] = old_sample_vec

        for nr, new_sample in enumerate(new_samples):
            new_sample_vec = self.elmo([new_sample[0]])
            if new_sample_vec is not None:
                new_samples_vec[nr, :] = new_sample_vec

        distances = spatial.distance.cdist(old_samples_vec, new_samples_vec, 'cosine')

        most_distant_ids = np.unravel_index(np.argsort(distances, axis=None), distances.shape)

        old_samples_ids = list()
        new_samples_ids = list()

        for i in range(0, len(most_distant_ids[0])):
            old = most_distant_ids[0][i]
            new = most_distant_ids[1][i]
            if (old not in old_samples_ids) and (new not in new_samples_ids):
                old_samples_ids.append(old)
                new_samples_ids.append(new)
                if len(new_samples_ids) == 5:
                     break

        five_old_samples = [old_samples[i][1] for i in list(old_samples_ids)]
        five_new_samples = [new_samples[i][1] for i in list(new_samples_ids)]

        old_contexts.append(five_old_samples)
        new_contexts.append(five_new_samples)

        base_years.append(years[0])
        new_years.append(years[1])

        log("")
        log("This took ", format_time(time.time() - start))
        log("")
        output_df = pd.DataFrame({"WORD": word, "BASE_YEAR": base_years,
                                  "OLD_CONTEXTS": old_contexts, "NEW_YEAR": new_years,
                                  "NEW_CONTEXTS": new_contexts})
        output_df.index.names = ["ID"]
        output_df.to_csv('contexts_by_year.csv')
        log('Contexts saved to contexts_by_year.csv')
