# python3
# coding: utf-8

import gensim
import logging
import argparse
from smart_open import open
import numpy as np
from os import path
from algos import smart_procrustes_align_gensim

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--emb0', '-e0', help='Base model', required=True)
    arg('--emb1', '-e1', help='Model to align with the base one', required=True)

    args = parser.parse_args()

    models = []

    for mfile in [args.emb0, args.emb1]:
        model = gensim.models.KeyedVectors.load(mfile)
        model.init_sims(replace=True)
        models.append(model)

    logger.info('Aligning models...')
    models[1] = smart_procrustes_align_gensim(models[0].wv, models[1].wv)
    logger.info('Alignment complete')

    models[1].save(args.emb1.replace('.model', '_aligned.model'))
