# python3
# coding: utf-8

import argparse
import logging
import gensim
from algos import smart_procrustes_align_gensim
from utils import load_model

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--emb0", "-e0", help="Base model", required=True)
    arg("--emb1", "-e1", help="Model to align with the base one", required=True)

    args = parser.parse_args()

    models = []

    for mfile in [args.emb0, args.emb1]:
        model = load_model(mfile)
        model.init_sims(replace=True)
        models.append(model)

    logger.info("Aligning models...")
    models[1] = smart_procrustes_align_gensim(models[0], models[1])
    logger.info("Alignment complete")

    name = args.emb1.split(".")[0]
    new_name = name + "_aligned"
    models[1].save_word2vec_format(new_name + ".bin.gz", binary=True)
