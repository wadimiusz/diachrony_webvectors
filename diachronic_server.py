#!/usr/bin/env python3
# coding: utf-8

import configparser
import csv
import datetime
import json
import logging
import socket
import sys
import threading
from functools import lru_cache
import numpy as np
import gensim
import joblib
import pickle
from algos import ProcrustesAligner
from creating_examples import GetExamples
from itertools import combinations
from os import path
from smart_open import open


class WebVectorsThread(threading.Thread):
    def __init__(self, connect, address):
        threading.Thread.__init__(self)
        self.connect = connect
        self.address = address

    def run(self):
        clientthread(self.connect, self.address)


def clientthread(connect, addres):
    # Sending message 'operation': '1'to connected client
    connect.send(bytes(b'word2vec model server'))

    # infinite loop so that function do not terminate and thread do not end.
    while True:
        # Receiving from client
        data = connect.recv(1024)
        if not data:
            break
        query = json.loads(data.decode('utf-8'))
        output = operations[query['operation']](query)
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M"), '\t', addres[0] + ':' + str(addres[1]), '\t',
              data.decode('utf-8'), file=sys.stderr)
        reply = json.dumps(output, ensure_ascii=False)
        connect.sendall(reply.encode('utf-8'))
        break

    # came out of loop
    connect.close()


config = configparser.RawConfigParser()
config.read('webvectors.cfg')

root = config.get('Files and directories', 'root')
HOST = config.get('Sockets', 'host')
PORT = config.getint('Sockets', 'port')
tags = config.getboolean('Tags', 'use_tags')
PICKLES = config.get('Files and directories', 'pickles')
CORPORA = config.get('Files and directories', 'corpora')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Loading models

our_models = {}
with open(root + config.get('Files and directories', 'models'), 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        our_models[row['identifier']] = {}
        our_models[row['identifier']]['path'] = row['path']
        our_models[row['identifier']]['default'] = row['default']
        our_models[row['identifier']]['tags'] = row['tags']
        our_models[row['identifier']]['algo'] = row['algo']
        our_models[row['identifier']]['corpus_size'] = int(row['size'])

models_dic = {}

for m in our_models:
    modelfile = our_models[m]['path']
    our_models[m]['vocabulary'] = True
    if our_models[m]['algo'] == 'fasttext':
        models_dic[m] = gensim.models.KeyedVectors.load(modelfile)
    else:
        if modelfile.endswith('.bin.gz'):
            models_dic[m] = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=True)
            our_models[m]['vocabulary'] = False
        elif modelfile.endswith('.vec.gz'):
            models_dic[m] = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)
            our_models[m]['vocabulary'] = False
        else:
            models_dic[m] = gensim.models.KeyedVectors.load(modelfile)
    models_dic[m].init_sims(replace=True)
    print("Model", m, "from file", modelfile, "loaded successfully.", file=sys.stderr)

shift_classifier = joblib.load("clf.pkl")


# Vector functions

def find_variants(word, usermodel):
    # Find variants of query word in the model
    model = models_dic[usermodel]
    results = None
    candidates_set = set()
    candidates_set.add(word.upper())
    if tags and our_models[usermodel]['tags'] == 'True':
        candidates_set.add(word.split('_')[0] + '_X')
        candidates_set.add(word.split('_')[0].lower() + '_' + word.split('_')[1])
        candidates_set.add(word.split('_')[0].capitalize() + '_' + word.split('_')[1])
    else:
        candidates_set.add(word.lower())
        candidates_set.add(word.capitalize())
    for candidate in candidates_set:
        if candidate in model.wv.vocab:
            results = candidate
            break
    return results


def frequency(word, model):
    corpus_size = our_models[model]['corpus_size']
    if word not in models_dic[model].vocab:
        return 0, 'low'
    if not our_models[model]['vocabulary']:
        return 0, 'mid'
    wordfreq = models_dic[model].vocab[word].count
    relative = wordfreq / corpus_size
    tier = 'mid'
    if relative > 0.0001:
        tier = 'high'
    elif relative < 0.00005:
        tier = 'low'
    return wordfreq, tier


def find_synonyms(query):
    q = query['query']
    pos = query['pos']
    usermodel = query['model']
    nr_neighbors = query['nr_neighbors']
    results = {'frequencies': {}}
    qf = q
    model = models_dic[usermodel]
    if qf not in model.vocab:
        qf = find_variants(qf, usermodel)
        if not qf:
            if our_models[usermodel]['algo'] == 'fasttext' and model.wv.__contains__(q):
                results['inferred'] = True
                qf = q
            else:
                results[q + " is unknown to the model"] = True
                results['frequencies'][q] = frequency(q, usermodel)
                return results
    results['frequencies'][q] = frequency(qf, usermodel)
    results['neighbors'] = []
    if pos == 'ALL':
        for i in model.most_similar(positive=qf, topn=nr_neighbors):
            results['neighbors'].append(i)
    else:
        counter = 0
        for i in model.most_similar(positive=qf, topn=1000):
            if counter == nr_neighbors:
                break
            if i[0].split('_')[-1] == pos:
                results['neighbors'].append(i)
                counter += 1
    if len(results) == 0:
        results['No results'] = True
        return results
    for res in results['neighbors']:
        freq, tier = frequency(res[0], usermodel)
        results['frequencies'][res[0]] = (freq, tier)
    raw_vector = model[qf]
    results['vector'] = raw_vector.tolist()
    return results


def find_similarity(query):
    q = query['query']
    usermodel = query['model']
    model = models_dic[usermodel]
    results = {'similarities': [], 'frequencies': {}}
    for pair in q:
        (q1, q2) = pair
        qf1 = q1
        qf2 = q2
        if q1 not in model.wv.vocab:
            qf1 = find_variants(qf1, usermodel)
            if not qf1:
                if our_models[usermodel]['algo'] == 'fasttext' and model.wv.__contains__(q1):
                    results['inferred'] = True
                    qf1 = q1
                else:
                    results["Unknown to the model"] = q1
                    return results
        if q2 not in model.wv.vocab:
            qf2 = find_variants(qf2, usermodel)
            if not qf2:
                if our_models[usermodel]['algo'] == 'fasttext' and model.wv.__contains__(q2):
                    results['inferred'] = True
                    qf2 = q2
                else:
                    results["Unknown to the model"] = q2
                    return results
        results['frequencies'][qf1] = frequency(qf1, usermodel)
        results['frequencies'][qf2] = frequency(qf2, usermodel)
        pair2 = (qf1, qf2)
        result = float(model.similarity(qf1, qf2))
        results['similarities'].append((pair2, result))
    return results


def vector(query):
    q = query['query']
    usermodel = query['model']
    results = {}
    qf = q
    results['frequencies'] = {}
    results['frequencies'][q] = frequency(q, usermodel)
    model = models_dic[usermodel]
    if q not in model.wv.vocab:
        qf = find_variants(qf, usermodel)
        if not qf:
            if our_models[usermodel]['algo'] == 'fasttext' and model.wv.__contains__(q):
                results['inferred'] = True
                qf = q
            else:
                results[q + " is unknown to the model"] = True
                return results
    raw_vector = model[qf]
    raw_vector = raw_vector.tolist()
    results['vector'] = raw_vector
    return results


@lru_cache()
def get_global_anchors_model(model1_name, model2_name):
    model1 = models_dic[model1_name]
    model2 = models_dic[model2_name]
    return ProcrustesAligner(w2v1=model1, w2v2=model2)


def find_shifts(query):
    model1 = models_dic[query['model1']]
    model2 = models_dic[query['model2']]
    pos = query.get("pos")
    n = query['n']
    results = {'frequencies': {}}
    shared_voc = list(set.intersection(set(model1.index2word), set(model2.index2word)))
    matrix1 = np.zeros((len(shared_voc), model1.vector_size))
    matrix2 = np.zeros((len(shared_voc), model2.vector_size))
    for nr, word in enumerate(shared_voc):
        matrix1[nr, :] = model1[word]
        matrix2[nr, :] = model2[word]
    sims = (matrix1 * matrix2).sum(axis=1)
    min_sims = np.argsort(sims)  # [:n]
    results['neighbors'] = list()
    results['frequencies'] = dict()
    freq_type_num = {"low": 0, "mid": 0, "high": 0}
    for nr in min_sims:
        if min(freq_type_num.values()) > n:
            break

        word = shared_voc[nr]
        sim = sims[nr]

        freq = frequency(word, query['model1'])
        if word.endswith(pos) or pos == "ALL":
            if freq_type_num[freq[1]] < n:
                results['neighbors'].append((word, sim))
                results['frequencies'][word] = freq
                freq_type_num[freq[1]] += 1

    return results


def is_semantic_shift(word, model_names):
    for model_pair in combinations(model_names, 2):
        proba, label = semantic_shift_predict(word, model_pair[0],
                                              model_pair[1])
        if int(label) == 1:
            return True
    return False


def multiple_neighbors(query):
    """
    :target_word: str
    :model_year_list: a reversed list of years for the selected models
    :model_list: a list of selected models to be analyzed
    """
    target_word = query["query"]
    pos = query["pos"]

    if target_word.count("_") == 0:
        return {target_word + " is unknown to the model": True}

    model_year_list = sorted(query["model"], reverse=True)
    model_list = [models_dic[year] for year in model_year_list]

    word, target_word_pos = target_word.split('_')
    target_word = word.lower() + '_' + target_word_pos

    word_list = [" ".join([target_word.split("_")[0], year]) for year in model_year_list]
    actual_years = len(word_list)
    vector_list = [model[target_word].tolist() for model in model_list if target_word in model]

    # get word labels and vectors
    for year, model in enumerate(model_list):
        if target_word not in model:
            continue
        similar_words = model.most_similar(target_word, topn=1000)
        neighbours_counter = 0
        for similar_word in similar_words:
            if neighbours_counter > 4:
                continue
            similar_word = similar_word[0]
            freq, tier = frequency(similar_word, model_year_list[year])
            # filter words of low frequency
            if freq > 20:
                try:
                    (lemma, similar_word_pos) = similar_word.split("_")
                except ValueError:
                    lemma = similar_word
                    similar_word_pos = None
                if lemma not in word_list:
                    # filter words by pos-tag
                    if pos == "ALL" or (similar_word_pos and pos == similar_word_pos):
                        word_list.append(lemma)
                        neighbours_counter += 1
                        # get the most recent meaning
                        for recent_model in model_list:
                            if similar_word in recent_model:
                                vector_list.append(recent_model[similar_word].tolist())
                                break

    result = {
        "pos": pos,
        "word_list": word_list,
        "vector_list": vector_list,
        "model_number": actual_years
    }

    return result


@lru_cache(2048)
def semantic_shift_predict(word, model1_name, model2_name, threshold=0.6):
    model1 = models_dic[model1_name]
    model2 = models_dic[model2_name]
    proba = shift_classifier.predict_proba([(word, model1, model2)])[0]
    label = int(proba > threshold)
    return proba, label


def classify_semantic_shifts(query):
    with_examples = query['with_examples']
    word = query["word"]
    model1_name = query["model1"]
    model2_name = query["model2"]
    model1 = models_dic[model1_name]
    model2 = models_dic[model2_name]
    frequencies = {model1_name: frequency(word, model1_name),
                   model2_name: frequency(word, model2_name)}

    if word not in model1:
        return {word + " is unknown to the model": True}

    if word not in model2:
        return {word + " is unknown to the model": True}

    proba, label = semantic_shift_predict(word, model1_name, model2_name)

    years = [int(model1_name), int(model2_name)]
    models = {int(model1_name): model1, int(model2_name): model2}
    if with_examples == 'slow':
        corpora_csv_1 = open(path.join(
            root, CORPORA, '{year1}_contexts.csv.gz'.format(year1=years[0])), 'r')
        corpora_csv_2 = open(path.join(
            root, CORPORA, '{year2}_contexts.csv.gz'.format(year2=years[1])), 'r')
        examples = GetExamples(word, years).create_examples(models,
                                                            [corpora_csv_1, corpora_csv_2], 2)
    elif with_examples:
        try:
            pickle_file = open(path.join(
                root, PICKLES, '{word}.pickle.gz'.format(word=word)), 'rb')
            pickle_data = pickle.load(pickle_file)
            examples = GetExamples(word, years).create_examples(models, [pickle_data], 1)
        except FileNotFoundError:
            examples = None
    else:
        examples = None
    return {"proba": str(proba), "label": str(label), "examples": examples,
            "frequencies": frequencies}


operations = {
    '1': find_synonyms,
    '2': find_similarity,
    '4': vector,
    '5': find_shifts,
    '6': multiple_neighbors,
    '7': classify_semantic_shifts
}

# Bind socket to local host and port

if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created', file=sys.stderr)

    try:
        s.bind((HOST, PORT))
    except socket.error as msg:
        print('Bind failed. Error Code and message: ' + str(msg), file=sys.stderr)
        sys.exit()

    print('Socket bind complete', file=sys.stderr)

    # Start listening on socket
    s.listen(100)
    print('Socket now listening on port', PORT, file=sys.stderr)

    # now keep talking with the client
    while 1:
        # wait to accept a connection - blocking call
        conn, addr = s.accept()

        # start new thread takes 1st argument as a function name to be run,
        # 2nd is the tuple of arguments to the function.
        thread = WebVectorsThread(conn, addr)
        thread.start()
