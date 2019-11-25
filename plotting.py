#!/usr/bin/env python
# coding: utf-8

import sys
import matplotlib
matplotlib.use('Agg')
import re
import pylab as plot
import numpy as np
from matplotlib import font_manager
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import configparser


config = configparser.RawConfigParser()
config.read('webvectors.cfg')

root = config.get('Files and directories', 'root')
path = config.get('Files and directories', 'font')
font = font_manager.FontProperties(fname=path)


def tsne_semantic_shifts(array, word_labels, word_vectors):

    np.set_printoptions(suppress=True)
    embedding = TSNE(n_components=2, random_state=0, learning_rate=150, init="pca")
    y = embedding.fit_transform(array)
    word_coordinates = [y[i] for i in range(0, len(word_vectors) + 1)]
    x_coordinates, y_coordinates = y[:, 0], y[:, 1]

    plot.scatter(x_coordinates, y_coordinates)
    plot.axis('off')

    for label, x, y in zip(word_labels, x_coordinates, y_coordinates):
        plot.annotate(label, xy=(x, y), xytext=(-len(label)*4.5, 4), textcoords='offset points')
    plot.xlim(x_coordinates.min() - 10, x_coordinates.max() + 10)
    plot.ylim(y_coordinates.min() - 10, y_coordinates.max() + 10)
    for i in range(len(word_coordinates) - 1, 0, -1):
        plot.annotate("", xy=(word_coordinates[i - 1][0], word_coordinates[i - 1][1]),
                     xytext=(word_coordinates[i][0], word_coordinates[i][1] + 5),
                     arrowprops=dict(arrowstyle='-|>', color='indianred'))

    plot.savefig(root + 'data/images/tsne_semantic_shift/' + modelname + '_' + fname + '.png',
                 dpi=150, bbox_inches='tight')
    plot.close()
    plot.clf()


def vizualize_semantic_shifts(all_similar_words, word_vectors, model, word, year, kind="TSNE"):

    rows = len(word_vectors) + len(all_similar_words) + 1
    array = np.empty((rows, model.vector_size), dtype='f')
    word_labels = [word + ' ' + year]

    num = int(year) - 1

    array[0, :] = model[word]

    row_counter = 1

    for i in range(len(word_vectors)):
        array[row_counter, :] = word_vectors[i]
        row_counter += 1
        word_labels.append(word + ' ' + str(num))
        num -= 1

    for word_ in all_similar_words:
        word_vector = model[word_]
        word_labels.append(word_)
        array[row_counter, :] = word_vector
        row_counter += 1

    word_labels = [re.sub(r'_[A-Z]+', '', w) for w in word_labels]

    if kind.lower() == "tsne":
        return tsne_semantic_shifts(array, word_labels, word_vectors)
    else:
        raise ValueError("Kind is {}, must be TSNE".format(kind))


def singularplot(word, modelname, vector, fname):
    xlocations = np.array(list(range(len(vector))))
    plot.clf()
    plot.bar(xlocations, vector)
    plot_title = word.split('_')[0].replace('::', ' ') + '\n' + modelname + u' model'
    plot.title(plot_title, fontproperties=font)
    plot.xlabel('Vector components')
    plot.ylabel('Components values')
    plot.savefig(root + 'data/images/singleplots/' + modelname + '_' + fname + '.png', dpi=150,
                 bbox_inches='tight')
    plot.close()
    plot.clf()


def embed(words, matrix, classes, usermodel, fname, kind='TSNE'):
    perplexity = 6.0  # Should be smaller than the number of points!

    if kind.lower() == "tsne":
        embedding = TSNE(n_components=2, perplexity=perplexity, metric='cosine', n_iter=500,
                         init='pca')
    elif kind.lower() == "pca":
        embedding = PCA(n_components=2)
    else:
        raise ValueError("Kind is {}, must be TSNE or PCA".format(kind))

    y = embedding.fit_transform(matrix)

    print('2-d embedding finished', file=sys.stderr)

    class_set = [c for c in set(classes)]
    colors = plot.cm.rainbow(np.linspace(0, 1, len(class_set)))

    class2color = [colors[class_set.index(w)] for w in classes]

    xpositions = y[:, 0]
    ypositions = y[:, 1]
    seen = set()

    plot.clf()

    for color, word, class_label, x, y in zip(class2color, words, classes, xpositions, ypositions):
        plot.scatter(x, y, 20, marker='.', color=color,
                     label=class_label if class_label not in seen else "")
        seen.add(class_label)

        lemma = word.split('_')[0].replace('::', ' ')
        plot.annotate(lemma, xy=(x, y), xytext=(-len(lemma)*4.5, 0), textcoords="offset points",
                      size='x-large', weight='bold', fontproperties=font, color=color)

    plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plot.legend(loc='best')

    plot.savefig(root + 'data/images/' + kind.lower() + 'plots/' + usermodel + '_' + fname + '.png',
                 dpi=150, bbox_inches='tight')
    plot.close()
    plot.clf()
