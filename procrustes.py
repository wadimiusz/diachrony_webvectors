import gensim
import numpy as np
import sys

def intersection_align_gensim(m1: gensim.models.KeyedVectors, m2: gensim.models.KeyedVectors,
                              pos_tag: (str, None) = None, words: (list, None) = None):
    """
    This procedure, taken from https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf and slightly
    modified, corrects two models in a way that only the shared words of the vocabulary are kept in the model,
    and both vocabularies are sorted by frequencies.
    Original comment is as follows:

    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new vectors and vectors_norm objects in both gensim models:
        -- so that Row 0 of m1.vectors will be for the same word as Row 0 of m2.vectors
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.

    :param m1: the first model
    :param m2: the second model
    :param pos_tag: if given, we remove words with other pos tags
    :param words: a container
    :return m1, m2: both models after their vocabs are modified
    """

    # Get the vocab for each model
    if pos_tag is None:
        vocab_m1 = set(m1.vocab.keys())
        vocab_m2 = set(m2.vocab.keys())
    else:
        vocab_m1 = set(word for word in m1.vocab.keys() if word.endswith("_" + pos_tag))
        vocab_m2 = set(word for word in m2.vocab.keys() if word.endswith("_" + pos_tag))

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words:
        common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
        return m1, m2

    # Otherwise sort lexicographically
    common_vocab = list(common_vocab)
    common_vocab.sort()

    # Then for each model...
    for m in (m1, m2):
        # Replace old vectors_norm array with new one (with common vocab)
        indices = [m.vocab[w].index for w in common_vocab]
        old_arr = m.vectors_norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.vectors_norm = m.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.vocab
        new_vocab = dict()
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.vocab = new_vocab

    return m1, m2

def smart_procrustes_align_gensim(base_embed: gensim.models.KeyedVectors,
                                  other_embed: gensim.models.KeyedVectors):
    """
    This code, taken from
    https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf and modified,
    uses procrustes analysis to make two word embeddings compatible.
    :param base_embed: first embedding
    :param other_embed: second embedding to be changed
    :return other_embed: changed embedding
    """
    base_embed.init_sims()
    other_embed.init_sims()

    base_vecs = base_embed.syn0norm
    other_vecs = other_embed.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.syn0norm = other_embed.syn0 = other_embed.syn0norm.dot(ortho)

    return other_embed


class ProcrustesAligner(object):
    def __init__(self, w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors):
        self.w2v1, self.w2v2 = intersection_align_gensim(w2v1, w2v2)
        self.w2v2_changed = smart_procrustes_align_gensim(w2v1, w2v2)

    def __repr__(self):
        return "ProcrustesAligner"

    def get_score(self, word):
        vector1 = self.w2v1.wv[word]
        vector2 = self.w2v2_changed.wv[word]
        score = np.dot(vector1, vector2)  # More straightforward computation
        return score

    def get_changes(self, top_n_changed_words: int, pos: str = None):
        print('Doing procrustes', file=sys.stderr)
        result = list()
        # their vocabs should be the same, so it doesn't matter over which to iterate:
        for word in self.w2v1.wv.vocab.keys():
            if pos is None or pos == "ALL" or word.endswith(pos):
                try:  # надо написать нормальный способ это обходить, исправив баг с рефами в intersection_align_gensim, но в любом случае выкидываются только редкие слов атут
                    score = self.get_score(word)
                except KeyError:
                    pass
                result.append((word, score))

        result = sorted(result, key=lambda x: x[1])
        result = result[:top_n_changed_words]
        print('Done', file=sys.stderr)
        return result


if __name__ == "__main__":
    pass
