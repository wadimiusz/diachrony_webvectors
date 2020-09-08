import functools
from typing import Iterable, Tuple
from gensim.models import KeyedVectors
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from algos import GlobalAnchors, Jaccard, KendallTau


@functools.lru_cache(maxsize=-1)
def get_algo_by_kind_and_two_models(kind: str, model1: KeyedVectors,
                                    model2: KeyedVectors):
    if kind.lower() == "global_anchors":
        return GlobalAnchors(model1, model2)
    elif kind.lower() == "jaccard":
        return Jaccard(model1, model2, top_n_neighbors=50)
    elif kind.lower() == "kendall_tau":
        return KendallTau(model1, model2, top_n_neighbors=50)


class ShiftClassifier:
    def __init__(self):
        self.clf = LogisticRegression(class_weight='balanced')
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, x: Iterable[Tuple[str, KeyedVectors, KeyedVectors]],
            y: Iterable[float]):
        x_processed, y_processed = list(), list()
        for (word, model1, model2), label in zip(x, y):
            try:
                features = self.feature_extract(word, model1, model2)
                x_processed.append(features)
                y_processed.append(label)
            except KeyError:
                pass

        x_processed = self.scaler.fit_transform(x_processed)
        self.clf.fit(X=x_processed, y=y_processed)
        print("Classes:", self.clf.classes_)
        print("Coefficients:", self.clf.coef_)
        print("Intercept:", self.clf.intercept_)
        print("Iterations:", self.clf.n_iter_)
        self.fitted = True
        return self

    def predict(self, x: Iterable[Tuple[str, KeyedVectors, KeyedVectors]]):
        if not self.fitted:
            raise NotFittedError

        x_processed = list()
        for word, model1, model2 in x:
            features = self.feature_extract(word, model1, model2)
            x_processed.append(features)
        x_processed = self.scaler.transform(x_processed)
        return self.clf.predict(x_processed)

    def predict_proba(self, x: Iterable[Tuple[str, KeyedVectors, KeyedVectors]]):
        if not self.fitted:
            raise NotFittedError

        x_processed = list()
        for word, model1, model2 in x:
            features = self.feature_extract(word, model1, model2)
            x_processed.append(features)
        x_processed = self.scaler.transform(x_processed)
        return self.clf.predict_proba(x_processed)[:, 1]

    @staticmethod
    def feature_extract(word, model1, model2):
        if word not in model1.vocab:
            raise KeyError("Word {} is not in the "
                           "vocab of model1".format(word))

        if word not in model2.vocab:
            raise KeyError("Word {} is not in the "
                           "vocab of model1".format(word))

        procrustes_score = model1[word] @ model2[
            word]  # models have previously been aligned with Procrustes analysis

        global_anchors_score = \
            get_algo_by_kind_and_two_models(
                "global_anchors", model1, model2).get_score(word)

        jaccard_score = get_algo_by_kind_and_two_models("jaccard", model1, model2).get_score(word)

        kendall_tau_score = get_algo_by_kind_and_two_models("kendall_tau", model1,
                                                            model2).get_score(word)
        features = [procrustes_score, global_anchors_score, jaccard_score,
                    kendall_tau_score]
        return features
