"""A small stdlib logistic regression and the collapse experiment.

The collapse experiment prices the leak in the currency benchmarks
are reported in: train one model on every feature and one on the
probe-cleaned feature set, score both on a held-out fleet, and report
how much of the headline AUC evaporates when the leaking families are
removed. The difference is performance the dataset never really had -
it was the maintenance crew's suspicion, memorised.
"""

import math

from .models import ALL_FEATURES, FAMILIES, auc


class LogisticRegression(object):
    """Deterministic full-batch logistic regression (stdlib only)."""

    def __init__(self, features, iterations=400, lr=0.5, l2=1e-4):
        self.features = list(features)
        self.iterations = iterations
        self.lr = lr
        self.l2 = l2
        self.mu = {}
        self.scale = {}
        self.weights = {}
        self.bias = 0.0

    def fit(self, cases):
        for feature in self.features:
            values = [case.features[feature] for case in cases]
            mu = sum(values) / len(values)
            var = sum((v - mu) * (v - mu) for v in values) / len(values)
            self.mu[feature] = mu
            self.scale[feature] = var ** 0.5 if var > 0 else 1.0
            self.weights[feature] = 0.0
        self.bias = 0.0
        # Precompute the standardised design matrix once; the gradient
        # loop then runs on plain lists.
        matrix = []
        labels = []
        for case in cases:
            matrix.append([self._z(case, feature) for feature in self.features])
            labels.append(case.label)
        w = [0.0] * len(self.features)
        bias = 0.0
        n = float(len(cases))
        for _it in range(self.iterations):
            grad = [0.0] * len(self.features)
            grad_bias = 0.0
            for row, y in zip(matrix, labels):
                z = bias
                for j in range(len(w)):
                    z += w[j] * row[j]
                if z > 30:
                    p = 1.0
                elif z < -30:
                    p = 0.0
                else:
                    p = 1.0 / (1.0 + math.exp(-z))
                err = p - y
                grad_bias += err
                for j in range(len(w)):
                    grad[j] += err * row[j]
            for j in range(len(w)):
                w[j] -= self.lr * (grad[j] / n + self.l2 * w[j])
            bias -= self.lr * grad_bias / n
        for j, feature in enumerate(self.features):
            self.weights[feature] = w[j]
        self.bias = bias
        return self

    def _z(self, case, feature):
        return (case.features[feature] - self.mu[feature]) / self.scale[feature]

    def _raw_predict(self, case):
        z = self.bias
        for feature in self.features:
            z += self.weights[feature] * self._z(case, feature)
        if z > 30:
            return 1.0
        if z < -30:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, case):
        return self._raw_predict(case)

    def score_auc(self, cases):
        scores = [self.predict(case) for case in cases]
        labels = [case.label for case in cases]
        return auc(scores, labels)


def features_of_families(families):
    out = []
    for family in families:
        out.extend(FAMILIES[family]["features"])
    return out


def collapse_experiment(train_cases, test_cases, flagged_families):
    """AUCs on the held-out fleet for four feature sets.

    - all: every feature (the headline number a benchmark would print)
    - cleaned: all features minus the probe-flagged families
    - sensors_only: the three sensor families
    - leak_only: the three maintenance-log families
    Plus the collapse itself: all minus cleaned.
    """
    from .models import CLEAN_FAMILIES, LEAK_FAMILIES

    flagged_features = set(features_of_families(flagged_families))
    cleaned = [f for f in ALL_FEATURES if f not in flagged_features]
    sets = {
        "all": list(ALL_FEATURES),
        "cleaned": cleaned,
        "sensors_only": features_of_families(CLEAN_FAMILIES),
        "leak_only": features_of_families(LEAK_FAMILIES),
    }
    out = {}
    for name, features in sets.items():
        if not features:
            out[name] = float("nan")
            continue
        model = LogisticRegression(features).fit(train_cases)
        out[name] = model.score_auc(test_cases)
    out["collapse"] = out["all"] - out["cleaned"]
    return out
