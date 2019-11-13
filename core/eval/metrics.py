import numpy as np
from beautifultable import BeautifulTable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from core.common import PRINT


def remove_ignore_labels(preds, label_ids, ignored_labels):
    final_preds = []
    final_label_ids = []
    for pred_t, label_id_t in zip(preds, label_ids):
        if not (pred_t == label_id_t and pred_t in ignored_labels):
            final_preds.append(pred_t)
            final_label_ids.append(label_id_t)

    return final_preds, final_label_ids


class FScorer:
    def __init__(self, beta=1.0):
        self._beta = beta
        self.clear()

    @property
    def precision(self):
        if self._true_positive + self._false_positive:
            return self._true_positive / (self._true_positive + self._false_positive)
        return 0.0

    @property
    def recall(self):
        if self._true_positive + self._false_negative:
            return self._true_positive / (self._true_positive + self._false_negative)
        return 0.0

    @property
    def fscore(self):
        numerator = (1 + self._beta ** 2) * self.precision * self.recall
        denominator = self._beta ** 2 * self.precision + self.recall
        if denominator:
            return numerator / denominator
        return 0.0

    def clear(self):
        self._true_positive = 0
        self._true_negative = 0
        self._false_positive = 0
        self._false_negative = 0

    def __call__(self, predicts, golds):
        raise NotImplementedError()


class SpanClassificationFscore(FScorer):
    def __init__(self, labels=[], beta=1.0):
        super().__init__(beta)
        self._labels = labels

    def __call__(self, predicts, golds):
        predicted_mentions = set(predicts)
        gold_mentions = set(golds)

        padded_predicted_mentions, padded_gold_mentions = [], []

        mentions = predicted_mentions | gold_mentions

        for mention in mentions:
            *_, mention_label = mention

            padded_predicted_mentions.append(
                mention in predicted_mentions and mention_label in self._labels
            )
            padded_gold_mentions.append(
                mention in gold_mentions and mention_label in self._labels
            )

        matcher = list(zip(padded_predicted_mentions, padded_gold_mentions))

        self._true_positive += matcher.count((True, True))
        self._true_negative += matcher.count((False, False))
        self._false_positive += matcher.count((True, False))
        self._false_negative += matcher.count((False, True))


class Metrics(object):

    def compute(self, preds, golds):
        raise NotImplementedError()


class AccAndF1Metrics(Metrics):

    def compute(self, preds, golds):
        _preds = None
        _golds = None
        for batch_predicts, batch_golds in zip(preds, golds):
            if _preds is None:
                _preds = batch_predicts
                _golds = batch_golds
            else:
                _preds = np.append(batch_predicts, preds, axis=0)
                _golds = np.append(batch_golds, golds, axis=0)

        acc = AccAndF1Metrics.simple_accuracy(_preds, _golds)
        f1 = f1_score(y_true=_golds, y_pred=_preds, average='micro')
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    @staticmethod
    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


class PearsonAndSpearman(Metrics):

    def _to_table(self, pearson_corr, spearman_corr):
        table = BeautifulTable()

        table.column_headers = [
            "Pearson",
            "Spearmanr",
            "Corr",
        ]

        table.column_alignments["Pearson"] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments["Spearmanr"] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments["Corr"] = BeautifulTable.ALIGN_RIGHT

        table.set_style(BeautifulTable.STYLE_COMPACT)
        table.append_row((pearson_corr, spearman_corr, (pearson_corr + spearman_corr) / 2))

        return table

    def compute(self, preds, golds):
        _preds = None
        _golds = None
        for batch_predicts, batch_golds in zip(preds, golds):
            if _preds is None:
                _preds = batch_predicts
                _golds = batch_golds
            else:
                _preds = np.append(batch_predicts, preds, axis=0)
                _golds = np.append(batch_golds, golds, axis=0)

        pearson_corr = pearsonr(_preds, _golds)[0]
        spearman_corr = spearmanr(_preds, _golds)[0]
        to_print = '\n' + str(self._to_table(pearson_corr, spearman_corr))
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
            PRINT: to_print
        }


class SpanClassificationMetrics(Metrics):
    def __init__(self, labels=[], beta=1.0):
        self._labels = labels
        self.single_scores = {
            label: SpanClassificationFscore(labels=[label], beta=beta) for label in labels
        }
        self.overall_score = SpanClassificationFscore(labels=labels, beta=beta)

    def compute(self, preds, golds):
        for batch_predicts, batch_golds in zip(preds, golds):
            for sentence_index in range(len(batch_golds)):
                for label in self._labels:
                    self.single_scores[label](
                        batch_predicts[sentence_index], batch_golds[sentence_index]
                    )

                self.overall_score(
                    batch_predicts[sentence_index], batch_golds[sentence_index]
                )

        return {
            "precision": self.overall_score.precision,
            "recall": self.overall_score.recall,
            "f1": self.overall_score.fscore
        }
