from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score


def remove_ignore_labels(preds, label_ids, ignored_labels):
    final_preds = []
    final_label_ids = []
    for pred_t, label_id_t in zip(preds, label_ids):
        if not (pred_t == label_id_t and pred_t in ignored_labels):
            final_preds.append(pred_t)
            final_label_ids.append(label_id_t)

    return final_preds, final_label_ids


class Metrics(object):
    @staticmethod
    def compute(preds, golds):
        raise NotImplementedError()


class AccAndF1Metrics(Metrics):

    @staticmethod
    def compute(preds, golds):
        acc = AccAndF1Metrics.simple_accuracy(preds, golds)
        f1 = f1_score(y_true=golds, y_pred=preds, average='micro')
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    @staticmethod
    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


class PearsonAndSpearman(Metrics):
    @staticmethod
    def compute(preds, golds):
        pearson_corr = pearsonr(preds, golds)[0]
        spearman_corr = spearmanr(preds, golds)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
