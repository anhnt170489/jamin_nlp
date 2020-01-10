from libs.blue.ext import pmetrics
from .metrics import Metrics


class AccAndF1Metrics(Metrics):
    def __init__(self, labels, ignore_labels=None, macro=False, micro=True, acc_only=False, ):
        self.acc_only = acc_only
        self.labels = labels
        self.ignore_labels = ignore_labels
        self.macro = macro
        self.micro = micro

    def compute(self, preds, golds):

        _preds = []
        _golds = []
        for batch_predicts, batch_golds in zip(preds, golds):
            _preds.extend(batch_predicts.tolist())
            _golds.extend(batch_golds.tolist())

        result = pmetrics.classification_report(_golds, _preds, macro=self.macro,
                                                micro=self.micro, classes_=self.labels)
        if self.ignore_labels:
            subindex = [i for i in range(len(self.labels)) if self.labels[i] not in self.ignore_labels]
            result = result.sub_report(subindex, macro=self.macro, micro=self.micro)
        # return {'f1': result.overall_acc.tolist()}
        return {'precision': float(result.table['Precision'].tolist()[-1]),
                'recall': float(result.table['Recall'].tolist()[-1]),
                'f1': float(result.table['F-score'].tolist()[-1]),
                'acc': float(result.table['Accuracy'].tolist()[-1])}
