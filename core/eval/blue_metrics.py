import numpy as np

from libs.blue.ext import pmetrics
from .metrics import Metrics


class ChemprotMetrics(Metrics):
    def __init__(self, labels, average='micro', acc_only=False, ):
        self.average = average
        self.acc_only = acc_only
        self.labels = labels

    def compute(self, preds, golds):
        _preds = None
        _golds = None
        for batch_predicts, batch_golds in zip(preds, golds):
            if _preds is None:
                _preds = batch_predicts
                _golds = batch_golds
            else:
                _preds = np.append(batch_predicts, _preds, axis=0)
                _golds = np.append(batch_golds, _golds, axis=0)

        _preds = []
        _golds = []
        for batch_predicts, batch_golds in zip(preds, golds):
            _preds.extend(batch_predicts.tolist())
            _golds.extend(batch_golds.tolist())

        result = pmetrics.classification_report(_golds, _preds, macro=False,
                                                micro=True, classes_=self.labels)
        subindex = [i for i in range(len(self.labels)) if self.labels[i] != 'false']
        result = result.sub_report(subindex, macro=False, micro=True)
        return {'f1': result.overall_acc.tolist()}
