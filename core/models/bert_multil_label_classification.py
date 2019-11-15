import torch
import torch.nn as nn

from core.common import *
from libs.transformers import BertConfig
from libs.transformers.modeling_bert import BertForMultiLabelClassification


class BertMultilabelClassification(nn.Module):
    def __init__(
            self, args
    ) -> None:
        super(BertMultilabelClassification, self).__init__()

        self._label_map = {i: label for i, label in enumerate(args.labels)}
        self._num_labels = len(args.labels)

        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            num_labels=self._num_labels)
        self._bert_for_multi_label_classification = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(
                '.ckpt' in args.model_name_or_path),
            config=config)
        self._mlb = args.mlb

    @staticmethod
    def flatten_labels(preds, label_ids):
        flat_preds = []
        flat_label_ids = []
        if label_ids is not None:
            for pred_t, label_id_t in zip(preds, label_ids):
                if len(pred_t) > 0 or len(label_id_t) > 0:
                    for pred in pred_t:
                        flat_preds.append(pred)
                        if pred in label_id_t:
                            flat_label_ids.append(pred)
                        else:
                            flat_label_ids.append(0)

                    for label_id in label_id_t:
                        if label_id not in pred_t:
                            flat_label_ids.append(label_id)
                            flat_preds.append(0)
                else:
                    flat_preds.append(0)
                    flat_label_ids.append(0)
        else:
            for pred_t in preds:
                if len(pred_t) > 0:
                    flat_preds.append(pred_t)
                else:
                    flat_preds.append([0])

        return flat_preds, flat_label_ids

    def forward(self, batch):
        tokens = batch['tokens']
        labels = batch['label'] if 'label' in batch else None

        outputs = self._bert_for_multi_label_classification(tokens[BERT_INPUT_IDS],
                                                            attention_mask=tokens[BERT_INPUT_MASKS], labels=labels)
        if labels is not None:
            loss, logits = outputs[:2]
        else:
            loss = None
            logits = outputs[0]

        preds = torch.sigmoid(logits) > 0.5
        preds = preds.detach().cpu().numpy()
        preds = self._mlb.inverse_transform(preds)

        if labels is not None:
            golds = labels.detach().cpu().numpy()
            golds = self._mlb.inverse_transform(golds)
        else:
            golds = None

        preds, golds = BertMultilabelClassification.flatten_labels(preds, golds)

        return {LOSS: loss, PREDICT: preds, GOLD: golds,
                OUTPUT: outputs}
