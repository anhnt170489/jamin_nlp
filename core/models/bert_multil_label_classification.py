import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from core.common import LOSS, PREDICT, GOLD, OUTPUT
from core.meta import BertInstance
from libs.transformers import BertPreTrainedModel


class BertMultilabelClassification(BertPreTrainedModel):
    def __init__(
            self, bert_config, args
    ) -> None:
        super(BertMultilabelClassification, self).__init__(bert_config)

        self._label_map = {i: label for i, label in enumerate(args.labels)}
        self._num_labels = len(args.labels)

        self._bert = BertInstance.get_bert()['model']
        self._dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self._classifier = nn.Linear(bert_config.hidden_size, self._num_labels)
        self._mlb = args.mlb

        self.init_weights()

    @staticmethod
    def flatten_labels(preds, label_ids):
        flat_preds = []
        flat_label_ids = []
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

        return flat_preds, flat_label_ids

    def forward(self, batch):
        tokens = batch['tokens']
        labels = batch['label']

        outputs = self._bert(tokens['input_ids'], attention_mask=tokens['input_mask'])
        sequence_output = outputs[1]

        sequence_output = self._dropout(sequence_output)
        logits = self._classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs  # (loss), logits, (hidden_states), (attentions)

            preds = torch.sigmoid(logits) > 0.5
            preds = preds.detach().cpu().numpy()
            golds = labels.detach().cpu().numpy()

            preds = self._mlb.inverse_transform(preds)
            golds = self._mlb.inverse_transform(golds)
            preds, golds = BertMultilabelClassification.flatten_labels(preds, golds)

            return {LOSS: loss, PREDICT: preds, GOLD: golds,
                    OUTPUT: outputs}
