import numpy as np
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss

from core.common import LOSS, PREDICT, GOLD, OUTPUT
from core.meta import BertInstance
from libs.transformers import BertPreTrainedModel


class BertSequenceClassification(BertPreTrainedModel):
    def __init__(
            self, bert_config, args
    ) -> None:
        super(BertSequenceClassification, self).__init__(bert_config)

        self._label_map = {i: label for i, label in enumerate(args.labels)}
        self._num_labels = len(args.labels)

        self._bert = BertInstance.get_bert(self.args)['model']
        self._dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self._classifier = nn.Linear(bert_config.hidden_size, self._num_labels)

        self.init_weights()

    def forward(self, batch):
        tokens = batch['tokens']
        labels = batch['label']

        outputs = self._bert(tokens['input_ids'], attention_mask=tokens['input_mask'])
        pooled_output = outputs[1]

        pooled_output = self._dropout(pooled_output)
        logits = self._classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self._num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self._num_labels), labels.view(-1))
            outputs = (loss,) + outputs  # (loss), logits, (hidden_states), (attentions)

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            golds = labels.detach().cpu().numpy()

            return {LOSS: loss, PREDICT: preds, GOLD: golds,
                    OUTPUT: outputs}
