import torch.nn as nn
import torch.nn.functional as f
from torch.nn import CrossEntropyLoss

from core.common import LOSS, PREDICT, GOLD, OUTPUT
from core.meta import BertInstance
from libs.transformers import BertPreTrainedModel


class BertTokenClassification(BertPreTrainedModel):
    def __init__(
            self, bert_config, args
    ) -> None:
        super(BertTokenClassification, self).__init__(bert_config)

        self._label_map = {i: label for i, label in enumerate(args.labels)}
        self._num_labels = len(args.labels)

        self._bert = BertInstance.get_bert(self.args)['model']
        self._dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self._classifier = nn.Linear(bert_config.hidden_size, self._num_labels)

        self.init_weights()

    def forward(self, batch):
        tokens = batch['tokens']
        labels = tokens['token_labels']

        outputs = self._bert(tokens['input_ids'], attention_mask=tokens['input_mask'])
        sequence_output = outputs[0]
        sequence_output = self._dropout(sequence_output)
        logits = self._classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if tokens['input_mask'] is not None:
                # active_loss = attention_mask.view(-1) == 1
                active_loss = labels.view(-1) != -1
                active_logits = logits.view(-1, self._num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self._num_labels), labels.view(-1))
            outputs = (loss,) + outputs  # (loss), scores, (hidden_states), (attentions)

            preds = f.softmax(logits, dim=-1).data
            preds = preds.argmax(dim=-1).view(-1)
            active_indices = labels.view(-1) != -1
            preds = preds.view(-1)[active_indices]
            golds = labels.view(-1)[active_indices]

            preds = preds.detach().cpu().numpy()
            golds = golds.detach().cpu().numpy()

            return {LOSS: loss, PREDICT: preds, GOLD: golds,
                    OUTPUT: outputs}
