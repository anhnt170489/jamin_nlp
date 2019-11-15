import torch.nn as nn
import torch.nn.functional as f

from core.common import *
from libs.transformers import BertConfig, BertForTokenClassification


class BertTokenClassification(nn.Module):
    def __init__(
            self, args
    ) -> None:
        super(BertTokenClassification, self).__init__()

        self._label_map = {i: label for i, label in enumerate(args.labels)}
        self._num_labels = len(args.labels)

        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            num_labels=self._num_labels)
        self._bert_for_token_classification = BertForTokenClassification.from_pretrained(args.model_name_or_path,
                                                                                         from_tf=bool(
                                                                                             '.ckpt' in args.model_name_or_path),
                                                                                         config=config)

    def forward(self, batch):
        tokens = batch['tokens']
        labels = tokens[BERT_TOKEN_LABELS]
        token_mask = tokens[BERT_TOKEN_MASKS]

        outputs = self._bert_for_token_classification(tokens[BERT_INPUT_IDS], token_mask,
                                                      attention_mask=tokens[BERT_INPUT_MASKS],
                                                      labels=labels)
        if labels is not None:
            loss, logits = outputs[:2]
        else:
            loss = None
            logits = outputs[0]

        preds = f.softmax(logits, dim=-1).data
        preds = preds.argmax(dim=-1).view(-1)
        active_indices = token_mask.view(-1) == -1
        preds = preds.view(-1)[active_indices]
        preds = preds.detach().cpu().numpy()
        if labels is not None:
            golds = labels.view(-1)[active_indices]
            golds = golds.detach().cpu().numpy()
        else:
            golds = None

        return {LOSS: loss, PREDICT: preds, GOLD: golds,
                OUTPUT: outputs}
