import numpy as np
from torch import nn

from core.common import *
from libs import BertConfig, BertForSequenceClassification, XLMRobertaConfig, XLMRobertaForSequenceClassification, \
    RobertaConfig, RobertaForSequenceClassification

TF_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification),
    'roberta': (RobertaConfig, RobertaForSequenceClassification),
    'xlm-r': (XLMRobertaConfig, XLMRobertaForSequenceClassification),
    'phobert': (RobertaConfig, RobertaForSequenceClassification),
}


class TFSequenceClassification(nn.Module):
    def __init__(
            self, args
    ) -> None:
        super(TFSequenceClassification, self).__init__()
        self._num_labels = len(args.labels)
        config_cls, tf_for_seq_classification_cls = TF_CLASSES[args.model_type]

        config = config_cls.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            num_labels=self._num_labels)
        self._bert_for_sequence_classification = tf_for_seq_classification_cls.from_pretrained(args.model_name_or_path,
                                                                                               from_tf=bool(
                                                                                                   '.ckpt' in args.model_name_or_path),
                                                                                               config=config)

    def forward(self, batch):
        tokens = batch['tokens']
        labels = batch['label'] if 'label' in batch else None

        outputs = self._bert_for_sequence_classification(tokens[TF_INPUT_IDS],
                                                         attention_mask=tokens[TF_INPUT_MASKS],
                                                         token_type_ids=tokens[TF_SEGMENT_IDS],
                                                         labels=batch['label'])
        if labels is not None:
            loss, logits = outputs[:2]
        else:
            loss = None
            logits = outputs[0]
        if self._num_labels == 1:
            #  We are doing regression
            preds = np.squeeze(logits.detach().cpu().numpy())
        else:
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)

        golds = labels.detach().cpu().numpy()

        return {LOSS: loss, PREDICT: preds, GOLD: golds,
                OUTPUT: outputs}
