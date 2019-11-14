import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from core.common import *
from core.meta import BertInstance
from libs.transformers import BertPreTrainedModel


class BertQuestionAnswering(BertPreTrainedModel):
    def __init__(
            self, bert_config, args
    ) -> None:
        super(BertQuestionAnswering, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels

        self._bert = BertInstance.get_bert(args)['model']
        self.qa_outputs = nn.Linear(bert_config.hidden_size, bert_config.num_labels)

        self.init_weights()
        self.device = args.device

    def forward(self, batch):
        tokens = batch['tokens']

        start_positions = [meta_data.start_position for meta_data in batch[META_DATA]]
        end_positions = [meta_data.end_position for meta_data in batch[META_DATA]]
        if all(v is not None for v in start_positions):
            start_positions = torch.tensor(start_positions, dtype=torch.long, device=self.device)
            end_positions = torch.tensor(end_positions, dtype=torch.long, device=self.device)
        else:
            start_positions, end_positions = None, None

        # cls_indexes = torch.tensor([meta_data['cls_index'] for meta_data in batch[META_DATA]],
        #                            dtype=torch.long,
        #                            device=self.device)
        # p_masks = tokens[BERT_P_MASKS]

        unique_ids = [meta_data.unique_id for meta_data in batch[META_DATA]]
        golds = [meta_data for meta_data in batch[META_DATA]]

        outputs = self._bert(tokens[BERT_INPUT_IDS],
                             attention_mask=tokens[BERT_INPUT_MASKS],
                             token_type_ids=tokens[BERT_SEGMENT_IDS])
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,) + outputs[2:]
        total_loss = None

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        start_logits = start_logits.detach().cpu().tolist()
        end_logits = end_logits.detach().cpu().tolist()

        if start_positions is not None and end_positions is not None:
            return {LOSS: total_loss, PREDICT: (unique_ids, start_logits, end_logits,), OUTPUT: outputs, GOLD: golds}
        else:
            return {LOSS: total_loss, PREDICT: (unique_ids, start_logits, end_logits, golds), OUTPUT: outputs,
                    GOLD: golds}
