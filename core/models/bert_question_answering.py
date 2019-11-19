import torch
import torch.nn as nn

from core.common import *
from libs import BertConfig, RobertaConfig, BertForQuestionAnswering
from .modeling_roberta import RobertaForQuestionAnswering


class BertQuestionAnswering(nn.Module):
    def __init__(
            self, args
    ) -> None:
        super(BertQuestionAnswering, self).__init__()
        if args.model_type == 'roberta':
            config_cls, model_cls_for_qa = RobertaConfig, RobertaForQuestionAnswering
        elif args.model_type == 'bert':
            config_cls, model_cls_for_qa = BertConfig, BertForQuestionAnswering

        config = config_cls.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        self._bert_for_qa = model_cls_for_qa.from_pretrained(args.model_name_or_path,
                                                             from_tf=bool('.ckpt' in args.model_name_or_path),
                                                             config=config)

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

        outputs = self._bert_for_qa(tokens[BERT_INPUT_IDS],
                                    attention_mask=tokens[BERT_INPUT_MASKS],
                                    token_type_ids=tokens[BERT_SEGMENT_IDS],
                                    start_positions=start_positions,
                                    end_positions=end_positions,
                                    )
        if start_positions is not None and end_positions is not None:
            start_logits = outputs[1].detach().cpu().tolist()
            end_logits = outputs[2].detach().cpu().tolist()
            return {LOSS: outputs[0], PREDICT: (unique_ids, start_logits, end_logits,), OUTPUT: outputs, GOLD: golds}
        else:
            total_loss = None
            start_logits = outputs[0].detach().cpu().tolist()
            end_logits = outputs[1].detach().cpu().tolist()
            return {LOSS: total_loss, PREDICT: (unique_ids, start_logits, end_logits, golds), OUTPUT: outputs,
                    GOLD: golds}
