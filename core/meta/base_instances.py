from collections import defaultdict
from typing import List

import torch

from libs.transformers import BertConfig, BertTokenizer, BertModel, \
    RobertaConfig, RobertaTokenizer, RobertaModel


class TensorInstance(object):
    def to_tensor(self):
        raise NotImplementedError()

    @property
    def empty_tensor(self):
        raise NotImplementedError()


class SequenceInstance(object):
    def pad(self):
        raise NotImplementedError()


class SpanInstance(TensorInstance):
    """A common span instance"""

    def __init__(self, start, end, sequence):
        self.start = start
        self.end = end
        self.sequence = sequence

    def to_tensor(self):
        tensor = torch.tensor([self.start, self.end], dtype=torch.long)
        return tensor

    @property
    def empty_tensor(self):
        return SpanInstance(-1, -1, None)


class LabelInstance(TensorInstance):
    """A common label instance"""

    def __init__(self, label_id, label_text):
        self.label_id = label_id
        self.label_text = label_text

    def to_tensor(self):
        tensor = torch.tensor(self.label_id, dtype=torch.long)
        return tensor

    @property
    def empty_tensor(self):
        return LabelInstance(-1, None)


class MultiLabelInstance(TensorInstance):
    """A common label instance"""

    def __init__(self, label_list, label_namespace):
        self.label_list = label_list
        self.label_namespace = label_namespace

    @property
    def label_id(self):
        label_id = [self.label_map[lbl] for lbl in self.label_list]
        return label_id

    @property
    def label_map(self):
        return {label: i for i, label in enumerate(['None'] + self.label_namespace)}

    @property
    def reverse_label_map(self):
        return {i: label for i, label in enumerate(['None'] + self.label_namespace)}

    @property
    def mlb(self):
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit([sorted(self.label_map.values())[1:]])  # [1:] skip label O
        return mlb

    @property
    def empty_tensor(self):
        return MultiLabelInstance([], self.label_namespace)

    def to_tensor(self):
        label_id = self.mlb.transform([self.label_id])[-1]
        tensor = torch.tensor(label_id, dtype=torch.float)
        return tensor


class ListInstance(TensorInstance, SequenceInstance):
    def __init__(self,
                 list_instances: List[TensorInstance]):
        self.list_instances = list_instances

    def to_tensor(self):

        tensor = torch.stack([instance.to_tensor() for instance in self.list_instances], dim=0)
        return tensor

    def __len__(self):
        return len(self.list_instances)

    @property
    def empty_tensor(self):
        if len(self.list_instances) > 0:
            return ListInstance(self.list_instances[0].empty_tensor * len(self.list_instances))
        else:
            return ListInstance(self.list_instances)

    def pad(self, padding_length):
        length_to_pad = padding_length - len(self.list_instances)
        if len(self.list_instances) > 0:
            self.list_instances += [self.list_instances[0].empty_tensor] * length_to_pad
            return self
        else:
            return self.empty_tensor


class BertInstance(SequenceInstance):
    """Bert instance, including Bert,Roberta, based on huggingface's Transformer"""

    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizer, BertModel),
        'roberta': (RobertaConfig, RobertaTokenizer, RobertaModel)
    }

    __bert = None

    @staticmethod
    def get_bert(args=None):
        if not BertInstance.__bert:
            config_class, tokenizer_class, model_class = BertInstance.MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
            tokenizer = tokenizer_class.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                do_lower_case=args.do_lower_case)
            model = model_class.from_pretrained(args.model_name_or_path,
                                                from_tf=bool('.ckpt' in args.model_name_or_path),
                                                config=config)
            BertInstance.__bert = {'tokenizer': tokenizer, 'model': model, 'model_config': config}
        return BertInstance.__bert

    def __init__(self, args, tokens=None, text=None, token_labels=None):
        if not BertInstance.__bert:
            BertInstance.get_bert(args)
        self.tokenizer = BertInstance.__bert['tokenizer']
        self.model = BertInstance.__bert['model']
        if not tokens:
            self.tokens = self.tokenizer.basic_tokenizer.tokenize(text)
        else:
            self.tokens = tokens
        if token_labels:
            label_map = {label: i for i, label in enumerate(args.labels)}
            token_labels = [label_map[lbl] for lbl in token_labels]

        sep_token_extra = bool(args.model_type in ['roberta'])
        cls_token_at_end = bool(args.model_type in ['xlnet'])
        pad_on_left = bool(args.model_type in ['xlnet'])

        self.configs = defaultdict()
        self.configs['model_config'] = BertInstance.__bert['model_config']
        self.configs['sep_token_extra'] = sep_token_extra
        self.configs['max_seq_length'] = args.max_seq_length
        self.configs['cls_token_at_end'] = cls_token_at_end
        self.configs['pad_on_left'] = pad_on_left
        self.configs['use_last_subword'] = args.use_last_subword
        self.configs['use_all_subwords'] = args.use_all_subwords
        # reservation
        self.configs['mask_padding_with_zero'] = True

        if self.configs['use_all_subwords']:
            import functools, operator
            subword_tokens = [self.tokenizer.tokenize((token)) for token in self.tokens]
            subword_tokens = functools.reduce(operator.iconcat, subword_tokens, [])
        elif self.configs['use_last_subword']:
            subword_tokens = [self.tokenizer.tokenize((token))[-1] for token in self.tokens]
        else:
            subword_tokens = [self.tokenizer.tokenize((token))[0] for token in self.tokens]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if self.configs['sep_token_extra'] else 2
        if len(subword_tokens) > self.configs['max_seq_length'] - special_tokens_count:
            subword_tokens = subword_tokens[:(self.configs['max_seq_length'] - special_tokens_count)]
            if token_labels:
                token_labels = token_labels[:(self.configs['max_seq_length'] - special_tokens_count)]

        sep_token = self.tokenizer.sep_token
        subword_tokens = subword_tokens + [sep_token]
        if token_labels:
            token_labels = token_labels + [-1]

        if self.configs['sep_token_extra']:
            # roberta uses an extra separator b/w pairs of sentences
            subword_tokens += [sep_token]
            if token_labels:
                token_labels = token_labels + [-1]
        cls_token = self.tokenizer.cls_token

        if self.configs['cls_token_at_end']:
            subword_tokens = subword_tokens + [cls_token]
            if token_labels:
                token_labels = token_labels + [-1]
        else:
            subword_tokens = [cls_token] + subword_tokens
            if token_labels:
                token_labels = [-1] + token_labels

        self.input_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        self.input_mask = [1 if self.configs['mask_padding_with_zero'] else 0] * len(self.input_ids)
        self.token_labels = token_labels

    def pad(self, padding_length):
        length_to_pad = padding_length - len(self)
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        if self.configs['pad_on_left']:
            self.input_ids = ([pad_token] * length_to_pad) + self.input_ids
            self.input_mask = ([0 if self.configs['mask_padding_with_zero'] else 1] * length_to_pad) + self.input_mask
            if self.token_labels:
                self.token_labels = ([-1] * length_to_pad) + self.token_labels

        else:
            self.input_ids = self.input_ids + ([pad_token] * length_to_pad)
            self.input_mask = self.input_mask + ([0 if self.configs['mask_padding_with_zero'] else 1] * length_to_pad)
            if self.token_labels:
                self.token_labels = self.token_labels + ([-1] * length_to_pad)

        assert len(self.input_ids) == padding_length
        assert len(self.input_mask) == padding_length
        if self.token_labels:
            assert len(self.token_labels) == padding_length

        return self

    def to_tensors(self):
        return (torch.tensor(self.input_ids, dtype=torch.long), torch.tensor(self.input_mask, dtype=torch.long),
                torch.tensor(self.token_labels, dtype=torch.long) if self.token_labels else None)

    def __len__(self):
        return len(self.input_ids)
