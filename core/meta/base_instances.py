from collections import defaultdict
from typing import List

import torch

from core.common import *
from libs import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, PhobertTokenizer


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
        if isinstance(self.label_id, float):
            tensor = torch.tensor(self.label_id, dtype=torch.float)
        else:
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
        try:
            label_id = [self.label_map[lbl] for lbl in self.label_list]
        except:
            label_id = []
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


class TFInstance(SequenceInstance):
    """TRansformer instance, including Bert,Roberta, XLM-R based on huggingface's Transformer"""

    TOKENIZER_CLASSES = {
        'bert': BertTokenizerFast,
        'roberta': RobertaTokenizerFast,
        'vctrans': BertTokenizerFast,
        'xlm-r': XLMRobertaTokenizerFast,
        'phobert': PhobertTokenizer
    }

    __tokenizer = None

    @staticmethod
    def get_tokenizer(args=None):
        if not TFInstance.__tokenizer:
            tokenizer_class = TFInstance.TOKENIZER_CLASSES[args.model_type]
            TFInstance.__tokenizer = tokenizer_class.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                do_lower_case=args.do_lower_case)
        return TFInstance.__tokenizer

    def __init__(self, args):
        if not TFInstance.__tokenizer:
            TFInstance.get_tokenizer(args)

    def pad(self, padding_length):
        raise NotImplementedError()

    def to_tensors(self):
        raise NotImplementedError()


class TFTokenInstance(TFInstance):
    """Bert Token instance, a special instance of Bert Instance"""

    def __init__(self, args, tokens, token_labels=None):
        super().__init__(args)
        self.tokenizer = TFTokenInstance.get_tokenizer()
        self.tokens = tokens

        if token_labels:
            label_map = {label: i for i, label in enumerate(args.labels)}
            token_labels = [label_map[lbl] for lbl in token_labels]

        cls_token_at_end = bool(args.model_type in ['xlnet'])
        pad_on_left = bool(args.model_type in ['xlnet'])

        self.configs = defaultdict()
        self.configs['max_seq_length'] = args.max_seq_length
        self.configs['cls_token_at_end'] = cls_token_at_end
        self.configs['pad_on_left'] = pad_on_left
        self.configs['use_last_subword'] = args.use_last_subword
        self.configs['use_all_subwords'] = args.use_all_subwords
        # reservation
        self.configs['mask_padding_with_zero'] = True
        token_masks = []
        subword_tokens = []
        subtoken_labels = []
        for i, token in enumerate(self.tokens):
            subtokens = self.tokenizer.tokenize((token))
            subword_tokens.extend(subtokens)
            if token_labels:
                label = token_labels[i]

            if self.configs['use_last_subword']:
                sub_tokens_mask = [0] * len(subtokens)
                sub_tokens_mask[-1] = 1
                if token_labels:
                    subtoken_label = [-1] * len(subtokens)
                    subtoken_label[-1] = label
            elif not self.configs['use_all_subwords']:
                sub_tokens_mask = [0] * len(subtokens)
                sub_tokens_mask[0] = 1
                if token_labels:
                    subtoken_label = [-1] * len(subtokens)
                    subtoken_label[0] = label
            else:
                sub_tokens_mask = [1] * len(subtokens)
                if token_labels:
                    subtoken_label = [label] * len(subtokens)

            token_masks.extend(sub_tokens_mask)
            if token_labels:
                subtoken_labels.extend(subtoken_label)

        if token_labels:
            token_labels = subtoken_labels

        # Account for [CLS] and [SEP] with "- 2"
        special_tokens_count = 2
        if len(subword_tokens) > self.configs['max_seq_length'] - special_tokens_count:
            subword_tokens = subword_tokens[:(self.configs['max_seq_length'] - special_tokens_count)]
            token_masks = token_masks[:(self.configs['max_seq_length'] - special_tokens_count)]
            if token_labels:
                token_labels = token_labels[:(self.configs['max_seq_length'] - special_tokens_count)]

        sep_token = self.tokenizer.sep_token
        subword_tokens = subword_tokens + [sep_token]
        token_masks = token_masks + [0]
        if token_labels:
            token_labels = token_labels + [-1]

        cls_token = self.tokenizer.cls_token

        if self.configs['cls_token_at_end']:
            subword_tokens = subword_tokens + [cls_token]
            token_masks = token_masks + [0]
            if token_labels:
                token_labels = token_labels + [-1]
        else:
            subword_tokens = [cls_token] + subword_tokens
            token_masks = [0] + token_masks
            if token_labels:
                token_labels = [-1] + token_labels

        self.input_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        self.input_mask = [1 if self.configs['mask_padding_with_zero'] else 0] * len(self.input_ids)
        self.token_labels = token_labels
        self.token_masks = token_masks

    def pad(self, padding_length):
        length_to_pad = padding_length - len(self)
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        if self.configs['pad_on_left']:
            self.input_ids = ([pad_token] * length_to_pad) + self.input_ids
            self.input_mask = ([0 if self.configs['mask_padding_with_zero'] else 1] * length_to_pad) + self.input_mask
            if self.token_labels:
                self.token_labels = ([-1] * length_to_pad) + self.token_labels
            self.token_masks = ([0] * length_to_pad) + self.token_masks

        else:
            self.input_ids = self.input_ids + ([pad_token] * length_to_pad)
            self.input_mask = self.input_mask + ([0 if self.configs['mask_padding_with_zero'] else 1] * length_to_pad)
            if self.token_labels:
                self.token_labels = self.token_labels + ([-1] * length_to_pad)
            self.token_masks = self.token_masks + ([0] * length_to_pad)

        assert len(self.input_ids) == padding_length
        assert len(self.input_mask) == padding_length
        if self.token_labels:
            assert len(self.token_labels) == padding_length
        assert len(self.token_masks) == padding_length

        return self

    def to_tensors(self):
        return (torch.tensor(self.input_ids, dtype=torch.long),
                torch.tensor(self.input_mask, dtype=torch.long),
                torch.tensor(self.token_labels, dtype=torch.long) if self.token_labels else None,
                torch.tensor(self.token_masks, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)


class TFSequenceInstance(TFInstance):
    """Bert Sequence instance, a special instance of Bert Instance"""

    def __init__(self, args, sequence, pair=None):
        super().__init__(args)
        self.tokenizer = TFSequenceInstance.get_tokenizer()
        self.configs = defaultdict()
        self.configs['max_seq_length'] = args.max_seq_length
        pad_on_left = bool(args.model_type in ['xlnet'])
        pad_token_segment_id = 4 if args.model_type in ['xlnet'] else 0
        self.configs['pad_token_segment_id'] = pad_token_segment_id
        self.configs['pad_on_left'] = pad_on_left

        # reservation
        self.configs['mask_padding_with_zero'] = True
        inputs = self.tokenizer.encode_plus(
            sequence,
            pair,
            add_special_tokens=True,
            max_length=self.configs['max_seq_length'],
        )
        self.input_ids = inputs["input_ids"]
        if args.ignore_segment_ids:
            self.segment_ids = [0] * len(self.input_ids)
        else:
            self.segment_ids = inputs["token_type_ids"] if "token_type_ids" in inputs.keys() else None

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        self.input_mask = [1 if self.configs['mask_padding_with_zero'] else 0] * len(self.input_ids)

    def pad(self, padding_length):
        length_to_pad = padding_length - len(self)
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        if self.configs['pad_on_left']:
            self.input_ids = ([pad_token] * length_to_pad) + self.input_ids
            self.input_mask = ([0 if self.configs['mask_padding_with_zero'] else 1] * length_to_pad) + self.input_mask
            self.segment_ids = ([self.configs[
                                     'pad_token_segment_id']] * length_to_pad) + self.segment_ids if self.segment_ids is not None else None
        else:
            self.input_ids = self.input_ids + ([pad_token] * length_to_pad)
            self.input_mask = self.input_mask + ([0 if self.configs['mask_padding_with_zero'] else 1] * length_to_pad)
            self.segment_ids = self.segment_ids + ([self.configs[
                                                        'pad_token_segment_id']] * length_to_pad) if self.segment_ids is not None else None

        assert len(self.input_ids) == padding_length
        assert len(self.input_mask) == padding_length
        if self.segment_ids is not None:
            assert len(self.segment_ids) == padding_length

        return self

    def to_tensors(self):
        return (torch.tensor(self.input_ids, dtype=torch.long),
                torch.tensor(self.input_mask, dtype=torch.long),
                torch.tensor(self.segment_ids, dtype=torch.long) if self.segment_ids is not None else None,)

    def __len__(self):
        return len(self.input_ids)


class TFQAInstance(TFInstance):
    """Bert instance for QA task, a special instance of Bert Instance"""

    def __init__(self, args, doc_text='', query_text='', doc_tokens=None, query_tokens=None):
        super().__init__(args)
        self.tokenizer = TFQAInstance.get_tokenizer()
        self.query_tokens = query_tokens
        self.doc_tokens = doc_tokens

        query_segment_id = Q_MASK
        answer_segment_id = A_MASK
        if args.model_type == 'roberta':
            query_segment_id, answer_segment_id = 0, 0
        cls_token_segment_id = 2 if args.model_type in ['xlnet'] else 0
        pad_token_segment_id = 3 if args.model_type in ['xlnet'] else 0

        cls_token_at_end = bool(args.model_type in ['xlnet'])
        pad_on_left = bool(args.model_type in ['xlnet'])
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        sequence_a_is_doc = True if args.model_type in ['xlnet'] else False

        self.configs = defaultdict()
        self.configs['max_seq_length'] = args.max_seq_length
        self.configs['max_query_length'] = args.max_query_length
        self.configs['cls_token_at_end'] = cls_token_at_end
        self.configs['pad_on_left'] = pad_on_left
        self.configs['cls_token'] = cls_token
        self.configs['sep_token'] = sep_token
        self.configs['sequence_a_is_doc'] = sequence_a_is_doc
        self.configs['query_segment_id'] = query_segment_id
        self.configs['answer_segment_id'] = answer_segment_id
        self.configs['cls_token_segment_id'] = cls_token_segment_id
        self.configs['pad_token_segment_id'] = pad_token_segment_id
        # reservation
        self.configs['mask_padding_with_zero'] = True
        self.configs['pad_token'] = 0

        if not query_tokens:
            query_tokens = self.tokenizer.tokenize(query_text)
        if not doc_tokens:
            doc_tokens = self.tokenizer.tokenize(doc_text)

        if len(query_tokens) > self.configs['max_query_length']:
            query_tokens = query_tokens[0:self.configs['max_query_length']]

        tokens = []
        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = []
        segment_ids = []

        # CLS token at the beginning
        if not self.configs['cls_token_at_end']:
            tokens.append(self.configs['cls_token'])
            segment_ids.append(self.configs['cls_token_segment_id'])
            p_mask.append(A_MASK)
            cls_index = 0

        # XLNet: P SEP Q SEP CLS
        # Roberta: CLS Q SEP SEP P SEP
        # Others: CLS Q SEP P SEP
        if not self.configs['sequence_a_is_doc']:
            # Query
            tokens += query_tokens
            segment_ids += [self.configs['query_segment_id']] * len(query_tokens)
            p_mask += [Q_MASK] * len(query_tokens)

            # SEP token
            tokens.append(self.configs['sep_token'])
            segment_ids.append(self.configs['query_segment_id'])
            p_mask.append(Q_MASK)
            if args.model_type == 'roberta':
                tokens.append(self.configs['sep_token'])
                segment_ids.append(self.configs['query_segment_id'])
                p_mask.append(Q_MASK)

        # Add doc tokens
        for token in doc_tokens:
            tokens.append(token)
            segment_ids.append(self.configs['answer_segment_id'])
            p_mask.append(A_MASK)

        if self.configs['sequence_a_is_doc']:
            # SEP token
            tokens.append(self.configs['sep_token'])
            segment_ids.append(self.configs['query_segment_id'])
            p_mask.append(Q_MASK)

            tokens += query_tokens
            segment_ids += [self.configs['query_segment_id']] * len(query_tokens)
            p_mask += [Q_MASK] * len(query_tokens)

        # SEP token
        tokens.append(self.configs['sep_token'])
        segment_ids.append(self.configs['query_segment_id'])
        p_mask.append(Q_MASK)

        # CLS token at the end
        if self.configs['cls_token_at_end']:
            tokens.append(self.configs['cls_token_at_end'])
            segment_ids.append(self.configs['answer_segment_id'])
            p_mask.append(A_MASK)
            cls_index = len(tokens) - 1  # Index of classification token

        self.input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        self.input_mask = [1 if self.configs['mask_padding_with_zero'] else 0] * len(self.input_ids)

        self.p_mask = p_mask
        self.cls_index = cls_index
        self.tokens = tokens
        self.segment_ids = segment_ids

    def pad(self, padding_length):

        # Zero-pad up to the sequence length.
        while len(self.input_ids) < padding_length:
            self.input_ids.append(self.configs['pad_token'])
            self.input_mask.append(0 if self.configs['mask_padding_with_zero'] else 1)
            self.segment_ids.append(self.configs['pad_token_segment_id'])
            self.p_mask.append(Q_MASK)

        assert len(self.input_ids) == padding_length
        assert len(self.input_mask) == padding_length
        assert len(self.segment_ids) == padding_length

        return self

    def to_tensors(self):
        return (torch.tensor(self.input_ids, dtype=torch.long), torch.tensor(self.input_mask, dtype=torch.long),
                torch.tensor(self.p_mask, dtype=torch.long), torch.tensor(self.segment_ids, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)
