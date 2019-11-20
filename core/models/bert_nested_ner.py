# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn import util

# from protonlp.training.metrics import SpanF1Measure

logger = logging.getLogger(__name__)

from libs import BertConfig, BertModel

from core.common import *


class BertNestedNER(nn.Module):
    def __init__(
            self, args
    ) -> None:

        super(BertNestedNER, self).__init__()

        self._num_labels = len(args.labels)

        self._label_mapping = {i: label for i, label in enumerate(args.labels)}

        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            num_labels=self._num_labels)
        self._bert = BertModel.from_pretrained(args.model_name_or_path,
                                               from_tf=bool('.ckpt' in args.model_name_or_path),
                                               config=config)
        self._global_attention = nn.Linear(
            config.hidden_size, 1
        )

        self._classifier = nn.Linear(
            config.hidden_size * 3, self._num_labels
        )
        self._label_type = args.label_type
        self._label_threshold = args.multi_label_threshold

    def forward(self, batch):

        tokens = batch['tokens']
        spans = batch['spans']
        batch_metadata = batch[META_DATA]
        span_labels = batch['span_labels']

        # Extract token embeddings
        # sequence_embeddings = self._dropout(
        #     self._text_field_embedder(tokens)
        # )  # (B, S, H)

        sequence_embeddings = self._bert(tokens[BERT_INPUT_IDS], attention_mask=tokens[BERT_INPUT_MASKS])[0]

        # Compute attention scores over each sequence embeddings
        global_attention_logits = self._global_attention(
            sequence_embeddings
        )  # (B, S, 1)

        # Retrieve sequence masks to remove tokens padded
        # sequence_mask = util.get_text_field_mask(tokens) > 0  # (B, S)
        sequence_mask = tokens[BERT_INPUT_MASKS] > 0

        # Remove token embeddings at padded positions
        sequence_embeddings = sequence_embeddings[sequence_mask]  # (S', H)

        # Similarly, remove attention logits at padded positions
        global_attention_logits = global_attention_logits[sequence_mask]  # (S', 1)

        # Infer sequence lengths from mask matrix
        sequence_lengths = util.get_lengths_from_binary_sequence_mask(
            sequence_mask
        )  # (B,)

        # Mask invalid spans due to padding or BERT's truncation
        span_mask = (0 <= spans[:, :, -1]) & (
                spans[:, :, -1] < sequence_lengths.unsqueeze(-1)
        )  # (B, N)

        # This array is used to map span indices to its corresponding sentence index
        sentence_index_mapping = (
            torch.arange(len(spans)).unsqueeze(-1).expand(span_mask.size())
        )[
            span_mask
        ].tolist()  # (N',)

        # The span indices will be modified directly so need to backup the original span indices first for later evaluation
        original_spans = spans[span_mask].tolist()  # (N', 2)

        # Because all sequences in the batch are now concatenated as a long sequence, so have to correct the span indices
        spans[1:] += sequence_lengths.cumsum(dim=0)[:-1].view(-1, 1, 1)  # (B, N, 2)

        # Retrieve valid spans using mask matrix
        spans = spans[span_mask]  # (N', 2)

        span_starts, span_ends = spans.split(1, dim=-1)  # (N', 1) (N', 1)

        # Obtain embeddings using start and end indices
        start_embeddings = sequence_embeddings[span_starts.squeeze(-1)]  # (N', H)
        end_embeddings = sequence_embeddings[span_ends.squeeze(-1)]  # (N', H)

        # Compute width for each spans
        span_widths = span_ends - span_starts  # (N', 1)

        # Create an intermediate-range vector for building a list of indices from start to end
        max_span_range_indices = util.get_range_vector(
            span_widths.max().item() + 1, util.get_device_of(sequence_embeddings)
        ).unsqueeze(
            0
        )  # (1, span_width)

        # Ignore span indices exceeding corresponding span width
        attended_span_mask = max_span_range_indices <= span_widths  # (N', span_width)

        # Create an intermediate-index matrix for building a list of indices from start to end
        raw_span = span_ends - max_span_range_indices  # (N', span_width)

        # Replace all negative indices by zero
        spans = F.relu(raw_span)  # (N', span_width)

        # Obtain embeddings using the matrix of indices
        span_embeddings = sequence_embeddings[spans]  # (N', span_width, H)

        # Obtain corresponding attention logits using the matrix of indices
        span_attention_logits = global_attention_logits[spans]  # (N', span_width, 1)

        # Compute Softmax over valid attention logits of a span (don't take padding spans into account)
        span_attention_weights = util.masked_softmax(
            span_attention_logits.squeeze(-1), attended_span_mask
        )  # (N', span_width)

        # Compute weighted embeddings of spans
        attended_embeddings = util.weighted_sum(
            span_embeddings, span_attention_weights
        )  # (N', H)

        # Form span embeddings following Sohrab's method:
        # Deep Exhaustive Model for Nested Named Entity Recognition
        span_embeddings = torch.cat(
            (start_embeddings, attended_embeddings, end_embeddings), dim=-1
        )  # (N', H)

        logits = self._classifier(span_embeddings)  # (N', num_labels)
        predicted_spans = [[] for _ in batch_metadata]
        gold_spans = [metadata["gold_spans"] for metadata in batch_metadata]
        loss = None

        # In case of Multi-label:
        if self._label_type == 'M':
            # Use sigmoid for multi-label classification task
            probabilities = torch.sigmoid(logits.detach().cpu())  # (N', num_labels)
            predictions = probabilities >= self._label_threshold  # (N', num_labels)
            for span_index, label_id in predictions.nonzero().tolist():
                predicted_span = original_spans[span_index]
                predicted_label = self._label_mapping[label_id]
                predicted_spans[sentence_index_mapping[span_index]].append(
                    (*predicted_span, predicted_label)
                )
            if isinstance(span_labels, torch.Tensor):
                span_labels = span_labels[span_mask].float()
                loss = F.multilabel_soft_margin_loss(logits, span_labels)
        else:
            probabilities = torch.softmax(logits.detach().cpu(), dim=-1)
            predictions = probabilities.argmax(dim=-1)
            for span_index, label_id in enumerate(predictions.tolist()):
                predicted_span = original_spans[span_index]
                predicted_label = self._label_mapping[label_id]
                predicted_spans[sentence_index_mapping[span_index]].append(
                    (*predicted_span, predicted_label)
                )
            if isinstance(span_labels, torch.Tensor):
                loss = F.cross_entropy(logits, span_labels[span_mask])

        outputs = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "doc_ids": [metadata["doc_id"] for metadata in batch_metadata],
            "sentence_indices": [metadata["sentence_index"] for metadata in batch_metadata],
        }

        return {LOSS: loss,
                PREDICT: predicted_spans,
                GOLD: gold_spans,
                OUTPUT: outputs}
