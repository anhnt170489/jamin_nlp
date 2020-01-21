import logging
import os
from collections import defaultdict

from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from core.common import META_DATA
from core.meta import BertTokenInstance, SpanInstance, LabelInstance, ListInstance
from .standoff_data_reader import StandoffDataReader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class SciERCReader(StandoffDataReader):

    def __init__(self, args):
        self.max_span_width = args.max_span_width
        self.args = args

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_standoff(os.path.join(data_dir, "train/")))

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_standoff(os.path.join(data_dir, "dev/")))

    def get_labels(self):
        return ['Material', 'OtherScientificTerm', 'Generic', 'Method', 'Task', 'Metric', 'None']

    @property
    def label_map(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_standoff(os.path.join(data_dir, "test/")))

    def _create_instances(self, docs):
        logger.info("Reading instances")
        intances = []
        for doc_id, doc in docs.items():
            for sentence_index, sentence in enumerate(doc["sentences"]):
                tokens = sentence["tokens"]
                sentence_index = sentence_index,
                gold_mentions = sentence["mentions"]

                sequence_instance = BertTokenInstance(tokens=tokens, args=self.args)

                gold_spans = []
                gold_span_labels = defaultdict()

                for mention in gold_mentions:
                    gold_spans.append((mention["start"], mention["end"], mention["label"]))
                    gold_span_labels[mention["start"], mention["end"]] = mention["label"]

                spans = []
                span_labels = []

                for start, end in enumerate_spans(tokens, max_span_width=self.max_span_width):
                    spans.append(SpanInstance(start, end, sequence_instance))
                    label = gold_span_labels.get((start, end), 'None')
                    label_id = self.label_map[label]
                    span_labels.append(LabelInstance(label_id, label))

                metadata = {
                    "doc_id": doc_id,
                    "sentence_index": sentence_index,
                    "tokens": tokens,
                    "gold_spans": gold_spans,
                }
                intances.append({
                    "tokens": sequence_instance,
                    "spans": ListInstance(spans),
                    "span_labels": ListInstance(span_labels),
                    META_DATA: metadata,
                })

        return intances
