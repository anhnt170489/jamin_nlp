import logging
import os

from tqdm import tqdm

from core.common import META_DATA
from core.meta import BLUEInstance, BertSequenceInstance, BertTokenInstance, LabelInstance, MultiLabelInstance
from .json_data_reader import JsonDataReader
from .tsv_data_reader import TSVDataReader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class BLUEReader(TSVDataReader):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def get_labels(self):
        return None

    def get_ignored_labels(self):
        return None

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


class BiossesReader(BLUEReader):
    """Processor for the Biosses data set (BLUE version)."""

    def get_labels(self):
        """See base class."""
        # return ["0", "1", "2", "3", "4"]
        return ["Regression"]

    @property
    def label_map(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[-3]
            text_b = line[-2]
            # label = str(round(float(line[-1])))
            label = float(line[-1])
            examples.append(BLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            sequence_instance = BertSequenceInstance(sequence=instance.text_a, pair=instance.text_b, args=self.args)
            # label = LabelInstance(self.label_map[instance.label], instance.label)
            label = LabelInstance(instance.label, '')
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances


class BC5CDRReader(BLUEReader):
    """Processor for the BC5CDR data set (BLUE version)."""

    def get_labels(self):
        """See base class."""
        # return ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
        return ["O", "B", "I"]

    def get_ignored_labels(self):
        return ["O"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        idx = 0
        tokens = []
        token_labels = []
        for line in lines:
            if len(line) < 2:
                guid = "%s-%s" % (set_type, idx)
                text_b = None
                examples.append(
                    BLUEInstance(guid=guid, text_a=tokens, text_b=text_b, label=token_labels))
                idx += 1
                tokens = []
                token_labels = []
            else:
                tokens.append(line[0])
                token_labels.append(line[1].split()[-1][0])

        # last sample
        if len(tokens) > 0:
            guid = "%s-%s" % (set_type, idx)
            text_b = None
            examples.append(
                BLUEInstance(guid=guid, text_a=tokens, text_b=text_b, label=token_labels))

        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            tokens = instance.text_a
            sequence_instance = BertTokenInstance(tokens=tokens, token_labels=instance.label, args=self.args)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                META_DATA: metadata
            })
        return intances


class HOCReader(BLUEReader):

    def get_labels(self):
        return ['sustaining proliferative signaling', 'evading growth suppressors', 'resisting cell death',
                'avoiding immune destruction', 'activating invasion and metastasis', 'tumor promoting inflammation',
                'enabling replicative immortality', 'genomic instability and mutation', 'inducing angiogenesis',
                'cellular energetics']

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = None
            label = line[2].split(',')
            examples.append(BLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            sequence_instance = BertSequenceInstance(sequence=instance.text_a, pair=instance.text_b, args=self.args)
            label = MultiLabelInstance(instance.label, self.get_labels())
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances


class DDI2013Reader(BLUEReader):

    def get_labels(self):
        return ["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism", 'DDI-false']

    def get_ignored_labels(self):
        return ['DDI-false']

    @property
    def label_map(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = line[0]
            text_a = line[1]
            if set_type == "test":
                label = self.get_labels()[-1]
            else:
                try:
                    label = line[2]
                except IndexError:
                    logging.exception(line)
                    exit(1)
            examples.append(BLUEInstance(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            sequence_instance = BertSequenceInstance(sequence=instance.text_a, pair=instance.text_b, args=self.args)
            label = LabelInstance(self.label_map[instance.label], instance.label)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances


class ChemProtReader(BLUEReader):

    def get_labels(self):
        return ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]

    def get_ignored_labels(self):
        return ["false"]

    @property
    def label_map(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = line[0]
            text_a = line[1]
            if set_type == "test":
                label = self.get_labels()[-1]
            else:
                try:
                    label = line[2]
                except IndexError:
                    logging.exception(line)
                    exit(1)
            examples.append(BLUEInstance(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            sequence_instance = BertSequenceInstance(sequence=instance.text_a, pair=instance.text_b, args=self.args)
            label = LabelInstance(self.label_map[instance.label], instance.label)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances


class MedNLIReader(JsonDataReader):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_json(os.path.join(data_dir, "train.json")))

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_json(os.path.join(data_dir, "dev.json")))

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_json(os.path.join(data_dir, "test.json")))

    def get_labels(self):
        """See base class."""
        return ["entailment", "contradiction", "neutral"]

    @property
    def label_map(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def _create_text_instances(self, json_data):
        examples = []
        for json_obj in json_data:
            guid = json_obj['pairID']
            text_a = json_obj['sentence1']
            text_b = json_obj['sentence2']
            label = json_obj['gold_label']
            examples.append(BLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_instances(self, json_data):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(json_data)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            sequence_instance = BertSequenceInstance(sequence=instance.text_a, pair=instance.text_b, args=self.args)
            label = LabelInstance(self.label_map[instance.label], instance.label)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances
