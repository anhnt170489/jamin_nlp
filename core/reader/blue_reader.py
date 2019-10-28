import logging
import os

from tqdm import tqdm

from core.common import META_DATA
from core.meta import BLUEInstance, BertInstance, LabelInstance, MultiLabelInstance
from .tsv_data_reader import TSVDataReader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class BLUEReader(TSVDataReader):

    def __init__(self, args):
        super().__init__()
        self.args = args

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
        return ["0", "1", "2", "3", "4"]

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
            label = float(line[-1])
            examples.append(BLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            if instance.text_b:
                text = instance.text_a + ' ' + instance.text_b
            else:
                text = instance.text_a
            sequence_instance = BertInstance(text=text, args=self.args)
            label = LabelInstance(self.label_map[instance.label], instance.label)
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
        return ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

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
                token_labels.append(line[1].split()[-1])

        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            tokens = instance.text_a
            sequence_instance = BertInstance(tokens=tokens, token_labels=instance.label, args=self.args)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                META_DATA: metadata
            })
        return intances


class HOCReader(BLUEReader):

    def get_labels(self):
        return ['sustaining', 'proliferative', 'signaling', 'evading', 'growth', 'suppressors', 'resisting', 'cell',
                'death', 'avoiding', 'immune', 'destruction', 'activating', 'invasion', 'metastasis', 'tumor',
                'promoting', 'inflammation', 'enabling', 'replicative', 'immortality', 'genomic', 'instability',
                'mutation', 'inducing', 'angiogenesis', 'cellular', 'energetics']

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = None
            label = line[2].replace('and', ' ').replace(',', ' ').split()
            examples.append(BLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_instances(self, lines, set_type):

        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            if instance.text_b:
                text = instance.text_a + ' ' + instance.text_b
            else:
                text = instance.text_a
            sequence_instance = BertInstance(text=text, args=self.args)
            label = MultiLabelInstance(instance.label, self.get_labels())
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances


class DDI2013Readers(BLUEReader):

    def get_labels(self):
        return ["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism", 'DDI-false']

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
            if instance.text_b:
                text = instance.text_a + ' ' + instance.text_b
            else:
                text = instance.text_a
            sequence_instance = BertInstance(text=text, args=self.args)
            label = LabelInstance(self.label_map[instance.label], instance.label)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances


class ChemProtReaders(BLUEReader):

    def get_labels(self):
        return ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]

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
            if instance.text_b:
                text = instance.text_a + ' ' + instance.text_b
            else:
                text = instance.text_a
            sequence_instance = BertInstance(text=text, args=self.args)
            label = LabelInstance(self.label_map[instance.label], instance.label)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances
