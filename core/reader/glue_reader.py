import logging
import os

from tqdm import tqdm

from core.common import META_DATA
from core.meta import BertSequenceInstance, LabelInstance
from core.meta import GLUEInstance
from .tsv_data_reader import TSVDataReader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class GLUEReader(TSVDataReader):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def label_map(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    def _create_text_instances(self, lines, set_type):
        raise NotImplementedError()

    def _create_instances(self, lines, set_type, task='classification'):
        logger.info("Reading text instances")
        text_instances = self._create_text_instances(lines, set_type)
        logger.info("Convert text instances to model instances")
        intances = []
        for instance in tqdm(text_instances):
            sequence_instance = BertSequenceInstance(sequence=instance.text_a, pair=instance.text_b, args=self.args)
            if task == 'classification':
                label = LabelInstance(self.label_map[instance.label], instance.label)
            elif task == 'regression':
                label = LabelInstance(instance.label, '')
            else:
                raise ValueError("Task not found: %s" % task)
            metadata = {'guid': instance.guid}
            intances.append({
                'tokens': sequence_instance,
                'label': label,
                META_DATA: metadata
            })
        return intances


class ColaReader(GLUEReader):
    """Processor for the CoLA data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            float(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MnliReader(GLUEReader):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedReader(MnliReader):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class MrpcReader(GLUEReader):
    """Processor for the MRPC data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliReader(GLUEReader):
    """Processor for the QNLI data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['question'].numpy().decode('utf-8'),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpReader(GLUEReader):
    """Processor for the QQP data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['question1'].numpy().decode('utf-8'),
                            tensor_dict['question2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteReader(GLUEReader):
    """Processor for the RTE data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Sst2Reader(GLUEReader):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbReader(GLUEReader):
    """Processor for the STS-B data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            float(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task='regression')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", task='regression')

    def get_labels(self):
        """See base class."""
        return ['Regression']

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = float(line[-1])
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliReader(GLUEReader):
    """Processor for the WNLI data set (GLUE version)."""

    def get_text_instances_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return GLUEInstance(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_instances(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_text_instances(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                GLUEInstance(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
