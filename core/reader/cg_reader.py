import logging
import os
from collections import defaultdict

from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from core.common import META_DATA
from core.meta import BertTokenInstance, SpanInstance, MultiLabelInstance, ListInstance
from .standoff_data_reader import StandoffDataReader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class CGReader(StandoffDataReader):

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
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
        return ['Gene_expression', 'Mutation', 'Regulation', 'Development', 'Negative_regulation', 'Cell_proliferation',
                'Transcription', 'Glycosylation', 'Positive_regulation', 'Binding', 'Localization', 'Planned_process',
                'Metastasis', 'Death', 'Blood_vessel_development', 'Breakdown', 'Growth', 'Cell_transformation',
                'Carcinogenesis', 'Cell_differentiation', 'Cell_death', 'Cell_division', 'Infection', 'Pathway',
                'Dephosphorylation', 'Synthesis', 'Catabolism', 'Protein_processing', 'Remodeling', 'Metabolism',
                'Dissociation', 'Phosphorylation', 'Glycolysis', 'Translation', 'DNA_methylation', 'Reproduction',
                'Acetylation', 'Ubiquitination', 'Amino_acid_catabolism', 'DNA_demethylation', 'Gene_or_gene_product',
                'Cancer', 'Cell', 'Organism', 'DNA_domain_or_region', 'Simple_chemical', 'Multi-tissue_structure',
                'Organ', 'Organism_subdivision', 'Tissue', 'Immaterial_anatomical_entity', 'Organism_substance',
                'Protein_domain_or_region', 'Cellular_component', 'Pathological_formation', 'Amino_acid',
                'Anatomical_system', 'Developing_anatomical_structure']

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
                gold_span_labels = defaultdict(set)

                for mention in gold_mentions:
                    gold_spans.append((mention["start"], mention["end"], mention["label"]))
                    gold_span_labels[mention["start"], mention["end"]].add(mention["label"])

                spans = []
                span_labels = []

                for start, end in enumerate_spans(tokens, max_span_width=self.max_span_width):
                    spans.append(SpanInstance(start, end, sequence_instance))
                    span_labels.append(MultiLabelInstance(gold_span_labels.get((start, end), []), self.get_labels()))

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
