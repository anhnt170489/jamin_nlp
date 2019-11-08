import logging
import os
from collections import defaultdict
from glob import glob

import regex

from utils import read_text, read_lines

MENTION_PATTERN = regex.compile(
    r"^(T[\w-]+)\t([\w-]+) (\d+) (\d+)\t(.+)$", flags=regex.MULTILINE
)

REFERENCE_PATTERN = regex.compile(
    r"^(N[\w-]+)\tReference ([\w-]+) ([\w-]+):([\w-]+)(?:\t(.+))?$",
    flags=regex.MULTILINE,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class StandoffDataReader(object):

    def get_train_examples(self, data_dir):
        """Gets a collection of instances for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of instances for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of instances for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_standoff(self, corpus_dir, encoding="UTF-8"):
        docs = {}

        for filename in glob(os.path.join(corpus_dir, "*.ann")):
            doc = read_text(filename.replace(".ann", ".txt"), encoding=encoding)

            cursor = 0
            start_offsets = {}
            end_offsets = {}

            sentences = []

            for sentence_index, sentence in enumerate(doc.split("\n")):
                tokens = sentence.split(" ")

                for token_index, token in enumerate(tokens):
                    start_offsets[cursor] = (sentence_index, token_index)
                    end_offsets[cursor + len(token)] = (sentence_index, token_index)
                    cursor += len(token) + 1

                sentences.append({"tokens": tokens, "mentions": []})

            assert len(doc) == cursor - 1

            mentions = {}
            references = {}

            for line in read_lines(filename, encoding=encoding):
                if line.startswith("T"):
                    matcher = MENTION_PATTERN.match(line)

                    mention_id, mention_label, mention_start_offset, mention_end_offset, mention_string = (
                        matcher.groups()
                    )

                    assert mention_id not in mentions

                    if mention_label in self.get_labels():
                        mentions[mention_id] = {
                            "id": mention_id,
                            "label": mention_label,
                            "start": int(mention_start_offset),
                            "end": int(mention_end_offset),
                            "string": mention_string,
                            "references": {},
                        }
                elif line.startswith("N"):
                    matcher = REFERENCE_PATTERN.match(line)

                    reference_id, mention_id, resource_name, record_id, reference_string = (
                        matcher.groups()
                    )

                    assert reference_id not in references

                    references[reference_id] = {
                        "id": reference_id,
                        "mention": mention_id,
                        "resource": resource_name,
                        "record": record_id,
                        "string": reference_string,
                    }

            for reference in references.values():
                if reference["mention"].startswith("T"):
                    resource_record_pair = (reference["resource"], reference["record"])

                    assert (
                            resource_record_pair
                            not in mentions[reference["mention"]]["references"]
                    )

                    mentions[reference["mention"]]["references"][
                        resource_record_pair
                    ] = reference["string"]

            seen_mentions = defaultdict(dict)

            for mention in mentions.values():
                left_sentence_index, mention_start_offset = start_offsets[mention["start"]]
                right_sentence_index, mention_end_offset = end_offsets[mention["end"]]

                assert (
                        left_sentence_index == right_sentence_index
                        and mention_start_offset <= mention_end_offset
                        and " ".join(
                    sentences[left_sentence_index]["tokens"][
                    mention_start_offset: mention_end_offset + 1
                    ]
                )
                        == mention["string"]
                )

                if (
                        mention_start_offset,
                        mention_end_offset,
                        mention["label"],
                ) in seen_mentions[left_sentence_index]:
                    seen_mention = seen_mentions[left_sentence_index][
                        mention_start_offset, mention_end_offset, mention["label"]
                    ]

                    assert not (
                            seen_mention["references"]
                            and mention["references"]
                            and seen_mention["references"] != mention["references"]
                    )

                    seen_mention["references"].update(mention["references"])
                else:
                    sentences[left_sentence_index]["mentions"].append(
                        {
                            "id": mention["id"],
                            "label": mention["label"],
                            "start": mention_start_offset,
                            "end": mention_end_offset,
                            "references": mention["references"],
                        }
                    )

                    seen_mentions[left_sentence_index][
                        mention_start_offset, mention_end_offset, mention["label"]
                    ] = mention

            docs[os.path.basename(filename)] = {"sentences": sentences}

        return docs
