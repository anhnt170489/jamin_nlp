import collections
import logging
import os

from tqdm import tqdm

from core.common import *
from core.meta import SQUADInstance, SQUADContent, BertInstance, BertQAInstance
from libs.transformers.transformers.tokenization_bert import whitespace_tokenize
from .json_data_reader import JsonDataReader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class SQUADReader(JsonDataReader):

    def __init__(self, args):
        self.args = args

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_json(os.path.join(data_dir, "train.json")), type='TRAIN')

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_json(os.path.join(data_dir, "dev.json")), type='DEV')

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_instances(
            self._read_json(os.path.join(data_dir, "test.json")), type='TEST')

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _create_text_instances(self, json_data, type):

        instances = []
        for json_obj in json_data:
            for json_obj_data in json_obj["data"]:
                for paragraph in json_obj_data["paragraphs"]:
                    paragraph_text = paragraph["context"]
                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True
                    for c in paragraph_text:
                        if self._is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)

                    for qa in paragraph["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                        is_impossible = False
                        if type == 'TRAIN':
                            if self.args.version_2_with_negative:
                                is_impossible = qa["is_impossible"]
                            if (len(qa["answers"]) != 1) and (not is_impossible):
                                raise ValueError(
                                    "For training, each question should have exactly 1 answer.")
                            if not is_impossible:
                                answer = qa["answers"][0]
                                orig_answer_text = answer["text"]
                                answer_offset = answer["answer_start"]
                                answer_length = len(orig_answer_text)
                                start_position = char_to_word_offset[answer_offset]
                                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                                # Only add answers where the text can be exactly recovered from the
                                # document. If this CAN'T happen it's likely due to weird Unicode
                                # stuff so we will just skip the example.
                                #
                                # Note that this means for training mode, every example is NOT
                                # guaranteed to be preserved.
                                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                                cleaned_answer_text = " ".join(
                                    whitespace_tokenize(orig_answer_text))
                                if actual_text.find(cleaned_answer_text) == -1:
                                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                                   actual_text, cleaned_answer_text)
                                    continue
                            else:
                                start_position = -1
                                end_position = -1
                                orig_answer_text = ""

                        instance = SQUADInstance(
                            qas_id=qas_id,
                            question_text=question_text,
                            doc_tokens=doc_tokens,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            is_impossible=is_impossible)
                        instances.append(instance)

        return instances

    def _create_instances(self, json_data, type):
        """Creates instances for the training and dev sets."""
        logger.info("Reading text instances")
        text_instances = self._create_text_instances(json_data, type)

        logger.info("Convert text instances to model instances")

        cls_token_at_end = bool(self.args.model_type in ['xlnet'])
        sequence_a_is_doc = True if self.args.model_type in ['xlnet'] else False

        unique_id = 1000000000
        # cnt_pos, cnt_neg = 0, 0
        # max_N, max_M = 1024, 1024
        # f = np.zeros((max_N, max_M), dtype=np.float32)
        tokenizer = BertInstance.get_tokenizer(args=self.args)

        intances = []

        for (index, instance) in enumerate(tqdm(text_instances)):
            query_tokens = tokenizer.tokenize(instance.question_text)

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(instance.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if type == 'TRAIN' and instance.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if type == 'TRAIN' and not instance.is_impossible:
                tok_start_position = orig_to_tok_index[instance.start_position]
                if instance.end_position < len(instance.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[instance.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    instance.orig_answer_text)

            # Roberta: -4 accounts for [CLS], [SEP][SEP] and [SEP]
            # Bert: -3 accounts for [CLS], [SEP] and [SEP]
            if self.args.model_type == 'roberta':
                max_tokens_for_doc = self.args.max_seq_length - len(query_tokens) - 4
            else:
                max_tokens_for_doc = self.args.max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.args.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                token_to_orig_map = {}
                token_is_max_context = {}

                len_tokens = 0
                # CLS token at the beginning
                if not cls_token_at_end:
                    len_tokens += 1

                # XLNet: P SEP Q SEP CLS
                # Roberta: CLS Q SEP SEP P SEP
                # Others: CLS Q SEP P SEP
                if not sequence_a_is_doc:
                    # Query
                    len_tokens += len(query_tokens)

                    # SEP token
                    if self.args.model_type == 'roberta':
                        len_tokens += 2
                    else:
                        len_tokens += 1

                # Paragraph
                doc_tokens = []
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len_tokens] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                                split_token_index)
                    token_is_max_context[len_tokens] = is_max_context
                    doc_tokens.append(all_doc_tokens[split_token_index])
                    len_tokens += 1

                paragraph_len = doc_span.length

                tokens = BertQAInstance(query_tokens=query_tokens,
                                        doc_tokens=doc_tokens,
                                        args=self.args)
                cls_index = tokens.cls_index

                span_is_impossible = instance.is_impossible
                start_position = None
                end_position = None
                if type == 'TRAIN' and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        span_is_impossible = True
                    else:
                        if sequence_a_is_doc:
                            doc_offset = 0
                        else:
                            doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if type == 'TRAIN' and span_is_impossible:
                    start_position = cls_index
                    end_position = cls_index

                # Debug first 20 samples
                if index < 20:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens.tokens))
                    logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in tokens.input_ids]))
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in tokens.input_mask]))
                    logger.info(
                        "p_mask: %s" % " ".join([str(x) for x in tokens.p_mask]))
                    if type == 'TRAIN' and span_is_impossible:
                        logger.info("impossible example")
                    if type == 'TRAIN' and not span_is_impossible:
                        answer_text = " ".join(tokens.tokens[start_position:(end_position + 1)])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info(
                            "answer: %s" % (answer_text))

                metadata = SQUADContent(unique_id=unique_id,
                                        example_index=index,
                                        doc_span_index=doc_span_index,
                                        tokens=tokens.tokens,
                                        token_to_orig_map=token_to_orig_map,
                                        token_is_max_context=token_is_max_context,
                                        cls_index=cls_index,
                                        paragraph_len=paragraph_len,
                                        start_position=start_position,
                                        end_position=end_position,
                                        is_impossible=span_is_impossible,
                                        doc_tokens=instance.doc_tokens,
                                        qas_id=instance.qas_id)

                intances.append({
                    'tokens': tokens,
                    META_DATA: metadata
                })

                unique_id += 1

        return intances
