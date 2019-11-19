import collections
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class PredictWriter(object):
    @staticmethod
    def write(preds, output, *args):
        raise NotImplementedError()


class SQUADPredictWriter(PredictWriter):
    @staticmethod
    def get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    @staticmethod
    def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heuristic between
        # `pred_text` and `orig_text` to get a character-to-character alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        from libs.transformers.transformers.tokenization_bert import BasicTokenizer
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logger.info(
                    "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logger.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logger.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    @staticmethod
    def _compute_softmax(scores):
        import math
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    @staticmethod
    def write(preds, output, args):
        results = preds['results']
        contents = preds['contents']

        output_prediction_file = output['output_prediction_file']
        output_nbest_file = output['output_nbest_file']
        output_null_log_odds_file = output['output_null_log_odds_file']

        """Write final predictions to the json file and log-odds of null if needed."""
        logger.info("Writing predictions to: %s" % (output_prediction_file))
        logger.info("Writing nbest to: %s" % (output_nbest_file))

        qas_index_to_contents = collections.defaultdict(list)
        for content in contents:
            qas_index_to_contents[content.qas_id].append(content)

        unique_id_to_result = {}
        for result in results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["content_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()
        for qas_id, contents in qas_index_to_contents.items():
            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_content_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (content_index, content) in enumerate(contents):
                result = unique_id_to_result[content.unique_id]
                start_indexes = SQUADPredictWriter.get_best_indexes(result.start_logits, args.n_best_size)
                end_indexes = SQUADPredictWriter.get_best_indexes(result.end_logits, args.n_best_size)
                # if we could have irrelevant answers, get the min score of irrelevant
                if args.version_2_with_negative:
                    content_null_score = result.start_logits[0] + result.end_logits[0]
                    if content_null_score < score_null:
                        score_null = content_null_score
                        min_null_content_index = content_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(content.tokens):
                            continue
                        if end_index >= len(content.tokens):
                            continue
                        if start_index not in content.token_to_orig_map:
                            continue
                        if end_index not in content.token_to_orig_map:
                            continue
                        if not content.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > args.max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                content_index=content_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))

            if args.version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        content_index=min_null_content_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= args.n_best_size:
                    break
                content = contents[pred.content_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = content.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = content.token_to_orig_map[pred.start_index]
                    orig_doc_end = content.token_to_orig_map[pred.end_index]
                    orig_tokens = content.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = SQUADPredictWriter.get_final_text(tok_text, orig_text, args.do_lower_case,
                                                                   args.verbose_logging)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit))
            # if we didn't include the empty option in the n-best, include it
            if args.version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(
                        _NbestPrediction(
                            text="",
                            start_logit=null_start_logit,
                            end_logit=null_end_logit))

                # In very rare edge cases we could only have single null prediction.
                # So we just create a nonce prediction in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0,
                                 _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = SQUADPredictWriter._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            if not args.version_2_with_negative:
                all_predictions[content.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
                scores_diff_json[qas_id] = score_diff
                if score_diff > args.null_score_diff_threshold:
                    all_predictions[qas_id] = ""
                else:
                    all_predictions[qas_id] = best_non_null_entry.text
            all_nbest_json[qas_id] = nbest_json

        from utils import dump_json
        with open(output_prediction_file, "w") as writer:
            writer.write(dump_json(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(dump_json(all_nbest_json, indent=4) + "\n")

        if args.version_2_with_negative:
            with open(output_null_log_odds_file, "w") as writer:
                writer.write(dump_json(scores_diff_json, indent=4) + "\n")

        return all_predictions
