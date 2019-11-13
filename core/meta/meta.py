class TextInstances(object):
    """A common text instance"""

    def __init__(self, text):
        self.text = text


class NERInstance(TextInstances):
    """ An Instance used for NER task"""

    def __init__(self, text, entities):
        super().__init__(text)
        self.entities = entities


class QAInstance(object):
    """ An Instance used for QA task"""

    def __init__(self, question, answer, label):
        super().__init__()
        self.question = question
        self.answer = answer
        self.label = label


class SQUADInstance(object):
    """
        A single training/test example for the Squad dataset.
        For examples without an answer, the start and end position are -1.
        """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class SQUADContent(object):

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 cls_index,
                 paragraph_len,
                 start_position,
                 end_position,
                 is_impossible,
                 doc_tokens,
                 qas_id, ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.cls_index = cls_index
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.doc_tokens = doc_tokens
        self.qas_id = qas_id


class SQUADResult(object):

    def __init__(self, unique_id, start_logits, end_logits):
        super().__init__()
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits
