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


class GLUEInstance(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        import copy
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        import json
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
