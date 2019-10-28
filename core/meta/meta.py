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


class BLUEInstance(object):
    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
