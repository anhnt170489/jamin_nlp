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


class SAInstance(TextInstances):
    def __init__(self, id, text, label):
        super().__init__(text)
        self.id = id
        self.label = label
