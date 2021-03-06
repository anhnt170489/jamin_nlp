import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class JsonDataReader(object):

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

    @classmethod
    def _read_json(cls, json_file):
        """Reads a json-based file."""
        import json

        with open(json_file, "r", encoding='utf-8') as json_file:
            try:
                json_data = [json.load(json_file)]
            except:
                json_data = None

        if not json_data:
            with open(json_file, "r", encoding='utf-8') as json_file:
                try:
                    json_data = [json.loads(json_line) for json_line in json_file]
                except:
                    import traceback
                    traceback.print_exc()

        return json_data
