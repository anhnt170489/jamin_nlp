import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def log_result(result):
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))


def read_text(filename, encoding="UTF-8"):
    with open(filename, "r", encoding=encoding) as f:
        return f.read()


def read_lines(filename, encoding="UTF-8"):
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\r\n\v")
