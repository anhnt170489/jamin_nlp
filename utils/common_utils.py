import logging

from core.common import PRINT, BEST_STEP, STEP

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def log_eval_result(result, outfile=None):
    if BEST_STEP in result:
        logger.info("***** Best result at step {} *****".format(result[BEST_STEP]))
    elif STEP in result:
        logger.info("***** Result at step {} *****".format(result[STEP]))
    if PRINT in result:
        logger.info(result[PRINT])
    else:
        for key in sorted(result.keys()):
            if key != BEST_STEP:
                logger.info("  %s = %s", key, str(result[key]))

    if outfile:
        with open(outfile, "a+") as writer:
            if BEST_STEP in result:
                writer.write("***** Best result at step {} *****\n".format(result[BEST_STEP]))
            elif STEP in result:
                writer.write("***** Result at step {} *****\n".format(result[STEP]))
            if PRINT in result:
                writer.write(result[PRINT] + '\n')
            else:
                for key in sorted(result.keys()):
                    if key != BEST_STEP:
                        writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')


def read_text(filename, encoding="UTF-8"):
    with open(filename, "r", encoding=encoding) as f:
        return f.read()


def read_lines(filename, encoding="UTF-8"):
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def dump_json(data, out_file=None, ensure_ascii=False, indent=2):
    import json
    if out_file is None:
        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
    else:
        with open(out_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=ensure_ascii, indent=indent)


def collate(batch):
    from core.meta import JaminBatch
    return JaminBatch(batch)
