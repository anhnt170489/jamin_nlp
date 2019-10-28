import logging
import os

import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def norm_path(*paths):
    return os.path.relpath(os.path.normpath(os.path.join(os.getcwd(), *paths)))


def make_dirs(*paths):
    os.makedirs(norm_path(*paths), exist_ok=True)


def cache_data(data, cache_dir, type=None, compress=9, protocol=None, cache_size=None):
    name_by_type = 'data' if type is None else (
        'train' if type == 'TRAIN' else ('dev' if type == 'DEV' else 'test'))
    cached_features_file = os.path.join(cache_dir, '{}.cached'.format(name_by_type))
    logger.info("Caching data to %s", cached_features_file)
    make_dirs(os.path.dirname(cached_features_file))
    joblib.dump(
        value=data,
        filename=norm_path(cached_features_file),
        compress=compress,
        protocol=protocol,
        cache_size=cache_size,
    )


def load_cached_data(cache_dir, type=None, mmap_mode=None):
    name_by_type = 'data' if type is None else (
        'train' if type == 'TRAIN' else ('dev' if type == 'DEV' else 'test'))
    cached_features_file = os.path.join(cache_dir, '{}.cached'.format(name_by_type))
    if not os.path.exists(cached_features_file):
        logging.info("Cached file doesn't exist")
        return None
    logger.info("Loading data from cached file: %s", cached_features_file)
    return joblib.load(filename=norm_path(cached_features_file), mmap_mode=mmap_mode)
