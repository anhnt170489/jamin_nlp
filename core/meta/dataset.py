import collections
from typing import List

from torch.utils.data import (
    Dataset
)

from core.processor import InstanceProcessor


class JaminDataset(Dataset):
    """Dataset instance for training/evaluating"""

    _DSArgs = collections.namedtuple(
        "DSArgs", ["fix_padding_length"])

    _DSItems = collections.namedtuple(
        "DSItems", ["data", "args"])

    def __init__(
            self, data, fix_padding_length=None
    ):
        self.data = data
        self.args = JaminDataset._DSArgs(fix_padding_length=fix_padding_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return JaminDataset._DSItems(data=self.data[index], args=self.args)


class JaminBatch(object):
    def __init__(self, batch: List[JaminDataset]):
        self.data = InstanceProcessor.pad(batch, fix_padding_length=batch[0].args.fix_padding_length)
