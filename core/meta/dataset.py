import collections
from typing import List

from torch.utils.data import (
    Dataset
)

from core.processor import InstanceProcessor


class JaminDataset(Dataset):
    """Dataset instance for training/evaluating"""

    _DSArgs = collections.namedtuple(
        "DSArgs", ["device", "fix_padding_length", "pin_memory"])

    _DSItems = collections.namedtuple(
        "DSItems", ["data", "args"])

    def __init__(
            self, data, device, fix_padding_length=None, pin_memory=True
    ):
        self.data = data
        self.args = JaminDataset._DSArgs(device=device, fix_padding_length=fix_padding_length, pin_memory=pin_memory)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # https://discuss.pytorch.org/t/pin-memory-vs-sending-direct-to-gpu-from-dataset/33891/2
        return JaminDataset._DSItems(data=self.data[index], args=self.args)


class JaminBatch(object):
    def __init__(self, batch: List[JaminDataset]):
        self.data = InstanceProcessor.pad_and_to_device(batch,
                                                        device=batch[0].args.device,
                                                        fix_padding_length=batch[0].args.fix_padding_length,
                                                        pin_memory=batch[0].args.pin_memory
                                                        )
