from typing import Dict

import torch
from torch import Tensor

from core.common import *
from core.meta import *


class InstanceProcessor(object):

    @staticmethod
    def to_device(tensor, device=None, dtype=None, non_blocking=True, copy=False):
        return tensor.to(
            device=device, dtype=dtype, non_blocking=non_blocking, copy=copy
        )

    @staticmethod
    def pin_memory_and_to_device(batch, device, pin_memory=True):
        # https://discuss.pytorch.org/t/pin-memory-vs-sending-direct-to-gpu-from-dataset/33891/2
        if isinstance(batch, Tensor):
            if pin_memory:
                batch = batch.pin_memory()
            batch = InstanceProcessor.to_device(batch, device=device)
        elif isinstance(batch, Dict):
            for k, v in batch.items():
                if k != META_DATA:
                    batch[k] = InstanceProcessor.pin_memory_and_to_device(v, device, pin_memory)
        else:
            raise ValueError("Can't identify batch type")
        return batch

    @staticmethod
    def pad(instances, fix_padding_length=None):
        batch = {}
        for instance in instances:
            for k, v in instance.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        for k, instances in batch.items():
            if any(instances):
                try:
                    if isinstance(instances[0], SequenceInstance):
                        padding_length = fix_padding_length if fix_padding_length else max(
                            [len(instance) for instance in instances])
                        instances = [instance.pad(padding_length) for instance in instances]
                    if isinstance(instances[0], BertInstance):
                        batch[k] = {}
                        batch[k][BERT_INPUT_IDS] = torch.stack([instance.to_tensors()[0] for instance in instances],
                                                               dim=0)
                        batch[k][BERT_INPUT_MASKS] = torch.stack([instance.to_tensors()[1] for instance in instances],
                                                                 dim=0)
                        if isinstance(instances[0], BertSequenceInstance):
                            batch[k][BERT_SEGMENT_IDS] = torch.stack(
                                [instance.to_tensors()[2] for instance in instances], dim=0)
                        elif isinstance(instances[0], BertTokenInstance):
                            if all(instance.to_tensors()[2] is not None for instance in instances):
                                batch[k][BERT_TOKEN_LABELS] = torch.stack(
                                    [instance.to_tensors()[2] for instance in instances], dim=0)
                            batch[k][BERT_TOKEN_MASKS] = torch.stack(
                                [instance.to_tensors()[3] for instance in instances], dim=0)
                        elif isinstance(instances[0], BertQAInstance):
                            batch[k][BERT_P_MASKS] = torch.stack([instance.to_tensors()[2] for instance in instances],
                                                                 dim=0)
                            batch[k][BERT_SEGMENT_IDS] = torch.stack(
                                [instance.to_tensors()[3] for instance in instances], dim=0)

                    else:
                        batch[k] = torch.stack([instance.to_tensor() for instance in instances], dim=0)
                except:
                    if k != META_DATA:
                        import traceback
                        traceback.print_exc()
                        print("Can't padding", k)

        return batch


class InstanceBatchProcessor(object):
    def __init__(self, args, fix_padding_length=None):
        self.args = args
        self.fix_padding_length = fix_padding_length

    def collate(self, batch):
        return InstanceProcessor.pad(batch, fix_padding_length=self.fix_padding_length)
