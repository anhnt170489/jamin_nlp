import torch

from core.common import *
from core.meta import *


class InstanceProcessor(object):

    @staticmethod
    def to_device(tensor, device=None, dtype=None, non_blocking=True, copy=False):
        return tensor.to(
            device=device, dtype=dtype, non_blocking=non_blocking, copy=copy
        )

    @staticmethod
    def pin_memory_and_to_device(tensor, device, pin_memory=True):
        if pin_memory:
            tensor = tensor.pin_memory()
        return InstanceProcessor.to_device(tensor, device=device)

    @staticmethod
    def pad_and_to_device(instances, device, fix_padding_length=None, pin_memory=True):
        batch = {}
        for instance in instances:
            for k, v in instance.data.items():
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
                        batch[k][BERT_INPUT_IDS] = InstanceProcessor.pin_memory_and_to_device(
                            torch.stack([instance.to_tensors()[0] for instance in instances], dim=0),
                            device=device,
                            pin_memory=pin_memory)
                        batch[k][BERT_INPUT_MASKS] = InstanceProcessor.pin_memory_and_to_device(
                            torch.stack([instance.to_tensors()[1] for instance in instances], dim=0),
                            device=device,
                            pin_memory=pin_memory)
                        if len(instances[0].to_tensors()) > 2 and instances[0].to_tensors()[2] is not None:
                            if isinstance(instances[0], BertSequenceInstance):
                                batch[k][BERT_SEGMENT_IDS] = InstanceProcessor.pin_memory_and_to_device(
                                    torch.stack([instance.to_tensors()[2] for instance in instances], dim=0),
                                    device=device,
                                    pin_memory=pin_memory)
                            elif isinstance(instances[0], BertTokenInstance):
                                if all(instance.to_tensors()[2] is not None for instance in instances):
                                    batch[k][BERT_TOKEN_LABELS] = InstanceProcessor.pin_memory_and_to_device(
                                        torch.stack([instance.to_tensors()[2] for instance in instances], dim=0),
                                        device=device,
                                        pin_memory=pin_memory)
                                batch[k][BERT_TOKEN_MASKS] = InstanceProcessor.pin_memory_and_to_device(
                                    torch.stack([instance.to_tensors()[3] for instance in instances], dim=0),
                                    device=device,
                                    pin_memory=pin_memory)
                            elif isinstance(instances[0], BertQAInstance):
                                batch[k][BERT_P_MASKS] = InstanceProcessor.pin_memory_and_to_device(
                                    torch.stack([instance.to_tensors()[2] for instance in instances], dim=0),
                                    device=device,
                                    pin_memory=pin_memory)
                                batch[k][BERT_SEGMENT_IDS] = InstanceProcessor.pin_memory_and_to_device(
                                    torch.stack(
                                        [instance.to_tensors()[3] for instance in instances], dim=0),
                                    device=device,
                                    pin_memory=pin_memory)

                    else:
                        batch[k] = InstanceProcessor.pin_memory_and_to_device(
                            torch.stack([instance.to_tensor() for instance in instances], dim=0),
                            device=device,
                            pin_memory=pin_memory).to(device)
                except:
                    if k != META_DATA:
                        import traceback
                        traceback.print_exc()
                        print("Can't padding", k)

        return batch
