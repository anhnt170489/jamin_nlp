import torch

from core.common import *
from core.meta import *


class InstanceProcessor(object):

    @staticmethod
    def pad_and_to_device(instances, device, fix_padding_length=None):
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
                                                               dim=0).to(device)
                        batch[k][BERT_INPUT_MASKS] = torch.stack([instance.to_tensors()[1] for instance in instances],
                                                                 dim=0).to(device)
                        if len(instances[0].to_tensors()) > 2 and instances[0].to_tensors()[2] is not None:
                            if isinstance(instances[0], BertTokenInstance):
                                batch[k][BERT_TOKEN_LABELS] = torch.stack(
                                    [instance.to_tensors()[2] for instance in instances], dim=0).to(device)
                            elif isinstance(instances[0], BertQAInstance):
                                batch[k][BERT_P_MASKS] = torch.stack(
                                    [instance.to_tensors()[2] for instance in instances], dim=0).to(device)
                                batch[k][BERT_SEGMENT_IDS] = torch.stack(
                                    [instance.to_tensors()[3] for instance in instances], dim=0).to(device)

                    else:
                        batch[k] = torch.stack([instance.to_tensor() for instance in instances], dim=0).to(device)
                except:
                    if k != META_DATA:
                        import traceback
                        traceback.print_exc()
                        print("Can't padding", k)

        return batch
