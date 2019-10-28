import logging

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    TensorDataset,
)
from tqdm import tqdm

from core.common import *
from core.processor import InstanceProcessor

logger = logging.getLogger(__name__)


class Evaluator(object):
    @staticmethod
    def remove_ignore_labels(preds, label_ids, ignored_labels):
        final_preds = []
        final_label_ids = []
        for pred_t, label_id_t in zip(preds, label_ids):
            if not (pred_t == label_id_t and pred_t in ignored_labels):
                final_preds.append(pred_t)
                final_label_ids.append(label_id_t)

        return final_preds, final_label_ids

    @staticmethod
    def evaluate(instances, model, args, metrics=None, predict=False):
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

        data_ids = TensorDataset(torch.arange(len(instances)))

        sampler = SequentialSampler(data_ids)
        batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = batch_size
        data_loader = DataLoader(data_ids, sampler=sampler, batch_size=batch_size)

        if predict:
            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
                model = amp.initialize(model, opt_level=args.fp16_opt_level)

            # multi-gpu training (should be after apex fp16 initialization)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Distributed training (should be after apex fp16 initialization)
            if args.local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                  output_device=args.local_rank,
                                                                  find_unused_parameters=True)

        logger.info("***** Running evaluating *****")
        logger.info("  Num examples = %d", len(data_ids))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        model.zero_grad()

        preds = None
        golds = None

        for step, data_ids in enumerate(
                tqdm(data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        ):
            model.eval()
            with torch.no_grad():
                batch = [
                    instances[id]
                    for id in data_ids[0].tolist()
                ]
                batch = InstanceProcessor.pad_and_to_device(batch, args.device)

                outputs = model(batch)

                if args.ignored_labels:
                    batch_predict, batch_golds = Evaluator.remove_ignore_labels(outputs[PREDICT], outputs[GOLD],
                                                                                args.ignored_labels)
                else:
                    batch_predict = outputs[PREDICT]
                    batch_golds = outputs[GOLD]

                if not predict:

                    if preds is None:
                        preds = batch_predict
                        golds = batch_golds

                    else:
                        preds = np.append(batch_predict, preds, axis=0)
                        golds = np.append(batch_golds, golds, axis=0)

                    tmp_eval_loss = outputs[LOSS]
                    eval_loss += tmp_eval_loss.mean().item()
                else:
                    preds = [] if not preds else preds
                    preds.append(outputs[PREDICT])

            nb_eval_steps += 1

        if not predict:
            eval_loss = eval_loss / nb_eval_steps
            results = metrics.compute(preds, golds)
            results['eval_loss'] = eval_loss

            return results
        else:
            return preds
