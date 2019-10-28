import logging

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    TensorDataset,
)
from tqdm import tqdm, trange

from core.common import *
from core.eval import Evaluator
from core.processor import InstanceProcessor
from libs.opt import Lamb, RAdam
from libs.transformers import AdamW, WarmupLinearSchedule
from utils import save_model, log_result

logger = logging.getLogger(__name__)

ADAMW, RADAM, LAMB = 'adamw', 'radam', 'lamb'


class Trainer(object):
    @staticmethod
    def train(train_instances, model, args, eval_instances=None, metrics=None):

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

        data_ids = TensorDataset(torch.arange(len(train_instances)))
        sampler = RandomSampler(data_ids)
        batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.train_batch_size = batch_size
        iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        data_loader = DataLoader(data_ids, sampler=sampler, batch_size=batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(data_loader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(data_loader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer_type = args.optimizer.lower()

        if optimizer_type == LAMB:
            optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif optimizer_type == RADAM:
            optimizer = RAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if optimizer_type != RADAM:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(data_ids))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps
                    )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        best_result = None
        best_score = -float('inf')

        for _ in iterator:
            epoch_iterator = tqdm(data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, data_ids in enumerate(epoch_iterator):
                model.train()
                batch = [
                    train_instances[id]
                    for id in data_ids[0].tolist()
                ]
                batch = InstanceProcessor.pad_and_to_device(batch, args.device)

                outputs = model(batch)

                loss = outputs[LOSS]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if optimizer_type != RADAM:
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # Evaluate during training if configured
                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            result = Evaluator.evaluate(eval_instances, model, args, metrics=metrics)
                            if args.eval_measure in result and result[args.eval_measure] > best_score:
                                best_score = result[args.eval_measure]
                                best_result = result
                            if optimizer_type != RADAM:
                                logging.info('lr %f', scheduler.get_lr()[0])
                            logging.info('loss %f', (tr_loss - logging_loss) / args.logging_steps)
                            logging.info("Step results")
                            log_result(result)
                            if best_result:
                                logging.info("Best results")
                                log_result(best_result)
                        logging_loss = tr_loss

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_model(args, model, global_step)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                iterator.close()
                break

        if args.save_steps > 0 and global_step % args.save_steps != 0:
            save_model(args, model, global_step)

        return global_step, tr_loss / global_step
