import logging
import os

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from core.common import *
from core.eval import Evaluator
from core.meta import JaminDataset
from core.processor import InstanceProcessor
from libs import AdamW, get_linear_schedule_with_warmup
from libs.opt import Lamb, RAdam, Ranger
from utils import collate
from utils import log_eval_result, handle_checkpoints

logger = logging.getLogger(__name__)

ADAMW, RADAM, LAMB, RANGER = 'adamw', 'radam', 'lamb', 'ranger'


class Trainer(object):
    @staticmethod
    def log_and_save_model(args, eval_instances, model, metrics, global_step, best_score, curr_loss, logging_loss,
                           curr_lr=None):
        do_save_model = True
        # Log metrics
        new_best_score = best_score
        output_eval_log = args.output_eval_log

        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            result = Evaluator.evaluate(eval_instances, model, args, metrics=metrics)
            if args.eval_measure in result and result[args.eval_measure] > best_score:
                new_best_score = result[args.eval_measure]
                result[BEST_STEP] = global_step
                do_save_model = True
            else:
                result[STEP] = global_step
                do_save_model = False

            if curr_lr:
                logging.info('lr %f', curr_lr)
            logging.info('loss %f', (curr_loss - logging_loss) / args.logging_steps)
            logging.info("Step results")
            log_eval_result(result, output_eval_log)
        new_logging_loss = curr_loss

        if do_save_model:
            handle_checkpoints(model=model,
                               checkpoint_dir=os.path.join(args.output_dir, 'checkpoint-{}'.format(str(global_step))),
                               params={
                                   "filename": "checkpoint",
                                   "global_step": global_step},
                               num_saved=1)

        return new_best_score, new_logging_loss

    @staticmethod
    def train(train_instances, model, args, eval_instances=None, metrics=None):

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

        train_data = JaminDataset(data=train_instances)
        if args.local_rank == -1:
            sampler = RandomSampler(train_data)
        else:
            sampler = DistributedSampler(train_data)
        batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.train_batch_size = batch_size
        iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        data_loader = DataLoader(dataset=train_data, sampler=sampler, batch_size=batch_size, collate_fn=collate,
                                 pin_memory=True, num_workers=5)

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
        elif optimizer_type == RANGER:
            optimizer = Ranger(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if optimizer_type not in [RADAM, RANGER]:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=t_total)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        if args.local_rank != -1 > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
            logging.info('Finish wrapping model with DistributedDataParallel')

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                    )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        best_result = None
        best_score = -float('inf')

        if args.logging_eval_to_file:
            # Remove old eval.log file:
            output_eval_log = os.path.join(args.output_dir, "eval.log")
            command = 'rm ' + output_eval_log
            os.system(command)
        else:
            output_eval_log = None
        args.output_eval_log = output_eval_log

        for _ in iterator:
            epoch_iterator = tqdm(data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()

                batch_data = InstanceProcessor.pin_memory_and_to_device(batch.data, device=args.device, pin_memory=True)
                outputs = model(batch_data)

                loss = outputs[LOSS]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if optimizer_type not in [RADAM, RANGER]:
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # Evaluate during training if configured
                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        best_score, logging_loss = Trainer.log_and_save_model(args, eval_instances, model, metrics,
                                                                              global_step,
                                                                              best_score,
                                                                              tr_loss, logging_loss,
                                                                              scheduler.get_lr()[
                                                                                  0] if optimizer_type != RADAM else None)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                iterator.close()
                break

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps != 0:
            Trainer.log_and_save_model(args, eval_instances, model, metrics,
                                       global_step,
                                       best_score,
                                       tr_loss, logging_loss,
                                       scheduler.get_lr()[0] if optimizer_type != RADAM else None)

        return global_step, tr_loss / global_step
