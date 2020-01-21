import argparse
import glob
import logging
import os
import random

import numpy as np
import torch

from core.eval import Evaluator
from core.eval import SpanClassificationMetrics
from core.reader import CGReader, SciERCReader
from core.training import Trainer
from libs import BertTokenizer, BertConfig, RobertaConfig

logger = logging.getLogger(__name__)
from core.models import BertNestedNER

from utils import cache_data, load_cached_data, handle_checkpoints, make_dirs

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)),
    ())

CG, SCIERC = 'cg', 'scierc'
CLASS_TYPES = {CG: (CGReader, 'M'),  # "S: Singly labeled, M: Multiply labeled"
               SCIERC: (SciERCReader, 'S'),
               }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Training parameter
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to predict on the test set.")
    parser.add_argument("--load_cached_data", action='store_true',
                        help="Load cached data instead of reading from beginning.")
    parser.add_argument("--cache_data", action='store_true',
                        help="Cache data after reading.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--model_to_predict", default=None, type=str,
                        help="model path used for prediction")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimizer",
                        type=str,
                        default='ADAM',
                        help="Optimizer to use. (ADAM, LAMB, RADAM)")
    parser.add_argument("--opt_level",
                        type=str,
                        default='O1',
                        help="opt_level offered by apex, using when training with fp16 configuration.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--eval_measure", default='f1', type=str,
                        help="Measure to evaluate.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--logging_eval_to_file", action='store_true',
                        help="Logging eval results to file eval.log in the output_dir")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument('--gpu', type=int, default=-1,
                        help="GPU id to use. -1 = using CPU")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    ## Model specific parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: bert, roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_span_width", default=8, type=int,
                        help="The maximum width for a span. Using for the models calculating on span")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_last_subword", action='store_true',
                        help="Set this flag if you choose the last subword to present the word. "
                             "If not, the first subword will be choosen")
    parser.add_argument("--use_all_subwords", action='store_true',
                        help="Set this flag if you choose all the subwords to present the word. ")
    # parser.add_argument("--label_type", default='S', type=str,
    #                     help="S: Singly labeled, M: Multiply labeled")
    parser.add_argument("--multi_label_threshold", default=0.5, type=float,
                        help="Threshold using in case of label_type = M")
    parser.add_argument("--corpus", default=None, type=str, required=True,
                        help="The data sets to train selected in the list: " + ", ".join(CLASS_TYPES.keys()))

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    make_dirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.gpu >= 0:
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        # ADD BY LONG
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
    args.device = device
    args.n_gpu = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(False), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    if args.max_seq_length <= 0:
        args.max_seq_length = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_length = min(args.max_seq_length, tokenizer.max_len_single_sentence)
    data_dir = args.data_dir

    logger.info("Reading data")
    args.corpus = args.corpus.lower()
    reader_class, label_type = CLASS_TYPES[args.corpus]
    args.label_type = label_type
    reader = reader_class(args)

    if args.do_train:
        train_instances = None
        if args.load_cached_data:
            train_instances = load_cached_data(data_dir, type='TRAIN')
        if not train_instances:
            train_instances = reader.get_train_examples(data_dir)
            if args.cache_data:
                cache_data(train_instances, type='TRAIN', cache_dir=data_dir, compress=None)

    if args.do_eval or (args.do_train and args.evaluate_during_training):
        dev_instances = None
        if args.load_cached_data:
            dev_instances = load_cached_data(data_dir, type='DEV')
        if not dev_instances:
            dev_instances = reader.get_dev_examples(data_dir)
            if args.cache_data:
                cache_data(dev_instances, type='DEV', cache_dir=data_dir)

    if args.do_predict:
        test_instances = None
        if args.load_cached_data:
            test_instances = load_cached_data(data_dir, type='TEST')
        if not test_instances:
            test_instances = reader.get_test_examples(data_dir)
            if args.cache_data:
                cache_data(test_instances, type='TEST', cache_dir=data_dir)

    args.labels = reader.get_labels()
    args.ignored_labels = None

    eval_labels = args.labels.copy()
    if args.corpus == 'scierc':
        eval_labels.pop(eval_labels.index('None'))
    metrics = SpanClassificationMetrics(eval_labels)

    logger.info("Preparing the model")

    model = BertNestedNER(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # Training
    if args.do_train:
        logger.info("Start training")
        if args.evaluate_during_training:
            global_step, tr_loss = Trainer.train(train_instances, model, args, eval_instances=dev_instances,
                                                 metrics=metrics)
        else:
            global_step, tr_loss = Trainer.train(train_instances, model, args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:
        # Remove old eval file:
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        command = 'rm ' + output_eval_file
        os.system(command)

        # Evaluation
        best_check_point = None
        best_result = None
        best_score = -float('inf')
        results = {}
        if args.local_rank in [-1, 0]:
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in
                    sorted(glob.glob(args.output_dir + '/**/*.pt', recursive=True)))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                handle_checkpoints(model=model,
                                   checkpoint_dir=checkpoint,
                                   params={
                                       'device': args.device
                                   },
                                   resume=True)
                result = Evaluator.evaluate(dev_instances, model, args, metrics=metrics)
                if args.eval_measure in result and result[args.eval_measure] > best_score:
                    best_score = result[args.eval_measure]
                    best_result = result
                    best_result['step'] = global_step
                    best_check_point = checkpoint
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

            # Write best result
            if best_result:
                with open(output_eval_file, "a+") as writer:
                    logger.info("***** Best result at step {} *****".format(best_result['step']))
                    writer.write("***** Best result at step {} *****\n".format(best_result['step']))
                    for key in sorted(best_result.keys()):
                        if key != 'step':
                            logger.info("  %s = %s", key, str(best_result[key]))
                            writer.write("%s = %s\n" % (key, str(best_result[key])))
                    writer.write('\n')


if __name__ == "__main__":
    main()
