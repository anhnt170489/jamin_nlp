import argparse
import logging
import os
import random

import numpy as np
import torch

from core.common import *
from core.eval import SQUADMetrics, SQUADPredictWriter
from core.meta import SQUADResult
from core.models import BertQuestionAnswering
from core.reader import SQUADReader
from core.training import Trainer
from libs.transformers import BertTokenizer
from utils import log_eval_result

logger = logging.getLogger(__name__)

from utils import cache_data, load_cached_data, make_dirs, handle_checkpoints

from core.eval import Evaluator

import glob

from libs.transformers import BertConfig, XLNetConfig, XLMConfig

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())


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
                        help="Model type selected in the list: bert,roberta")
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

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_last_subword", action='store_true',
                        help="Set this flag if you choose the last subword to present the word. "
                             "If not, the first subword will be choosen")
    parser.add_argument("--use_all_subwords", action='store_true',
                        help="Set this flag if you choose all the subwords to present the word. ")
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument("--pretrained_qa_model", default=None, type=str,
                        help="Path to pre-trained qa model ")

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
                   # args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
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
    reader = SQUADReader(args)
    args.labels = reader.get_labels()
    args.ignored_labels = None

    if args.do_train:
        train_instances = None
        if args.load_cached_data:
            train_instances = load_cached_data(data_dir, type='TRAIN')
        if not train_instances:
            train_instances = reader.get_train_examples(data_dir)
            if args.cache_data:
                cache_data(train_instances, type='TRAIN', cache_dir=data_dir, compress=None)

    if args.do_eval or (args.do_train and args.evaluate_during_training):
        args.predict_file = os.path.join(data_dir, "dev.json")
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

    logger.info("Preparing the model")
    model = BertQuestionAnswering(args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    if args.pretrained_qa_model:
        handle_checkpoints(model=model,
                           checkpoint_dir=args.pretrained_qa_model,
                           params={
                               'device': device
                           },
                           resume=True)

    # Training
    if args.do_train:
        logger.info("Start training")
        if args.evaluate_during_training:
            metrics = SQUADMetrics(args)
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

        # Prepare metrics
        metrics = SQUADMetrics(args)

        # Evaluation
        best_check_point = None
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
                                       'device': device
                                   },
                                   resume=True)
                result = Evaluator.evaluate(dev_instances, model, args, metrics=metrics)
                if args.eval_measure in result and result[args.eval_measure] > best_score:
                    best_score = result[args.eval_measure]
                    result[BEST_STEP] = global_step
                    best_check_point = checkpoint
                else:
                    result[STEP] = global_step
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

                # Write result
                log_eval_result(result, output_eval_file)

    if args.do_predict:
        checkpoint = args.model_to_predict
        if not checkpoint:
            if args.do_eval and best_check_point is not None:
                logger.info("There's no model to predict, the best checkpoint when evaluating will be used")
                checkpoint = best_check_point

        assert checkpoint, ('No model to predict')
        logger.info("Predict the following checkpoints: %s", checkpoint)
        handle_checkpoints(model=model,
                           checkpoint_dir=checkpoint,
                           params={
                               'device': device
                           },
                           resume=True)
        preds = Evaluator.evaluate(test_instances, model, args, predict=True)

        all_results = []
        all_contents = []
        for batch_predicts in preds:
            for unique_id, start_logits, end_logits, contents in zip(batch_predicts[0], batch_predicts[1],
                                                                     batch_predicts[2], batch_predicts[3]):
                result = SQUADResult(unique_id=unique_id,
                                     start_logits=start_logits,
                                     end_logits=end_logits)
                all_results.append(result)
                all_contents.append(contents)

        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
        else:
            output_null_log_odds_file = None
        SQUADPredictWriter.write({"results": all_results, "contents": all_contents},
                                 {"output_prediction_file": output_prediction_file,
                                  "output_nbest_file": output_nbest_file,
                                  "output_null_log_odds_file": output_null_log_odds_file}
                                 , args)


if __name__ == "__main__":
    main()
