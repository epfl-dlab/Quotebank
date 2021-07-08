import argparse
import glob
import logging
import os

import torch
from transformers import WEIGHTS_NAME, AutoTokenizer

from quobert.model import fit, evaluate, BertForQuotationAttribution
from quobert.utils import set_seed
from quobert.utils.data import ConcatParquetDataset, ParquetDataset

QUOTE_TOKEN = "[QUOTE]"
QUOTE_TARGET = "[TARGET_QUOTE]"

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ['bert-base-cased']",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--train_dir",
        default=None,
        type=str,
        help="The input training directory. Should contain (.gz.parquet) files",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=24,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-7,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=5000,
        type=int,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="If you want to train the model",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="If you want to evaluate a model on the validation set (model in --output_dir)",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="If you want to evaluate all checkpoints on the validation set (`do_eval` should be set)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--tb_path",
        type=str,
        default=None,
        help="Tensorboard logging dir if you want to overwrite `runs`.",
    )

    args = parser.parse_args()
    args.azure = False

    if args.fp16:
        try:
            import apex  # type: ignore

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    if args.do_train:
        # Load the dataset and train
        logger.info(f"Started loading the dataset from {args.train_dir}")
        if args.train_dir.endswith(".pt"):
            logger.info(f"Loading .pt file")
            train_dataset = torch.load(args.train_dir)
        else:
            files = glob.glob(os.path.join(args.train_dir, "**.gz.parquet"))
            train_dataset = ConcatParquetDataset([ParquetDataset(f) for f in files])

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer.add_tokens([QUOTE_TOKEN, QUOTE_TARGET])
        model = BertForQuotationAttribution.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)

        logger.info("Start the training function")
        fit(args, train_dataset, model)

        # Save the trained model and the tokenizer
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.do_eval:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        logger.info(f"Started loading the validation dataset from {args.train_dir}")
        files = glob.glob(os.path.join(args.train_dir, "val_dataset/**.gz.parquet"))
        validation_dataset = ConcatParquetDataset([ParquetDataset(f) for f in files])
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = BertForQuotationAttribution.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            evaluate(args, model, validation_dataset, no_save=True)
