import argparse
import glob
import logging
import os

import torch

from quobert.model import BertForQuotationAttribution, evaluate
from quobert.utils.data import ConcatParquetDataset, ParquetDataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model results will be written.",
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        required=True,
        help="The input test directory. Should contain (.gz.parquet) files",
    )
    parser.add_argument(
        "--output_speakers_only",
        action="store_true",
        help="If set, only output the top1 speakers instead of the probabilities associated",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Started loading the dataset from {args.test_dir}")
    files = glob.glob(os.path.join(args.test_dir, "**.gz.parquet"))
    datasets = [ParquetDataset(f) for f in files]
    concat_dataset = ConcatParquetDataset(datasets)

    model = BertForQuotationAttribution.from_pretrained(args.model_dir)
    model.to(args.device)
    args.output_file = os.path.join(args.output_dir, f"results.csv")
    evaluate(args, model, concat_dataset, output_proba=not args.output_speakers_only)
    # logger.info(f"EM: {result * 100:.2f}%")
