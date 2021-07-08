import argparse
import glob
import logging
import os

import torch

from quobert.model import BertForQuotationAttribution, evaluate
from quobert.utils.data import ParquetDataset

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
        "--inference_dir",
        default=None,
        type=str,
        required=True,
        help="The input inference directory. Should contain (.gz.parquet) files",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=256,
        type=int,
        help="Batch size per GPU/CPU for Inference.",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model = BertForQuotationAttribution.from_pretrained(args.model_dir)
    model.to(args.device)

    logger.info(f"Started loading the dataset from {args.inference_dir}")
    files = sorted(glob.glob(os.path.join(args.inference_dir, "**.gz.parquet")))

    for i, f in enumerate(files):
        dataset = ParquetDataset(f)
        args.output_file = os.path.join(args.output_dir, f"results_{i:04d}.csv")
        evaluate(args, model, dataset, has_target=False)
