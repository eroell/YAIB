#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.ml_models.sklearn import LogisticRegression
from icu_benchmarks.models.dl_models.rnn import RNNet
from icu_benchmarks.models.dl_models.transformer import Transformer

import pandas as pd
import polars as pl
import logging
from icu_benchmarks.models.train import train_common
from icu_benchmarks.data.constants import DataSplit, DataSegment, VarType
from icu_benchmarks.data.split_process_data import make_single_split_pandas, make_single_split_polars

from icu_benchmarks.data.preprocessor import PandasClassificationPreprocessor, PolarsClassificationPreprocessor, PolarsRegressionPreprocessor, Preprocessor
# unwrapped to see the barebones (and train single models that I can set myself)
# execute_repeated_cv
# preprocess_data()
# train_common() 
mortality24_path = Path("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/demo_data/mortality24/")

from icu_benchmarks.custom_code.custom_toplevel_function import custom_toplevel_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
# -----------------------
# Model registry
# -----------------------
MODEL_REGISTRY = {
    "logreg": LogisticRegression,
    "rnnet": RNNet,
    "transformer": Transformer,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YAIB models with custom preprocessing and CV"
    )

    # -----------------------
    # Data & paths
    # -----------------------
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-dataset-name", type=str, required=True)
    parser.add_argument("--val-dataset-name", type=str, required=True)
    parser.add_argument("--test-dataset-name", type=str, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--log-suffix", type=str, default="run")

    # -----------------------
    # Model
    # -----------------------
    parser.add_argument(
        "--model",
        choices=MODEL_REGISTRY.keys(),
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--layer-dim",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--ff-hidden-mult",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )


    # -----------------------
    # Training
    # -----------------------
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")

    # -----------------------
    # CV
    # -----------------------
    parser.add_argument("--cv-repetitions", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    model_cls = MODEL_REGISTRY[args.model]

    dataset_names = {
        "train": args.train_dataset_name,
        "val": args.val_dataset_name,
        "test": args.test_dataset_name,
    }

    if args.model == "logreg":
        model_kwargs = {}
    elif args.model == "rnnet":
        model_kwargs = {
            "hidden_dim": args.hidden_dim,
            "layer_dim": args.layer_dim,
            "num_classes": args.num_classes,
        }
    elif args.model == "transformer":
        model_kwargs = {
            "hidden_dim": args.hidden_dim,
            "layer_dim": args.layer_dim,
            "num_classes": args.num_classes,
            "hidden": args.hidden,
            "heads": args.heads,
            "ff_hidden_mult": args.ff_hidden_mult,
            "depth": args.depth,
        }


    custom_toplevel_function(
        data_dir=args.data_dir,
        cv_repetitions=args.cv_repetitions,
        cv_folds=args.cv_folds,
        runmode=RunMode.classification,
        model=model_cls,
        dataset_names=dataset_names,
        model_kwargs=model_kwargs,
        cpu=args.cpu,
        batch_size=args.batch_size,
        use_wandb=args.use_wandb,
        epochs=args.epochs,
        log_dir=args.log_dir,
        log_dir_name_suffix=args.log_suffix,
    )
    logging.info(f"logged to {args.log_dir}/{args.log_suffix}")
    logging.info(f"Training complete")


if __name__ == "__main__":
    main()