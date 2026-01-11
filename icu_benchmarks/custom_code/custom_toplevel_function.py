import pandas as pd
import polars as pl
from pathlib import Path

from icu_benchmarks.models.train import train_common
from icu_benchmarks.data.constants import DataSplit, DataSegment, VarType
from icu_benchmarks.data.split_process_data import make_single_split_pandas, make_single_split_polars
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.ml_models.sklearn import LogisticRegression
from icu_benchmarks.models.dl_models.rnn import RNNet
from icu_benchmarks.models.dl_models.transformer import Transformer

from icu_benchmarks.data.preprocessor import PandasClassificationPreprocessor, PolarsClassificationPreprocessor, PolarsRegressionPreprocessor, Preprocessor
# unwrapped to see the barebones (and train single models that I can set myself)
# execute_repeated_cv
# preprocess_data()
# train_common() 
mortality24_path = Path("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/demo_data/mortality24/")
import wandb

import pickle
# from within preprocess_data.py
vars = pickle.load(open("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/vars.pkl", "rb"))

def custom_toplevel_function(
    data_dir = mortality24_path / "mimic_demo",
    cv_repetitions = 2,
    cv_repetitions_to_train = None,
    cv_folds = 2,
    cv_folds_to_train = None,
    runmode = RunMode.classification,
    model = LogisticRegression,
    dataset_names = {"train": "demo_data/mortality24/mimic_demo", "val": "demo_data/mortality24/mimic_demo", "test": "demo_data/mortality24/mimic_demo"},
    model_kwargs = {},
    cpu=False,
    batch_size=512,
    use_wandb=False,
    epochs=100,
    log_dir = Path("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/custom_logs"),
    log_dir_name_suffix = "run_name",
    log_every_n_steps=None,
    ):

    log_dir = log_dir / log_dir_name_suffix

    import pickle
    # from within preprocess_data.py
    vars = pickle.load(open("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/vars.pkl", "rb"))

    import gin
    import torch
    from pathlib import Path

    CONFIG_PATH = Path("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/configs/tasks/BinaryClassification.gin")
    gin.add_config_file_search_path("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB")
    gin.parse_config_file(CONFIG_PATH)

    #-------------------------------------#

    import json
    import logging
    from datetime import datetime
    from pathlib import Path
    from typing import Optional


    from pytorch_lightning import seed_everything

    from icu_benchmarks.constants import RunMode
    from icu_benchmarks.data.split_process_data import preprocess_data
    from icu_benchmarks.models.train import train_common
    from icu_benchmarks.models.utils import JsonResultLoggingEncoder
    from icu_benchmarks.run_utils import aggregate_results, log_full_line
    from icu_benchmarks.wandb_utils import wandb_log
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

    #-------------------------------------#

    # data_dir = mortality24_path / "mimic_demo" 
    # log_dir = Path("/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/custom_logs")
    seed = 42
    eval_only = False
    train_size = None
    load_weights = False
    source_dir = Path("")
    # cv_repetitions = 2
    # cv_repetitions_to_train = None
    # cv_folds = 2
    # cv_folds_to_train = None
    reproducible = True
    debug = False
    generate_cache = False
    load_cache = False
    test_on = "test"
    # mode = RunMode.classification
    pretrained_imputation_model = None
    # cpu = False
    verbose = False
    # wandb = False
    complete_train = False

    #-------------------------------------#

    if not cv_repetitions_to_train:
        cv_repetitions_to_train = cv_repetitions
    if not cv_folds_to_train:
        cv_folds_to_train = cv_folds
    agg_loss = 0
    # seed_everything(seed, reproducible)
    if complete_train:
        logging.info("Will train full model without cross validation.")
        cv_repetitions_to_train = 1
        cv_folds_to_train = 1

    else:
        logging.info(f"Starting nested CV with {cv_repetitions_to_train} repetitions of {cv_folds_to_train} folds.")
    # Train model for each repetition (a manner of splitting the folds)
    for repetition in range(1):
        # Train model for each fold configuration (i.e, one fold is test fold and the rest are train/val folds)
        for fold_index in range(1):
            repetition_fold_dir = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}"
            repetition_fold_dir.mkdir(parents=True, exist_ok=True)

            start_time = datetime.now()
            #-------------------------------------#
            #######################################
            #-------------------------------------#
            # data = preprocess_data(
            #     data_dir,
            #     seed=seed,
            #     debug=debug,
            #     load_cache=load_cache,
            #     generate_cache=generate_cache,
            #     cv_repetitions=cv_repetitions,
            #     repetition_index=repetition,
            #     train_size=train_size,
            #     cv_folds=cv_folds,
            #     fold_index=fold_index,
            #     pretrained_imputation_model=pretrained_imputation_model,
            #     runmode=mode,
            #     complete_train=complete_train,
            # )
            # data_dir
            import hashlib
            import os
            from timeit import default_timer as timer
            from icu_benchmarks.data.split_process_data import check_sanitize_data

            file_names = { # correct?
                DataSegment.static: "sta.parquet",
                DataSegment.dynamic: "dyn.parquet",
                DataSegment.outcome: "outc.parquet",
            }
            if runmode == RunMode.classification:
                preprocessor = PolarsClassificationPreprocessor
            else:
                preprocessor = PolarsRegressionPreprocessor
            use_static = True,
            vars = vars
            modality_mapping = None
            selected_modalities = None
            seed = 42
            debug = False
            repetition_index = repetition
            train_size = None
            load_cache= False
            generate_cache= False
            pretrained_imputation_model = None
            complete_train= False
            # runmode = RunMode.classification
            label = None
            required_var_types = None
            required_segments = None

            if modality_mapping is None:
                modality_mapping = {}
            if selected_modalities is None:
                selected_modalities = ["all"]
            if required_var_types is None:
                required_var_types = ["GROUP", "SEQUENCE", "LABEL"]
            if required_segments is None:
                required_segments = [DataSegment.static, DataSegment.dynamic, DataSegment.outcome]

            # check_required_keys(vars, required_var_types)
            #check_required_keys(file_names, required_segments)

            if not use_static:
                file_names.pop(DataSegment.static)
                vars.pop(DataSegment.static)

            if isinstance(vars[VarType.label], list) and len(vars[VarType.label]) > 1:
                if label is not None:
                    vars[VarType.label] = [label]
                else:
                    logging.debug(f"Multiple labels found and no value provided. Using first label: {vars[VarType.label]}")
                    vars[VarType.label] = vars[VarType.label][0]
                logging.info(f"Using label: {vars[VarType.label]}")

            if not vars[VarType.label]:
                raise ValueError("No label selected after filtering.")

            dumped_file_names = json.dumps(file_names, sort_keys=True)
            dumped_vars = json.dumps(vars, sort_keys=True)

            logging.info(f"Using preprocessor: {preprocessor.__name__}")

            cat_clinical_notes = modality_mapping.get("cat_clinical_notes")
            cat_med_embeddings_map = modality_mapping.get("cat_med_embeddings_map")
            if cat_clinical_notes is not None and cat_med_embeddings_map is not None:
                vars_to_exclude = cat_clinical_notes + cat_med_embeddings_map
            else:
                vars_to_exclude = None

            cache_dir = data_dir / "cache"
            cache_filename = f"s_{seed}_r_{repetition_index}_f_{fold_index}_t_{train_size}_d_{debug}"
            preprocessor_instance: Preprocessor = preprocessor(
                use_static_features=use_static,
                save_cache=data_dir / "preproc" / (cache_filename + "_recipe") if generate_cache else None,
                vars_to_exclude=vars_to_exclude,
            )
            if isinstance(preprocessor_instance, PandasClassificationPreprocessor):
                preprocessor_instance.set_imputation_model(pretrained_imputation_model)

            hash_config = hashlib.md5(f"{preprocessor_instance.to_cache_string()}{dumped_file_names}{dumped_vars}".encode("utf-8"))
            cache_filename += f"_{hash_config.hexdigest()}"
            cache_file = cache_dir / cache_filename

            if load_cache:
                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        logging.info(f"Loading cached data from {cache_file}.")
                        # return pickle.load(f)
                else:
                    logging.info(f"No cached data found in {cache_file}, loading raw features.")

            # Read parquet files into dataframes and remove the parquet file from memory
            logging.info(f"Loading data from directory {data_dir.absolute()}")
            data: dict[str, pl.DataFrame] = {
                f: pl.read_parquet(data_dir / file_names[f]) for f in file_names.keys() if os.path.exists(data_dir / file_names[f])
            }

            logging.info(f"Loaded data: {list(data.keys())}")
            sanitized_data = check_sanitize_data(data, vars)

            if DataSegment.dynamic not in sanitized_data.keys():
                logging.warning("No dynamic data found, using only static data.")

            logging.debug(f"Modality mapping: {modality_mapping}")
            if len(modality_mapping) > 0:
                # Optional modality selection
                if selected_modalities not in [None, "all", ["all"]]:
                    data, vars = modality_selection(sanitized_data, modality_mapping, selected_modalities, vars)
                else:
                    logging.info("Selecting all modalities.")

            # Generate the splits
            logging.info("Generating splits.")
            if not complete_train:
                sanitized_data = make_single_split_polars(
                    sanitized_data,
                    vars,
                    cv_repetitions,
                    repetition_index,
                    cv_folds,
                    fold_index,
                    train_size=train_size,
                    seed=seed,
                    debug=debug,
                    runmode=runmode,
                )
            else:
                # If full train is set, we use all data for training/validation
                sanitized_data = make_train_val_polars(data, vars, train_size=None, seed=seed, debug=debug, runmode=runmode)

            # Apply preprocessing
            start = timer()
            sanitized_data = preprocessor_instance.apply(sanitized_data, vars)
            end = timer()
            logging.info(f"Preprocessing took {end - start:.2f} seconds.")
            logging.info(f"Checking for NaNs and nulls in {data.keys()}.")
            for _dict in sanitized_data.values():
                for key, val in _dict.items():
                    logging.debug(f"Data type: {key}")
                    logging.debug("Is NaN:")
                    sel = _dict[key].select(pl.selectors.numeric().is_nan().max())
                    logging.debug(sel.select(col.name for col in sel if col.item(0)))
                    logging.debug("Has nulls:")
                    sel = _dict[key].select(pl.all().has_nulls())
                    logging.debug(sel.select(col.name for col in sel if col.item(0)))
                    _dict[key] = val.fill_null(strategy="zero")
                    _dict[key] = val.fill_nan(0)
                    logging.debug("Dropping columns with nulls")
                    sel = _dict[key].select(pl.all().has_nulls())
                    logging.debug(sel.select(col.name for col in sel if col.item(0)))

            # Generate cache
            if generate_cache:
                caching(cache_dir, cache_file, sanitized_data, load_cache)
            else:
                logging.info("Cache will not be saved.")

            logging.info("Finished preprocessing.")

            data = sanitized_data
            #-------------------------------------#
            #######################################
            #-------------------------------------#

            preprocess_time = datetime.now() - start_time
            start_time = datetime.now()
            #-------------------------------------#
            #######################################
            #-------------------------------------#
            # agg_loss += train_common(
            #     data,
            #     log_dir=repetition_fold_dir,
            #     eval_only=eval_only,
            #     load_weights=load_weights,
            #     source_dir=source_dir,
            #     reproducible=reproducible,
            #     test_on=test_on,
            #     mode=mode,
            #     cpu=cpu,
            #     verbose=verbose,
            #     use_wandb=wandb,
            #     train_only=complete_train,
            # )

            from pytorch_lightning import Trainer
            from torch.optim import Adam
            from icu_benchmarks.data.loader import ImputationPandasDataset, PredictionPandasDataset, PredictionPolarsDataset
            from torch.utils.data import DataLoader
            from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
            from icu_benchmarks.models.utils import JSONMetricsLogger, save_config_file

            # data: dict[str, dict[str, pl.DataFrame]],
            # log_dir: Path,
            eval_only = False
            load_weights = False
            source_dir = Path("")
            reproducible = True
            # mode = RunMode.classification 
            model = model # LogisticRegression #gin.REQUIRED # TODO: replace, model.LOGISTICREGRESSION or so?
            weight = ""
            optimizer = Adam
            precision = 32
            # batch_size = 1
            # epochs = 100
            patience = 20 # Early stopping checks at end of every validation epoch. This here means 20 such checks. see here: https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
            min_delta = 1e-5 # for val_loss
            test_on = DataSplit.test
            dataset_names = dataset_names #{"train": "demo_data/mortality24/mimic_demo", "val": "demo_data/mortality24/mimic_demo", "test": "demo_data/mortality24/mimic_demo"} # from inside train_common with debugger
            # use_wandb = False
            # cpu = False
            verbose = False
            ram_cache = False
            pl_model = True
            train_only = False
            num_workers = 1
            polars = True
            persistent_workers = False

            if dataset_names is None:
                dataset_names = {}

            logging.info(f"Training model: {model.__name__}.")
            # TODO: add support for polars versions of datasets
            dataset_classes: dict = {
                RunMode.imputation: ImputationPandasDataset,
                RunMode.classification: PredictionPolarsDataset if polars else PredictionPandasDataset,
                RunMode.regression: PredictionPolarsDataset if polars else PredictionPandasDataset,
            }
            dataset_class = dataset_classes[runmode]

            logging.info(f"Using dataset class: {dataset_class.__name__}.")
            logging.info(f"Logging to directory: {log_dir}.")
            save_config_file(log_dir)  # We save the operative config before and also after training
            train_dataset = dataset_class(data, split=DataSplit.train, ram_cache=ram_cache, name=dataset_names["train"])
            val_dataset = dataset_class(data, split=DataSplit.val, ram_cache=ram_cache, name=dataset_names["val"])
            # train_dataset, val_dataset = assure_minimum_length(train_dataset), assure_minimum_length(val_dataset)
            batch_size = min(batch_size, len(train_dataset), len(val_dataset))

            if not eval_only:
                logging.info(
                    f"Training on {train_dataset.name} with {len(train_dataset)} samples and validating on {val_dataset.name} with"
                    f" {len(val_dataset)} samples."
                )
            # logging.info(f"Using {num_workers} workers for data loading.")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                # num_workers=num_workers,
                drop_last=True,
                persistent_workers=persistent_workers,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                # num_workers=num_workers,
                drop_last=True,
                persistent_workers=persistent_workers,
            )

            data_shape = next(iter(train_loader))[0].shape

            if load_weights:
                model = load_model(model, source_dir, pl_model=pl_model)
            else:
                model = model(
                    optimizer=optimizer,
                    input_size=data_shape,
                    epochs=epochs,
                    run_mode=runmode,
                    cpu=cpu,
                    **model_kwargs,
                )

            model.set_weight(weight, train_dataset)
            model.set_trained_columns(train_dataset.get_feature_names())
            loggers = [TensorBoardLogger(log_dir), JSONMetricsLogger(log_dir)]
            devices = max(torch.cuda.device_count(), 1)

            if use_wandb:
                loggers.append(WandbLogger(save_dir=log_dir))
                logging.info("Use of wandb is detected. Only single gpu training is supported with wandb.")
                devices = 1

            callbacks = [
                EarlyStopping(monitor="val/loss", min_delta=min_delta, patience=patience, strict=False, verbose=verbose),
                ModelCheckpoint(log_dir, filename="model_{step}", save_top_k=1, save_last=True),
                LearningRateMonitor(logging_interval="step"),
            ]
            if verbose:
                callbacks.append(TQDMProgressBar(refresh_rate=min(100, len(train_loader) // 2)))
            if precision == 16 or "16-mixed":
                torch.set_float32_matmul_precision("medium")

            trainer = Trainer(
                max_epochs=epochs if model.requires_backprop else 1,
                min_epochs=1,  # We need at least one epoch to get results.
                callbacks=callbacks,
                precision=precision,
                accelerator="auto" if not cpu else "cpu",
                devices=devices,
                deterministic="warn" if reproducible else False,
                benchmark=not reproducible,
                enable_progress_bar=verbose,
                logger=loggers,
                log_every_n_steps=log_every_n_steps,
                num_sanity_val_steps=2,  # Helps catch errors in the validation loop before training begins.
            )
            if not eval_only:
                if model.requires_backprop:
                    logging.info("Training DL model.")
                    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                    logging.info("Training complete.")
                else:
                    logging.info("Training ML model.")
                    model.fit(train_dataset, val_dataset)
                    model.save_model(log_dir, "last")
                    logging.info("Training complete.")
            if train_only:
                logging.info("Finished training full model.")
                save_config_file(log_dir)
                # return 0
            test_dataset = dataset_class(data, split=test_on, name=dataset_names["test"], ram_cache=ram_cache)
            #test_dataset = assure_minimum_length(test_dataset)
            logging.info(f"Testing on {test_dataset.name}  with {len(test_dataset)} samples.")
            test_loader = (
                DataLoader(
                    test_dataset,
                    batch_size=min(batch_size * 4, len(test_dataset)),
                    shuffle=False,
                    # num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=persistent_workers,
                )
                if model.requires_backprop
                else DataLoader([test_dataset.to_tensor()], batch_size=1)
            )

            model.set_weight("balanced", train_dataset)
            test_loss = trainer.test(model, dataloaders=test_loader, verbose=verbose)[0]["test/loss"]
            #persist_shap_data(trainer, log_dir)
            #save_config_file(log_dir)

            agg_loss += test_loss
            #--------------------------#
            ############################
            #--------------------------#
        

            train_time = datetime.now() - start_time

            log_full_line(
                f"FINISHED FOLD {fold_index}| PREPROCESSING DURATION {preprocess_time}| PROCEDURE DURATION {train_time}",
                level=logging.INFO,
            )
            durations = {"preprocessing_duration": preprocess_time, "train_duration": train_time}

            with open(repetition_fold_dir / "durations.json", "w") as f:
                json.dump(durations, f, cls=JsonResultLoggingEncoder)
            if wandb:
                wandb_log({"Iteration": repetition * cv_folds_to_train + fold_index})
            if repetition * cv_folds_to_train + fold_index > 1:
                try:
                    aggregate_results(log_dir)
                except Exception as e:
                    logging.error(f"Failed to aggregate results: {e}")
        log_full_line(f"FINISHED CV REPETITION {repetition}", level=logging.INFO, char="=", num_newlines=3)

        if use_wandb:
            wandb.finish()

    return agg_loss / (cv_repetitions_to_train * cv_folds_to_train)