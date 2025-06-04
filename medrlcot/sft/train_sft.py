"""
Super-vised fine-tuning (SFT) for MedRL-CoT

▪︎ Loads the two clinical-note corpora with   preprocess_datasets()
▪︎ Collapses note-level rows into full cases with xy_split_processing_sft()
▪︎ Converts to a single 🤗 Dataset   (train/val split)
▪︎ Tokenises with a 32k-sentencepiece Flan-T5-base tokenizer
▪︎ Fine-tunes the model for long (1 000-token) inputs/outputs
"""

from __future__ import annotations
import os, random, logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import datasets as hf_datasets
from tqdm.auto import tqdm                                                    # progress bars

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)

from medrlcot.preprocessing import preprocess_datasets, xy_split_processing_sft
from medrlcot.medrlcot_logger import setup_logger


# -------------------------------------------------------
# Globals / hyper-params
# -------------------------------------------------------
MODEL_NAME         = "google/flan-t5-small"
MAX_SOURCE_LENGTH  = 1_000      # tokens in the clinical note
MAX_TARGET_LENGTH  = 1_000      # tokens in the CoT + diagnosis
BATCH_SIZE         = 1          # fits on 12 GB GPUs with gradient-ckpt
VAL_SPLIT          = 0.1
SEED               = 42


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def build_case_dataframe() -> pd.DataFrame:
    """Notebook-style pipeline ⇒ one row == (X, Y) per clinical case."""
    log.info("◼︎ Pre-processing raw note rows …")
    raw_ds: Dict[str, pd.DataFrame] = preprocess_datasets()                    # rows
    case_frames: List[pd.DataFrame] = []

    for name, df in raw_ds.items():
        log.info(f"  – aggregating {name} into cases …")
        grouped = (
            df.groupby("case_id")
              .apply(xy_split_processing_sft)                                  # → (X,Y)
              .reset_index(drop=True)
        )
        case_frames.append(grouped[["X", "Y"]])

    merged = pd.concat(case_frames, ignore_index=True).sample(
        frac=1.0, random_state=SEED                                            # shuffle
    )
    log.info(f"  → final case dataframe: {len(merged):,} examples")
    return merged


def tokenise(batch, tok: AutoTokenizer):
    """Vectorise a mini-batch → model inputs."""
    model_inputs = tok(
        batch["X"],
        max_length=MAX_SOURCE_LENGTH,
        padding="max_length",
        truncation=True,
    )
    with tok.as_target_tokenizer():
        labels = tok(
            batch["Y"],
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    global log
    save_dir = Path("checkpoints") / "sft_flan_base_long"
    save_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger("train_sft", save_dir / "train_sft.log")

    # 1. data -----------------------------------------------------------------
    cases_df           = build_case_dataframe()
    hf_ds              = hf_datasets.Dataset.from_pandas(cases_df)
    ds_splits          = hf_ds.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    log.info(ds_splits)

    # 2. tokenizer & model ----------------------------------------------------
    tok_cfg            = AutoConfig.from_pretrained(MODEL_NAME)
    tok_cfg.max_position_embeddings = MAX_SOURCE_LENGTH
    tok                = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model_cfg          = AutoConfig.from_pretrained(MODEL_NAME)
    model_cfg.update(
        {
            "vocab_size"        : tok.vocab_size,
            "max_position_embeddings": MAX_SOURCE_LENGTH,
        }
    )
    model              = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        config=model_cfg,
        gradient_checkpointing=True,
    )

    # 3. tokenisation  --------------------------------------------------------
    log.info("◼︎ Tokenising …")
    tokenised_splits   = ds_splits.map(
        lambda b: tokenise(b, tok),
        batched=True,
        remove_columns=["X", "Y"],
        desc="tokenising",
    )

    # 4. training args / trainer ---------------------------------------------
    collator           = DataCollatorForSeq2Seq(tok, model, pad_to_multiple_of=8)
    args               = Seq2SeqTrainingArguments(
        output_dir          = str(save_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = 4,
        learning_rate       = 5e-5,
        num_train_epochs    = 3,
        fp16                = True,
        logging_strategy    = "steps",
        logging_steps       = 50,
        evaluation_strategy = "epoch",
        save_strategy       = "epoch",
        predict_with_generate= True,
        seed                = SEED,
    )

    trainer            = Seq2SeqTrainer(
        model              = model,
        args               = args,
        data_collator      = collator,
        train_dataset      = tokenised_splits["train"],
        eval_dataset       = tokenised_splits["test"],
    )

    log.info("◼︎ Starting training …")
    trainer.train()
    trainer.save_model(save_dir / "final")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
