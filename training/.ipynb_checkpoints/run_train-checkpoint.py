import os
import torch
import time
import logging
import json
import math
import argparse
from tqdm import tqdm
from functools import partial
import random
import numpy as np
import bitsandbytes as bnb
import transformers
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
import datasets
from datasets import load_dataset
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, get_scheduler, CONFIG_MAPPING, HfArgumentParser, Trainer
from torch.utils.data import DataLoader, Dataset
from util.loss import lambdarank, listNet, rank_net
# from dataset import RankingDataset, GenerationDataset, ranking_collate_fn, generation_collate_fn, combined_collate_fn
from util.train_utils import load_data, NEFTune, parse_args, _replace_number

from transformers.utils.logging import enable_explicit_format, set_verbosity_info, set_verbosity_warning, enable_progress_bar
from transformers.trainer_callback import PrinterCallback
# from logger_config import logger, LoggerCallback
from arguments import Arguments

from collator import GenerationCollator
from dataset import GenerationDataset
from model import GenerationRanker
from trainer import GenerationTrainer

# logger = get_logger(__name__)
os.environ["PYSERINI_CACHE"] = "../../../cache"
os.environ["WANDB_DISABLED"] = "true"

def _common_setup(args: Arguments):
    enable_progress_bar()
    set_seed(args.seed)


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = GenerationRanker(args, args.model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # treat [i] as one token
    # if args.only_output_docid:
    # rankid_tokens = [f'[{i}]' for i in range(1, 101)]
    # tokenizer.add_tokens(rankid_tokens)
    # model.added_tokens = tokenizer.convert_tokens_to_ids(rankid_tokens)

    embedding_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if args.noisy_embedding_alpha is not None:
        model.model = NEFTune(model.model, args.noisy_embedding_alpha)
    model.tokenizer = tokenizer
    
    train_dataset = GenerationDataset(args, tokenizer)
    train_collate_fn = GenerationCollator(args, tokenizer)
    trainer = GenerationTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        data_collator=train_collate_fn,
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    # trainer.add_callback(LoggerCallback)
    if args.do_train:
        train_result = trainer.train()
        # trainer.save_model()

        # metrics = train_result.metrics
        # metrics["train_samples"] = len(train_dataset)
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    main()