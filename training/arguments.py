import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments
import sys
# from logger_config import logger
sys.path.append('../')
from rerank.rankllm import PromptMode
from config import PROJECT_DIR

@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = field(
        default=f'../llm/Mistral-7B-Instruct-v0.3',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train_dataset_path: str = field(
        default=f'{PROJECT_DIR}/training_data/msmarco_train_100to100.jsonl',
        metadata={"help": "Training dataset path in jsonl format"}
    )
    noisy_embedding_alpha: int = field(
        default=None,
        metadata={"help": "NEFT https://arxiv.org/abs/2310.05914, set this to a number (paper default is 5) to add noise to embeddings"}
    )
    seed: int = field(
        default=1,
        metadata={"help": "A seed for reproducible training."}
    )
    with_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to enable experiment trackers for logging."}
    )
    max_passage_len: int = field(
        default=100,
        metadata={"help": "The max length of a passage."}
    )
    system_message: str = field(
        default="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        metadata={"help": "The system message used in prompts."}
    )
    variable_passages: bool = field(
        default=False,
        metadata={"help": "Whether the model can account for a variable number of passages in input."}
    )
    weighted_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use our importance-aware loss"}
    )
    prompt_mode: PromptMode = field(
        default=PromptMode.RANK_GPT,
        metadata={'required': True, 'choices': list(PromptMode)}
    )
