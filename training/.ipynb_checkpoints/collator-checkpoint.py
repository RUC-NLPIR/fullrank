import torch
from typing import Any, Dict, List, Union, Optional, Sequence
from dataclasses import dataclass, field
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from config import Arguments
import copy
import logging
from fastchat.model import get_conversation_template
import transformers
from torch.utils.data import Dataset
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset
from copy import deepcopy
import re
import json

IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels, sources_tokenized["input_ids_lens"]

@dataclass
class GenerationCollator:
    args: Arguments
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]):
        prompts: List[str] = [f['inputs'] for f in features]
        labels: List[str] = [f['labels'] for f in features]
        tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
        tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
            tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch_dict = {
            'inputs': tokenized_inputs,
            'attention_mask': tokenized_inputs.ne(tokenizer.pad_token_id),
            'labels': labels,
        }
        return batch_dict

