import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Any, List, Union
from fastchat.model import get_conversation_template
import torch
import transformers
from torch.utils.data import Dataset
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset
from copy import deepcopy
import re
import json
import sys
import os
sys.path.append('../')
from utils import convert_doc_to_prompt_content, replace_number, add_prefix_prompt, add_post_prompt
from rerank.rankllm import PromptMode

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

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    numbers = [int(num) for num in numbers]
    return numbers

def get_ranking_label(init_passage_list, reranked_passage_list):
    ranking_label = ' > '.join([f'[{init_passage_list.index(passage) + 1}]' for passage in reranked_passage_list])
    return ranking_label

def _clean_response(response: str) -> str:
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response

def _remove_duplicate(response: List[int]) -> List[int]:
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

class GenerationDataset(Dataset):
    def __init__(self, args, tokenizer, combined=False) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.combined = combined
        self.system_message_supported = "system" in self.tokenizer.chat_template

        self.id_query = {}
        with open(f'../data/ms_marco/passage_ranking/queries/queries.train.tsv', 'r', encoding='utf-8') as f:
            for line in f:
                qid, query = line.split('\t')
                self.id_query[qid] = query
        self.searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        self.dataset: Dataset = load_dataset('json', data_files=args.train_dataset_path, split='train')
        # self.dataset = self.dataset.select(range(1000))
        self.dataset.set_transform(self._transform_func)
        print(f'dataset size: {len(self.dataset)}')

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        examples = deepcopy(examples)
        inputs, labels = [], []
        for i in range(len(examples['qid'])):
            qid = examples['qid'][i]
            initial_list = examples['initial_list'][i]
            final_list = examples['final_list'][i]
            label = get_ranking_label(initial_list, final_list)
            query = self.id_query[qid].strip()
            query = replace_number(query)
            num = len(initial_list)
            conv = get_conversation_template(self.args.model_name_or_path)
            if self.args.system_message:
                conv.set_system_message(self.args.system_message)
            prefix = add_prefix_prompt(self.args.prompt_mode, query=query, num=num)
            rank = 0
            input_context = f"{prefix}\n"
            for passage_id in initial_list:
                rank += 1
                passage = json.loads(self.searcher.doc(passage_id).raw())
                passage_content = convert_doc_to_prompt_content(self.tokenizer, passage, self.args.max_passage_len, truncate_by_word=False)
                input_context += f"[{rank}] {passage_content}\n"

            input_context += add_post_prompt(promptmode=self.args.prompt_mode, variable_passages=self.args.variable_passages, query=query, num=num)
            conv.append_message(conv.roles[0], input_context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompt = fix_text(prompt)
            inputs.append(prompt)
            labels.append(label)
        batch_dict = {
            'inputs': inputs,
            'labels': labels
        }
        return batch_dict

