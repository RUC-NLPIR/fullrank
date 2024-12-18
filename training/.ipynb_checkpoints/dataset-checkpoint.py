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
from config import WORKSPACE_DIR

START_IDX = ord('A')
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

def correct_label(candidate_passage_list, label, pospids):
    idx_list = extract_numbers(label)
    rerank_passage_list = [candidate_passage_list[idx - 1] for idx in idx_list]
    gold_rerank_passage_list = [pid for pid in rerank_passage_list if pid in pospids] + [pid for pid in rerank_passage_list if pid not in pospids]
    corrected_label = ' > '.join([f'[{candidate_passage_list.index(pid) + 1}]' for pid in gold_rerank_passage_list])
    return corrected_label

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
        # self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.combined = combined
        self.system_message_supported = "system" in self.tokenizer.chat_template

        self.id_query = {}
        with open(f'{WORKSPACE_DIR}/data/ms_marco/passage_ranking/queries/queries.train.tsv', 'r', encoding='utf-8') as f:
            for line in f:
                qid, query = line.split('\t')
                self.id_query[qid] = query
        self.searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        self.dataset: Dataset = load_dataset('json', data_files=args.train_dataset_path, split='train')
        self.dataset = self.dataset.select(range(300))
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
            # rerank_details = examples['rerank_details'][i]
            final_list = examples['final_list'][i]
            # pospids = examples['pospids'][i]
            # candidate_passage_list = examples['retrieved_passage_list'][i]
            # label = examples['label'][i]
            label = get_ranking_label(initial_list, final_list)
            # if self.args.label_correction:
            #     label = correct_label(candidate_passage_list, label, pospids)
            if self.args.prompt_mode == str(PromptMode.RANK_GPT_new):
                label = label.replace('[', '').replace(']', '').replace(' > ', '>')
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
                input_context += f"[{rank}] {passage_content}\n" if self.args.prompt_mode == str(PromptMode.RANK_GPT) else f"{rank} {passage_content}\n"
            # if self.args.only_output_docid:
            #     input_context += add_post_prompt_new(self.args, query, num)
            #     label = label.replace(' > ', '')
            # else:
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

    # def _transform_func_for_slidingwindow(self, examples: Dict[str, List]) -> Dict[str, List]:
    #     examples = deepcopy(examples)
    #     inputs, labels = [], []
    #     for i in range(len(examples['qid'])):
    #         qid = examples['qid'][i]
    #         initial_list = examples['initial_list'][i]
    #         rerank_details = examples['rerank_details'][i]
    #         final_list = examples['final_list'][i]
    #         # pospids = examples['pospids'][i]
    #         # candidate_passage_list = examples['retrieved_passage_list'][i]
    #         # label = examples['label'][i]

    #         for detail in rerank_details: 
    #             in_window_passages = detail['passage_ids']
    #             llm_output = detail['llm_output']
    #             response = _clean_response(llm_output)
    #             response = [int(x) for x in response.split()]
    #             response = _remove_duplicate(response)
    #             original_rank = [tt for tt in range(1, len(in_window_passages) + 1)]
    #             response = [ss for ss in response if ss in original_rank]
    #             response = response + [tt for tt in original_rank if tt not in response]
    #             label = ' > '.join([f'[{rankid}]' for rankid in response])
    #             # if self.args.label_correction:
    #             #     label = correct_label(candidate_passage_list, label, pospids)
    #             if self.args.prompt_mode == str(PromptMode.RANK_GPT_new):
    #                 label = label.replace('[', '').replace(']', '').replace(' > ', '>')
    #             query = self.id_query[qid].strip()
    #             query = replace_number(query)
    #             num = len(initial_list)
    #             conv = get_conversation_template(self.args.model_name_or_path)
    #             if self.args.system_message:
    #                 conv.set_system_message(self.args.system_message)
    #             prefix = add_prefix_prompt(self.args.prompt_mode, query=query, num=num)
    #             rank = 0
    #             input_context = f"{prefix}\n"
    #             for passage_id in initial_list:
    #                 rank += 1
    #                 passage = json.loads(self.searcher.doc(passage_id).raw())
    #                 passage_content = convert_doc_to_prompt_content(self.tokenizer, passage, self.args.max_passage_len, truncate_by_word=False)
    #                 input_context += f"[{rank}] {passage_content}\n" if self.args.prompt_mode == str(PromptMode.RANK_GPT) else f"{rank} {passage_content}\n"
    #             # if self.args.only_output_docid:
    #             #     input_context += add_post_prompt_new(self.args, query, num)
    #             #     label = label.replace(' > ', '')
    #             # else:
    #             input_context += add_post_prompt(promptmode=self.args.prompt_mode, variable_passages=self.args.variable_passages, query=query, num=num)
    #             conv.append_message(conv.roles[0], input_context)
    #             conv.append_message(conv.roles[1], None)
    #             prompt = conv.get_prompt()
    #             prompt = fix_text(prompt)
    #             inputs.append(prompt)
    #             labels.append(label)
    #     batch_dict = {
    #         'inputs': inputs,
    #         'labels': labels,
    #     }
    #     return batch_dict

# def generation_collate_fn(data, tokenizer):
#     # print(data)
#     # print('-------------------')
#     prompts = [item['inputs'] for item in data]
#     labels = [item['labels'] for item in data]
#     # prompts, labels = list(zip(*data))
#     tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
#     tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
#         tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
#     )
#     labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#     return tokenized_inputs, labels

# class RankingDataset(Dataset):
#     def __init__(self, raw_data, tokenizer, type) -> None:
#         self.raw_data = raw_data
#         self.tokenizer = tokenizer
#         self.tokenizer.padding_side="left"
#         self.type = type
#         self.system_message_supported = "system" in self.tokenizer.chat_template
    
#     def __getitem__(self, index):
#         conversation = self.raw_data[index]["conversations"]
#         sys_msg = conversation[0]['value']
#         input_context = conversation[1]['value']
#         target_generation = conversation[2]["value"]

#         if self.system_message_supported:
#             messages = [
#                 {"role": "system", "content": sys_msg},
#                 {"role": "user", "content": input_context}
#             ]
#         else:
#             messages = [
#                 {"role": "user", "content": sys_msg + "\n " + input_context}
#             ]
#         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         prompt += "["
#         prompt = fix_text(prompt)

#         if self.type == "train":
#             label_map = {}
#             label_rank = 0
#             for token in target_generation:
#                 if token.isalpha():
#                     label_map[token] = label_rank
#                     label_rank += 1
            
#             label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]

#         elif self.type == "eval":
#             label = [self.raw_data[index]["id"]] + self.raw_data[index]["docids"] + self.raw_data[index]["scores"]
#         else:
#             raise Exception("Invalid run type specified for Dataset. Choose from ['train', 'eval']")
#         return prompt, label
    
#     def __len__(self):
#         return len(self.raw_data)

# def ranking_collate_fn(data, tokenizer):
#     prompts, labels = list(zip(*data))
#     tokenized_inputs = tokenizer(prompts, padding="longest", truncation=False, return_tensors="pt")
#     return tokenized_inputs, labels

# def combined_collate_fn(data, tokenizer):
#     prompts, labels, rank_labels = list(zip(*data))
#     tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
#     tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
#         tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
#     )
#     labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#     return tokenized_inputs, labels, rank_labels, source_lens
