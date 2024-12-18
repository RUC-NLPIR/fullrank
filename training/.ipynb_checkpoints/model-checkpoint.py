import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.models.mistral.modeling_mistral import MistralConfig, MistralModel, MistralForCausalLM
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, MistralPreTrainedModel
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math

@dataclass
class RankingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

class GenerationRanker(nn.Module):
    def __init__(self, args, model_name_or_path, attn_implementation, torch_dtype):
        super().__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype
        )
        self.tokenizer = None
        self.added_tokens = None

    def forward(
        self,
        batch_dict
    ) -> RankingOutput:
        # print(batch_dict['input_ids'].shape)
        outputs = self.model(
            input_ids=batch_dict['input_ids'],
            attention_mask=batch_dict['attention_mask'],
        )
        # print(f'!!!!!!!!: {self.cnt}')
        # self.cnt += 1
        logits = outputs.logits
        # ################### first implementation ###################
        logits = logits[..., :-1, :].contiguous()
        # shift target for causal langauge modeling
        labels = batch_dict['labels'][..., 1:].contiguous()
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        epsilon = 0.0
        ignore_index = -100
        padding_mask = labels.eq(ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        nll_loss.masked_fill_(padding_mask, 0.0)
        if self.args.weighted_loss:
            # Compute weights
            batch_size, seq_length = labels.size()[:2]
            weights = torch.zeros_like(labels, dtype=torch.float, device=logits.device)
            for i in range(batch_size):
                # Find the first non-padding index
                non_pad_idx = (labels[i,:,0] != 0).nonzero(as_tuple=True)[0]
                start_idx = non_pad_idx[0].item()
                rank = 1
                for j in range(start_idx, seq_length):
                    if labels[i,j,0].item() == 1644: # '_>'
                        weights[i, j, 0] = 1
                        rank += 1
                    else:
                        weights[i, j, 0] = (1 / math.log2(rank + 1) + 1)
                # print('rank=', rank)
                    # weights[i, j, 0] = 1 / math.log2(j - start_idx + 2)
            # Apply weights to non-padding positions
            weights = weights.masked_fill(padding_mask, 0.0)
            nll_loss = nll_loss * weights

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        # print(labels.shape)
        # print(labels[0,-350:,0], self.tokenizer.convert_ids_to_tokens(labels[0,-350:,0].tolist()))
        loss = nll_loss
        ###################### MistralForCasualLM implementation ######################
        # logits = logits.float()
        # # Shift so that tokens < n predict n
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = batch_dict['labels'][..., 1:].contiguous()
        # # print(shift_labels.shape)
        # # print(shift_labels[0], self.tokenizer.convert_ids_to_tokens(shift_labels[0, -298:].tolist()))
        # # Flatten the tokens
        # # print(f'self.config.vocab_size:{self.config.vocab_size}')
        # shift_logits = shift_logits.view(-1, self.config.vocab_size)
        # shift_labels = shift_labels.view(-1)
        # # Ensure tensors are on the same device
        # shift_labels = shift_labels.to(shift_logits.device)
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(shift_logits, shift_labels)






        return RankingOutput(
            loss = loss,
            logits = logits
        )
    
    def save_pretrained(self, output_dir: str, *args, **kwargs):
        self.model.save_pretrained(output_dir, *args, **kwargs)

    @property
    def config(self):
        return self.model.config

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)
 
    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.model.gradient_checkpointing_disable(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def resize_token_embeddings(self, num):
        self.model.resize_token_embeddings(num)









