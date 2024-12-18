import os
from transformers.trainer import Trainer
import subprocess
from multiprocessing import Process
import json
import torch
from typing import Optional

class GenerationTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(GenerationTrainer, self).__init__(*pargs, **kwargs)
        self.model: GenerationRanker

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("Saving model checkpoint to {}".format(output_dir))

        # NOTE: we should remove the prefix of the outmost model
        prefix = 'model.'
        new_state_dict = type(state_dict)()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                new_state_dict[new_k] = v
        # save model
        self.model.save_pretrained(output_dir, state_dict=new_state_dict, safe_serialization=self.args.save_safetensors)
        # save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))


    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
