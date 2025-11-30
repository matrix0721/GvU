# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from peft import LoraConfig
from omni_r1.trainer import XOmniGRPOTrainer
from omni_r1.utils.rewards import gvu_score

import json

from omni_r1.models import XOmniForCausalLM, XOmniConfig, XOmniModel

AutoModel.register(XOmniConfig, XOmniModel)
AutoModelForCausalLM.register(XOmniConfig, XOmniForCausalLM)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

# add image_generation_prompt in the GRPOConfig
@dataclass
class GRPOConfig(GRPOConfig):
    """
    Configuration class for the GRPO training script.
    """
    flux_model_name_or_path : str = field(default="/root/paddlejob/workspace/env_run/models/FLUX.1-dev")
    token_model_name_or_path : str = field(default="/root/paddlejob/workspace/env_run/models/X-Omni-SFT")
    num_generations: int = field(default=2, metadata={"help": "The number of new generations of image to generate"})
    cfg_scale: float = field(default=1.0, metadata={"help": "The cfg weight for image generation"})
    min_p: float = field(default=0.03, metadata={"help": "min-p value to sample with"})
    top_p: float = field(default=1.0, metadata={"help": "top-p value to sample with"})
    temperature: float = field(default=1.0, metadata={"help": "temperature value to sample with"})

    image_size: int = field(default=576, metadata={"help": "The size of the image to generate"})
    downsample_size: int = field(default=16, metadata={"help": "The downsample size of the image to generate"})

    seed: int = field(default=1234, metadata={"help": "The seed for random number generation"})
    eval_strategy: str = field(default='no', metadata={"help": "The evaluation strategy"})
    lora_namespan_exclude : List[str] = field(default_factory=lambda: [], metadata={"help": "The names of the modules to exclude from LoRA"})


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'hps', 'git', 'gdino'.
    """

    reward_funcs: list[str] = field(
        #default_factory=lambda: ["hps", "git", "gdino", "orm"],
        default_factory= lambda: [ "prob_score_2"],
        metadata={"help": "List of reward functions. Possible values: 'hps', 'git', 'gdino', 'orm'"},
    )

    train_dataset_name: str = field(default='/root/paddlejob/workspace/env_run/output/projects/flow_grpo/dataset/geneval/train_metadata.jsonl', metadata={"help": "The name of the training dataset"})
    test_dataset_name: str = field(default='/root/paddlejob/workspace/env_run/output/projects/flow_grpo/dataset/geneval/test_metadata.jsonl', metadata={"help": "The name of the test dataset"})


reward_funcs_registry = {
    'prob_score_2': gvu_score,
}

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names


def main(script_args, training_args, model_args):
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    if script_args.train_dataset_name.endswith('.csv'):
        suffix = 'csv'
    elif script_args.train_dataset_name.endswith('.json'):
        suffix = 'json'
    elif script_args.train_dataset_name.endswith('.parquet'):
        suffix = 'parquet'
    elif script_args.train_dataset_name.endswith('.jsonl'):
        suffix = 'jsonl'
    
    train_dataset = GenevalPromptDataset(script_args.dataset_name, 'train')
    test_dataset = GenevalPromptDataset(script_args.dataset_name, 'test')
    #train_dataset = TextPromptDataset(script_args.dataset_name, 'train')
    #test_dataset = TextPromptDataset(script_args.dataset_name, 'test')

   
    def make_conversation(example):
        prompt= example["prompt"]
        return prompt


    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    if model_args.use_peft:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=-1),
        )
    else:
        peft_config = None

    del model
    torch.cuda.empty_cache()
    
    trainer_cls = XOmniGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        attn_implementation=model_args.attn_implementation,
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

def main_for_debug():
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


