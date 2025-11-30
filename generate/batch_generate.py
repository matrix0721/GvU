import os
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from peft import PeftModel
import json
from tqdm import tqdm

def read_jsonl(file_name):
    results=[]
    with open(file_name, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results
        


def main(args):
    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).cuda()
    if args.lora_weights_path:
        model = PeftModel.from_pretrained(model, args.lora_weights_path, torch_dtype=torch_dtype)
    model.init_vision(args.flux_model_name_or_path)
    model.set_generation_mode('image')
    model.eval()

    if isinstance(args.image_size, int):
        image_size = (args.image_size, args.image_size) 
    elif len(args.image_size) == 1:
        image_size = (args.image_size[0], args.image_size[0])
    else:
        image_size = args.image_size

    token_h, token_w = image_size[0] // args.downsample_size, image_size[1] // args.downsample_size
    image_prefix = f'<SOM>{token_h} {token_w}<IMAGE>'
    generation_config = GenerationConfig(
        max_new_tokens=token_h * token_w,
        do_sample=True,
        temperature=args.temperature,
        min_p=args.min_p,
        top_p=args.top_p,
        guidance_scale=args.cfg_scale,
        suppress_tokens=tokenizer.convert_tokens_to_ids(model.config.mm_special_tokens),
    )

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        data = read_jsonl(args.prompt_file)
    
    prompts=[]
    for item in data:
        prompts.append(item['prompt'])

    for i, prompt in enumerate(tqdm(prompts)):
        if os.path.exists(f"{args.output_path}/{i}.png"):
            continue

        text = prompt + image_prefix

        # Sample inputs:
        tokens = tokenizer(
            [text],
            return_tensors='pt', 
            padding='longest', 
            padding_side='left',
        )
        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        negative_ids = tokenizer.encode(
            image_prefix, 
            add_special_tokens=False, 
            return_tensors='pt',
        ).cuda().expand(1, -1)

        torch.manual_seed(args.seed)
        tokens = model.generate(
            inputs=input_ids, 
            attention_mask=attention_mask,
            generation_config=generation_config,
            negative_prompt_ids=negative_ids,
        )
        
        tokens = torch.nn.functional.pad(tokens, (0, 1), value=tokenizer.convert_tokens_to_ids('<EOM>'))
        torch.manual_seed(args.seed)
        _, images = model.mmdecode(tokenizer, tokens[0], skip_special_tokens=False)
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        save_path = f"{args.output_path}/{i}.png"
        images[0].save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--flux_model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str)
    parser.add_argument("--prompt_file",type=str,required=True)
    parser.add_argument("--image-size", type=int, nargs='+', default=1152)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--output-path", type=str, default="./output.png")
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--min-p", type=float, default=0.03, help="min-p value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    main(args)
