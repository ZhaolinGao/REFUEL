from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os
import pickle
import random
import numpy as np
import torch


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--num_turns", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":

    # init
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
    )

    set_seed(args.seed)
    
    # construct prompts
    with open(args.dataset, 'rb') as handle:
        trajectory = pickle.load(handle)
    prompts, prompt_i_to_traj_i = [], {}
    for i, t in enumerate(trajectory):
        if len(t) < args.num_turns * 2:
            prompts.append(tokenizer.apply_chat_template(t, tokenize=False, add_generation_prompt=True))
            prompt_i_to_traj_i[len(prompts) - 1] = i

    # generate
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.maxlen,
        seed=args.seed
    )
    response = llm.generate(prompts, sampling_params)
    output = list(map(lambda x: x.outputs[0].text, response))

    # merge to trajectory and save
    for r in range(len(output)):
        trajectory[prompt_i_to_traj_i[r]].append({"role": "assistant", "content": output[r]})
    with open(args.dataset, 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'response saved to {args.dataset}')
