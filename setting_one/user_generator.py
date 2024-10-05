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
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--num_turns", type=int, default=5)
    return parser.parse_args()


def get_prompt(trajectory):
    prompt = 'Below is a dialogue between the user and the assistant. Pretend you are the user in this conversation. What question would you ask next? \n\n'
    for turn in trajectory:
        prompt += '### ' + turn['role']
        prompt += ': '
        prompt += turn['content']
        prompt += '\n\n'
    prompt += '### Instructions: \nFIRST provide a justification of the question you want to ask. \nSECOND, on a new line, state only the question. Your response should use the format: \nJustification: \nQuestion: '
    return [{"role": "user", "content": prompt}]

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
            prompts.append(get_prompt(t))
            prompt_i_to_traj_i[len(prompts) - 1] = i
    prompts = [tokenizer.apply_chat_template(t, tokenize=False, add_generation_prompt=True) for t in prompts]

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
        try:
            trajectory[prompt_i_to_traj_i[r]].append({"role": "user", "content": output[r].rsplit('Question:', 1)[1].strip()})
        except:
            print(prompt_i_to_traj_i[r], 'added all outputs')
            trajectory[prompt_i_to_traj_i[r]].append({"role": "user", "content": output[r].strip()})
    with open(args.dataset, 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'prompt saved to {args.dataset}')
