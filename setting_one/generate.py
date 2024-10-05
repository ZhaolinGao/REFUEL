from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import argparse
import os
import pickle
import random
import time
import subprocess
import numpy as np


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)


# format the right chat template
def change_format(row):
    new_trajectory = []
    for i in row["trajectory"]:
        new_trajectory.append({'role' : i['from'], 'content' : i['value']})
    row["trajectory"] = new_trajectory
    return row


# filter redundant
unique_traj = {}
def check_redundant(tokenizer, row):
    tokenized = tokenizer.apply_chat_template(row["trajectory"], add_generation_prompt=False, tokenize=True)
    tokenized = tuple(tokenized)
    if tokenized in unique_traj:
        return False
    unique_traj[tokenized] = 1
    return True


def call_scripts(args, seed, gen_type):

    if gen_type == 'response':
        try:
            subprocess.run(['python', './setting_one/response_generator.py', \
                            '--dataset', f'{os.path.join(args.output_dir, "temp.pkl")}', \
                            '--temperature', f'{args.temperature}', \
                            '--maxlen', f'{args.maxlen}', \
                            '--world_size', f'{args.world_size}', \
                            '--model', f'{args.model}', \
                            '--seed', f'{seed}', \
                            '--num_turns', f'{args.num_turns}'], check=True)
        except:
            return False
    else:
        try:
            subprocess.run(['python', './setting_one/user_generator.py', \
                            '--dataset', f'{os.path.join(args.output_dir, "temp.pkl")}', \
                            '--temperature', f'{args.temperature}', \
                            '--maxlen', f'{args.maxlen}', \
                            '--world_size', f'{args.world_size}', \
                            '--model', f'{args.user_model}', \
                            '--seed', f'{seed}', \
                            '--num_turns', f'{args.num_turns}'], check=True)
        except:
            return False
    
    return True


def call_scripts_wrapper(args, seed, gen_type):
    while not call_scripts(args, seed, gen_type):
        time.sleep(20)
        print(f'error when generating {gen_type}')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--user_model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")

    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_repo", type=str, default="")

    parser.add_argument("--dataset", type=str, default="openbmb/UltraInteract_pair")
    parser.add_argument("--dataset_split", type=str, default="train")

    parser.add_argument("--num_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num_data", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":

    # init
    args = parse_arguments()
    set_seed(args.seed)
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # preprocess
    if args.num_data != 0:
        dataset = dataset.select(range(args.num_data))
    dataset = dataset.map(change_format)
    dataset = dataset.filter(lambda row: check_redundant(tokenizer, row))

    # save prompt from the initial turn
    trajectory = []
    for i in range(len(dataset)):
        trajectory.append([dataset[i]['trajectory'][0]])
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'initial prompt saved to {os.path.join(args.output_dir, "temp.pkl")}')

    # generate for num_turns
    for i in range(args.num_turns):
        call_scripts_wrapper(args, args.seed, gen_type='response')

        # load saved trajectories
        with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
            trajectory = pickle.load(handle)

        # save checkpoint
        temp_dataset = Dataset.from_dict({"trajectory": trajectory})
        temp_dataset.push_to_hub(args.output_repo + f'_{args.num_turns}_turns_only_ckp_{i}')
        
        if i < args.num_turns - 1:
            call_scripts_wrapper(args, args.seed, gen_type='user')  

    # load saved trajectories
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
        trajectory = pickle.load(handle)
    generated = Dataset.from_dict({"trajectory": trajectory})
    generated.push_to_hub(args.output_repo + f'_{args.num_turns}_turns_only')

    # ==========================================================================

    # randomly sample an h from num_turns
    sampled_len = np.random.choice(args.num_turns, size=len(generated))
    sampled_h = np.random.randint(0, sampled_len + 1)
    generated = generated.add_column(f"sampled_len_from_{args.num_turns}", sampled_len)
    generated = generated.add_column(f"sampled_h_from_sampled_len", sampled_h)

    # save prompt from the sampled turn
    trajectory = []
    for i in range(len(generated)):
        trajectory.append(generated[i]['trajectory'][:sampled_h[i]*2+1])
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'sampled prompt saved to {os.path.join(args.output_dir, "temp.pkl")}')

    # generate for num_turns
    for i in range(args.num_turns):
        call_scripts_wrapper(args, args.seed + 20000, gen_type='response')

        # load saved trajectories
        with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
            trajectory = pickle.load(handle)

        # save checkpoint
        temp_dataset = Dataset.from_dict({f"trajectory_sampled_h_from_sampled_len": trajectory})
        temp_dataset = temp_dataset.add_column(f"sampled_len_from_{args.num_turns}", sampled_len)
        temp_dataset = temp_dataset.add_column(f"sampled_h_from_sampled_len", sampled_h)
        temp_dataset.push_to_hub(args.output_repo + f'_sampled_h_from_sampled_len_ckp_{i}')

        if i < args.num_turns - 1:
            call_scripts_wrapper(args, args.seed + 20000, gen_type='user')

    # load saved trajectories
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
        trajectory = pickle.load(handle)

    # save checkpoint
    generated = generated.add_column(f"trajectory_sampled_h_from_sampled_len", trajectory)
    generated.push_to_hub(args.output_repo)

    os.remove(os.path.join(args.output_dir, 'temp.pkl'))