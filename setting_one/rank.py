import argparse
import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List


torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--output_repo", type=str, default="")
    parser.add_argument("--prompts", type=str, default="")
    parser.add_argument('--columns', nargs='+', default=["trajectory", "trajectory_sampled_h_from_sampled_len"])
    parser.add_argument("--num_turns", type=int, default=5)
    return parser.parse_args()


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return score


def main(args):

    dataset = load_dataset(args.prompts, split='train')

    rewards = {}
    rm = ArmoRMPipeline(args.reward_model, trust_remote_code=True)

    # gather reward
    for col in args.columns:
        print(f'gathering reward for {col}')
        assert len(dataset[0][col]) == 2 * args.num_turns
        rewards[col] = []
        for row in tqdm(dataset):
            reward = []
            for i in range(1, args.num_turns+1):
                try:
                    r = rm(row[col][:i*2])
                except:
                    print(row[col][:i*2], -99999)
                    r = -99999
                reward.append(r)
            rewards[col].append(reward)
    for k, v in rewards.items():
        dataset = dataset.add_column(k+'_reward', v)

    dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
    