import os
import torch
import math
import transformers
import numpy as np
from tqdm import tqdm
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
from huggingface_hub import snapshot_download
from datasets import load_dataset, DatasetDict
from huggingface_hub import create_repo


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', padding_side='left')
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

tokenizer_right = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', padding_side='right')
tokenizer_right.add_special_tokens({"pad_token": "[PAD]"})


MAX_NUM_TURNS = 5
MAX_PROMPT_TOKENS = 128
MAX_RESPONSE_TOKENS = 512
repo_name = ''


# check the order in the data
def check_order(row):

    if len(row['chosen']) % 2 != 0:
        return False

    for idx, response in enumerate(row['chosen']):
        if idx % 2 == 0 and response['role'] == 'assistant':
            return False
        elif idx % 2 == 1 and response['role'] == 'user':
            return False

    return True


# filter the length of the dialogue
def filter_long_dialogue(row, max_num_turns=MAX_NUM_TURNS, max_prompt_tokens=MAX_PROMPT_TOKENS, max_response_tokens=MAX_RESPONSE_TOKENS):

    if len(row['chosen']) > 2 * max_num_turns:
        return False

    for idx, i in enumerate(row['chosen']):

        if idx == 0:
            if tokenizer.apply_chat_template([i], add_generation_prompt=True, tokenize=True, return_tensors='pt').shape[-1] > max_prompt_tokens:
                return False
            continue

        if i['role'] == 'user':
            if tokenizer.apply_chat_template([i], add_generation_prompt=True, tokenize=True, return_tensors='pt')[:, 1:].shape[-1] > max_prompt_tokens:
                return False
        else:
            if tokenizer.apply_chat_template([i], add_generation_prompt=False, tokenize=True, return_tensors='pt')[:, 5:].shape[-1] > max_response_tokens:
                return False

    return True

# ==================

### train
dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split='train')

# filter row with long dialogue
print('filtering llama')
print(len(dataset))
filtered_dataset = dataset.filter(lambda row: check_order(row))
print(len(filtered_dataset))
filtered_dataset = filtered_dataset.filter(lambda row: filter_long_dialogue(row))
print(len(filtered_dataset))

data_dict = {'llama_dialogue': [],
             'llama_dialogue_tokens': [],
             'num_turn': []}
for idx in range(MAX_NUM_TURNS):
    data_dict[f'llama_prompt_turn_{str(idx)}'] = []
    data_dict[f'llama_prompt_token_turn_{str(idx)}'] = []
    data_dict[f'llama_response_turn_{str(idx)}'] = []
    data_dict[f'llama_response_token_turn_{str(idx)}'] = []

# add llama prompts
print('adding llama tokens')
for row in tqdm(filtered_dataset):

    data_dict['llama_dialogue'].append(tokenizer.apply_chat_template(row['chosen'], tokenize=False, add_generation_prompt=False))
    data_dict['llama_dialogue_tokens'].append(tokenizer.apply_chat_template(row['chosen'], tokenize=True, add_generation_prompt=False))
    data_dict['num_turn'].append(len(row['chosen']) // 2)

    for idx, i in enumerate(row['chosen']):
        
        if idx == 0:
            data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'].append(tokenizer.apply_chat_template([i], tokenize=True, add_generation_prompt=True, max_length=MAX_PROMPT_TOKENS, padding=True,))
            data_dict[f'llama_prompt_turn_{str(idx // 2)}'].append(tokenizer.decode(data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'][-1], skip_special_tokens=False))
            continue

        if i['role'] == 'user':
            data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'].append(tokenizer_right.apply_chat_template([i], tokenize=True, add_generation_prompt=True, max_length=MAX_PROMPT_TOKENS+1, padding=True,)[1:])
            data_dict[f'llama_prompt_turn_{str(idx // 2)}'].append(tokenizer_right.decode(data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'][-1], skip_special_tokens=False))
        else:
            data_dict[f'llama_response_token_turn_{str(idx // 2)}'].append(tokenizer_right.apply_chat_template([i], tokenize=True, add_generation_prompt=False, max_length=MAX_RESPONSE_TOKENS+5, padding=True,)[5:])
            data_dict[f'llama_response_turn_{str(idx // 2)}'].append(tokenizer_right.decode(data_dict[f'llama_response_token_turn_{str(idx // 2)}'][-1], skip_special_tokens=False))
    
    for j in range((idx+1)//2, MAX_NUM_TURNS):
        data_dict[f'llama_prompt_turn_{str(j)}'].append("".join([tokenizer.pad_token for i in range(MAX_PROMPT_TOKENS)]))
        data_dict[f'llama_prompt_token_turn_{str(j)}'].append([tokenizer.pad_token_id for i in range(MAX_PROMPT_TOKENS)])
        data_dict[f'llama_response_turn_{str(j)}'].append("".join([tokenizer.pad_token for i in range(MAX_RESPONSE_TOKENS)]))
        data_dict[f'llama_response_token_turn_{str(j)}'].append([tokenizer.pad_token_id for i in range(MAX_RESPONSE_TOKENS)])

    for idx in range(MAX_NUM_TURNS):
        assert len(data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1]) == MAX_PROMPT_TOKENS
        assert len(data_dict[f'llama_response_token_turn_{str(idx)}'][-1]) == MAX_RESPONSE_TOKENS
        if idx == 0:
            assert data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][0] == 128000 or data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][0] == 128256
        else:
            assert data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][-1] == 128256 or data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][-1] == 271

for k, v in data_dict.items():
    filtered_dataset = filtered_dataset.add_column(k, v)

### test
dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split='test')

# filter row with long dialogue
print('filtering llama')
print(len(dataset))
filtered_test_dataset = dataset.filter(lambda row: check_order(row))
print(len(filtered_test_dataset))
filtered_test_dataset = filtered_test_dataset.filter(lambda row: filter_long_dialogue(row))
print(len(filtered_test_dataset))

data_dict = {'llama_dialogue': [],
             'llama_dialogue_tokens': [],
             'num_turn': []}
for idx in range(MAX_NUM_TURNS):
    data_dict[f'llama_prompt_turn_{str(idx)}'] = []
    data_dict[f'llama_prompt_token_turn_{str(idx)}'] = []
    data_dict[f'llama_response_turn_{str(idx)}'] = []
    data_dict[f'llama_response_token_turn_{str(idx)}'] = []

# add llama prompts
print('adding llama tokens')
for row in tqdm(filtered_test_dataset):

    data_dict['llama_dialogue'].append(tokenizer.apply_chat_template(row['chosen'], tokenize=False, add_generation_prompt=False))
    data_dict['llama_dialogue_tokens'].append(tokenizer.apply_chat_template(row['chosen'], tokenize=True, add_generation_prompt=False))
    data_dict['num_turn'].append(len(row['chosen']) // 2)

    for idx, i in enumerate(row['chosen']):
        
        if idx == 0:
            data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'].append(tokenizer.apply_chat_template([i], tokenize=True, add_generation_prompt=True, max_length=MAX_PROMPT_TOKENS, padding=True,))
            data_dict[f'llama_prompt_turn_{str(idx // 2)}'].append(tokenizer.decode(data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'][-1], skip_special_tokens=False))
            continue

        if i['role'] == 'user':
            data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'].append(tokenizer_right.apply_chat_template([i], tokenize=True, add_generation_prompt=True, max_length=MAX_PROMPT_TOKENS+1, padding=True,)[1:])
            data_dict[f'llama_prompt_turn_{str(idx // 2)}'].append(tokenizer_right.decode(data_dict[f'llama_prompt_token_turn_{str(idx // 2)}'][-1], skip_special_tokens=False))
        else:
            data_dict[f'llama_response_token_turn_{str(idx // 2)}'].append(tokenizer_right.apply_chat_template([i], tokenize=True, add_generation_prompt=False, max_length=MAX_RESPONSE_TOKENS+5, padding=True,)[5:])
            data_dict[f'llama_response_turn_{str(idx // 2)}'].append(tokenizer_right.decode(data_dict[f'llama_response_token_turn_{str(idx // 2)}'][-1], skip_special_tokens=False))
    
    for j in range((idx+1)//2, MAX_NUM_TURNS):
        data_dict[f'llama_prompt_turn_{str(j)}'].append("".join([tokenizer.pad_token for i in range(MAX_PROMPT_TOKENS)]))
        data_dict[f'llama_prompt_token_turn_{str(j)}'].append([tokenizer.pad_token_id for i in range(MAX_PROMPT_TOKENS)])
        data_dict[f'llama_response_turn_{str(j)}'].append("".join([tokenizer.pad_token for i in range(MAX_RESPONSE_TOKENS)]))
        data_dict[f'llama_response_token_turn_{str(j)}'].append([tokenizer.pad_token_id for i in range(MAX_RESPONSE_TOKENS)])

    for idx in range(MAX_NUM_TURNS):
        assert len(data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1]) == MAX_PROMPT_TOKENS
        assert len(data_dict[f'llama_response_token_turn_{str(idx)}'][-1]) == MAX_RESPONSE_TOKENS
        if idx == 0:
            assert data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][0] == 128000 or data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][0] == 128256
        else:
            assert data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][-1] == 128256 or data_dict[f'llama_prompt_token_turn_{str(idx)}'][-1][-1] == 271

for k, v in data_dict.items():
    filtered_test_dataset = filtered_test_dataset.add_column(k, v)

# push to hub
ds_dict = {'train' : filtered_dataset,
            'test' : filtered_test_dataset}
ds = DatasetDict(ds_dict)
ds.push_to_hub(repo_name)