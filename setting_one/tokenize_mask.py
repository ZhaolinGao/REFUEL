import numpy as np
import pickle
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer


MAX_LEN = 2048


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_repo", type=str, default="")
    parser.add_argument("--input_repo", type=str, default="")
    return parser.parse_args()


def filter_reward(dataset, columns, reward_to_filter, mask):

    output_mask = mask

    for i, row in tqdm(enumerate(dataset), total=len(dataset)):
        if mask[i] == 0:
            continue
        for c in columns:
            if reward_to_filter in row[c]:
                output_mask[i] = 0
                break
    
    return output_mask


def filter_same_response(dataset, col_1, col_2, turn, mask):

    output_mask = mask

    if type(turn) == str:
        for i, row in tqdm(enumerate(dataset), total=len(dataset)):
            if mask[i] == 0:
                continue
            if row[col_1][row[turn]*2 + 1] == row[col_2][row[turn]*2 + 1]:
                output_mask[i] = 0
    elif type(turn) == int:
        for i, row in tqdm(enumerate(dataset), total=len(dataset)):
            if mask[i] == 0:
                continue
            if row[col_1][turn] == row[col_2][turn]:
                output_mask[i] = 0
    
    return output_mask


def filter_length(dataset, tokenizer, columns, max_len, mask, turns = None):

    output_mask = mask

    for i, row in tqdm(enumerate(dataset), total=len(dataset)):
        if mask[i] == 0:
            continue
        for c, t in zip(columns, turns):
            if t:
                if len(tokenizer.apply_chat_template(row[c][:(row[t]+1)*2], tokenize=True, add_generation_prompt=False)) > max_len:
                    output_mask[i] = 0
                    break
            elif len(tokenizer.apply_chat_template(row[c], tokenize=True, add_generation_prompt=False)) > max_len:
                output_mask[i] = 0
                break
    
    return output_mask


def generate_token_mask(dataset, tokenizer, col, turns, max_len):

    # init
    token_masks = {}
    for t in turns:
        token_masks[col + f'_turn={t}_token'] = []
        token_masks[col + f'_turn={t}_mask'] = []

    for row in tqdm(dataset[col]):
        for t in turns:
            if t == -1:
                token = tokenizer.apply_chat_template(row, tokenize=True, add_generation_prompt=False, padding='max_length', max_length=max_len)
                mask = np.zeros(max_len).astype(int)
                mask[len(tokenizer.apply_chat_template(row[:-1], tokenize=True, add_generation_prompt=True)) : len(tokenizer.apply_chat_template(row, tokenize=True, add_generation_prompt=False))] = 1
            else:
                token = tokenizer.apply_chat_template(row[:(t+1)*2], tokenize=True, add_generation_prompt=False, padding='max_length', max_length=max_len)
                mask = np.zeros(max_len).astype(int)
                mask[len(tokenizer.apply_chat_template(row[:t*2+1], tokenize=True, add_generation_prompt=True)) : len(tokenizer.apply_chat_template(row[:(t+1)*2], tokenize=True, add_generation_prompt=False))] = 1
            token_masks[col + f'_turn={t}_token'].append(token)
            token_masks[col + f'_turn={t}_mask'].append(mask)

    return token_masks


def main(args):

    # init
    dataset_multi = load_dataset(args.input_repo, split='train')

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ================================filter====================================
    filter_mask = np.ones(len(dataset_multi))

    # filter reward with -99999
    print('filtering reward')
    filter_mask = filter_reward(dataset_multi, ['trajectory_reward', 'trajectory_sampled_h_from_sampled_len_reward'], -99999, filter_mask)
    print(f'after filtering: {filter_mask.sum()}')

    # filter same response
    print('filtering same response')
    filter_mask = filter_same_response(dataset_multi, 'trajectory', 'trajectory_sampled_h_from_sampled_len', 'sampled_h_from_sampled_len', filter_mask)
    print(f'after filtering: {filter_mask.sum()}')

    # filter length
    print('filtering length')
    filter_mask = filter_length(dataset_multi, tokenizer, ['trajectory', 'trajectory_sampled_h_from_sampled_len'], MAX_LEN, filter_mask, \
                                turns=[None, 'sampled_h_from_sampled_len'])
    print(f'after filtering: {filter_mask.sum()}')

    with open('temp.pkl', 'wb') as handle:
        pickle.dump(filter_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dataset_multi = dataset_multi.filter(lambda _, idx: filter_mask[idx] != 0, with_indices=True)

    # ===========================generate & mask================================

    # generate token and mask
    print('generating tokens and masks')
    token_masks = generate_token_mask(dataset_multi, tokenizer, 'trajectory', [0, 1, 2, 3, 4], MAX_LEN)
    for k, v in token_masks.items():
        dataset_multi = dataset_multi.add_column(k, v)
    token_masks = generate_token_mask(dataset_multi, tokenizer, 'trajectory_sampled_h_from_sampled_len', [0, 1, 2, 3, 4], MAX_LEN)
    for k, v in token_masks.items():
        dataset_multi = dataset_multi.add_column(k, v)

    # ==========================================================================

    traj_1 = 'trajectory'
    traj_2 = 'trajectory_sampled_h_from_sampled_len'
    reward_idx = 'sampled_len_from_5'
    turn_idx = 'sampled_h_from_sampled_len'

    data_dict = {'chosen' : [], \
                 'reject' : [], \
                 'chosen_token' : [], \
                 'reject_token' : [], \
                 'chosen_mask' : [], \
                 'reject_mask' : [], \
                 'chosen_reward' : [], \
                 'reject_reward' : []}

    for row in tqdm(dataset_multi):
        r1 = row[traj_1 + '_reward'][row[reward_idx]]
        r2 = row[traj_2 + '_reward'][row[reward_idx]]
        if r1 > r2:
            data_dict['chosen'].append(row[traj_1][:(row[reward_idx] + 1)*2])
            data_dict['reject'].append(row[traj_2][:(row[reward_idx] + 1)*2])
            data_dict['chosen_token'].append(row[f'{traj_1}_turn={row[turn_idx]}_token'])
            data_dict['reject_token'].append(row[f'{traj_2}_turn={row[turn_idx]}_token'])
            data_dict['chosen_mask'].append(row[f'{traj_1}_turn={row[turn_idx]}_mask'])
            data_dict['reject_mask'].append(row[f'{traj_2}_turn={row[turn_idx]}_mask'])
            data_dict['chosen_reward'].append(r1)
            data_dict['reject_reward'].append(r2)
        else:
            data_dict['chosen'].append(row[traj_2][:(row[reward_idx] + 1)*2])
            data_dict['reject'].append(row[traj_1][:(row[reward_idx] + 1)*2])
            data_dict['chosen_token'].append(row[f'{traj_2}_turn={row[turn_idx]}_token'])
            data_dict['reject_token'].append(row[f'{traj_1}_turn={row[turn_idx]}_token'])
            data_dict['chosen_mask'].append(row[f'{traj_2}_turn={row[turn_idx]}_mask'])
            data_dict['reject_mask'].append(row[f'{traj_1}_turn={row[turn_idx]}_mask'])
            data_dict['chosen_reward'].append(r2)
            data_dict['reject_reward'].append(r1)

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.train_test_split(test_size=500)
    dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    