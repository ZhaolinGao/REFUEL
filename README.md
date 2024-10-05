# Regressing the Relative Future: Efficient Policy Optimization for Multi-turn RLHF

Zhaolin Gao, Wenhao Zhan, Jonathan D. Chang, Gokul Swamy, Kiante Brantley, Jason D. Lee, Wen Sun.

![front page](./figs/refuel_ffig.png)

## Environment

```
torch>=2.1.0
transformers>=4.34
accelerate>=0.23
peft==0.6.2
bitsandbytes>=0.41.1
deepspeed>=0.10.3
vllm
tyro
scipy
rouge
shortuuid
jsonlines
rich
wandb
tensorboard
pandas
evaluate
```

## Setting One

#### Dataset Generation

In this setting, at each iteration, we first generate the dialogues for the entire [UltraInteract](https://huggingface.co/datasets/openbmb/UltraInteract_pair) dataset using our policy as the assistant and [Llama-3.1-70B-it](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) as the user. We use [Llama-3-8B-it](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) as our initial policy. You can directly use our processed dataset or generate from scratch:

1. We first generate the dialogues for the entire dataset
```
python ./setting_one/generate.py --model POLICY --output_dir OUTPUT_DIR --output_repo OUTPUT_REPO
```
You can also set ```num_data``` as a small number to test out the generation process.

2. We generate the rewards for all the dialogues using the [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) as the reward model.
```
python ./setting_one/rank.py --prompts INPUT_REPO --output_repo OUTPUT_REPO
```
We assign a -99999 for trajectories that does not have a valid reward. ```INPUT_REPO``` is the ```OUTPUT_REPO``` from step 1.

3. The dataset go through a rigorous filtering process. We filter out the dialogues in the dataset that are longer than 2048 tokens, have the same set of responses, and do not produce a valid reward score. We tokenize the dialogue and generate a mask for each dialogue.
```
python ./setting_one/tokenize_masks.py --input_repo INPUT_REPO --output_repo OUTPUT_REPO
```
```INPUT_REPO``` is the ```OUTPUT_REPO``` from step 2.

#### Training
Now, we can train our policy by running:
```
accelerate launch \
    --config_file accelerate_cfgs/ds_config2.yaml \
    --num_processes 8 \
    ./setting_one/refuel.py \
        --task.query_dataset DATASET_REPO \
        --task.cluster CLUSTER \
        --task.total_length 2048 \
        --task.temperature 0.8 \
        --lr 3e-7 \
        --rebel.eta 1e3 \
        --warmup_ratio 0.1 \
        --total_episodes 64000 \
        --output_dir OUTPUT_DIR \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --per_device_eval_batch_size 1 \
        --print_sample_output_freq 100 \
        --base_model PREV_POLICY
```
```task.query_dataset```: the repo of the generated dataset.

```task.cluster```: cluster name. We discover that different GPU/CPU/CUDA configurations could result in different logprobs. ```task.cluster``` allows us to recompute the logprobs automatically on a new cluster.

```output_dir```: local save directory.

```base_model```: the policy from the previous iteration. At the first iteration, we use ```meta-llama/Meta-Llama-3-8B-Instruct```.

#### Datasets and Models

Below we include our trained models and processed datasets, as well as their winrate w.r.t. the initial policy ```meta-llama/Meta-Llama-3-8B-Instruct```. REFUEL outperforms Llama-3.1-70B-it on dialogues with more than three turns.

<table>
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">Dataset</th>
    <th colspan="6">Winrate at Turn</th>
  </tr>
  <tr>
    <th>h = 1</th>
    <th>h = 2</th>
    <th>h = 3</th>
    <th>h = 4</th>
    <th>H = 5</th>
    <th>avg</th>
  </tr>
  <tr>
    <td>Llama-3.1-70B-it</td>
    <td> N/A </td>
    <td>70.4</td>
    <td>66.4</td>
    <td>61.0</td>
    <td>53.0</td>
    <td>55.4</td>
    <td>61.24</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Cornell-AGI/REFUEL-Llama-3-Armo-iter_1">REFUEL (iter 1)</a></td>
    <td><a href="https://huggingface.co/datasets/Cornell-AGI/REFUEL-Ultrainteract-Llama-3-Armo-iter_1">REFUEL-Ultrainteract-Llama-3-Armo-iter_1</a></td>
    <td>54.6</td>
    <td>53.6</td>
    <td>57.8</td>
    <td>56.2</td>
    <td>59.4</td>
    <td>56.32</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Cornell-AGI/REFUEL-Llama-3-Armo-iter_2">REFUEL (iter 2)</a></td>
    <td><a href="https://huggingface.co/datasets/Cornell-AGI/REFUEL-Ultrainteract-Llama-3-Armo-iter_2">REFUEL-Ultrainteract-Llama-3-Armo-iter_2</a></td>
    <td>55.2</td>
    <td>53.4</td>
    <td>58.8</td>
    <td>57.2</td>
    <td>58.6</td>
    <td>56.64</td>
  </tr>
</table>

## Setting Two

#### Anthropic HH

First, we process the [HH](https://huggingface.co/datasets/trl-internal-testing/hh-rlhf-trl-style) dataset by filtering out dialogues with more than 5 turns, prompts more than 128 tokens, responses with more than 512 tokens.
```
python ./setting_two/preprocess_hh.py
```
The processed dataset is available at [REFUEL-hh-setting-two](https://huggingface.co/datasets/Cornell-AGI/REFUEL-hh-setting-two).

Then, we train the Llama-3-8B-it with reward model [FsfairX](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) by running:
```
accelerate launch \
    --config_file accelerate_cfgs/deepspeed_config.yaml \
    --num_processes 8 \
    ./setting_two/anthropic_hh/refuel.py \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --task.query_dataset DATASET_REPO \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 1 \
        --lr 3e-7 \
        --eps 1e-8 \
        --weight_decay 1e-6 \
        --reward.kl_coef 0.05 \
        --rebel.eta 1.0 \
        --output_dir OUTPUT_DIR \
        --task.penalty_reward_value -10 \
        --print_sample_output_freq 200 \
        --task.response_length 512 \
        --offload
```
```task.query_dataset```: the repo of the processed dataset.

```output_dir```: local save directory.

#### Ultrainteract

First, we process the [UltraInteract](https://huggingface.co/datasets/openbmb/UltraInteract_pair)  dataset by filtering out dialogues with more than 5 turns, and prompts and responses that exceed the length in Table 5 of the paper.
```
python ./setting_two/preprocess_ultrainteract_diff_len.py
```
The processed dataset is available at [REFUEL-UltraInteract-setting-two](https://huggingface.co/datasets/Cornell-AGI/REFUEL-UltraInteract-setting-two).

Then, we train the Llama-3-8B-it with reward model [FsfairX](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) by running:

```
accelerate launch \
    --config_file accelerate_cfgs/deepspeed_config.yaml \
    --num_processes 8 \
    ./setting_two/ultrainteract/refuel.py \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --task.query_dataset DATASET_REPO \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 1 \
        --wandb_project_name multiturn \
        --lr 3e-7 \
        --eps 1e-8 \
        --weight_decay 1e-6 \
        --reward.kl_coef 0 \
        --rebel.eta 1.0 \
        --output_dir OUTPUT_DIR \
        --task.penalty_reward_value -4 \
        --print_sample_output_freq 200 \
        --offload
```
```task.query_dataset```: the repo of the processed dataset.

```output_dir```: local save directory.