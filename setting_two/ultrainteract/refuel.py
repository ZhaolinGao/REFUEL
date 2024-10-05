import os
import random
import time
import math
import functools
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional, Tuple, Union
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
import deepspeed
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from huggingface_hub import snapshot_download
torch.set_printoptions(threshold=10_000)

QUERY_LENGTH = {1 : 1024, 2 : 768, 3 : 512, 4 : 256, 5 : 128}
RESPONSE_LENGTH= {1 : 1024, 2 : 768, 3 : 512, 4 : 512, 5 : 512}
TOTAL_LENGTH = 3200
MAX_RESPONSE_LENGTH = 1024
PRINT = True

@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    use_adaptive_kl: bool = False
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    kl_coef: float = 0.05


@dataclass
class REFUELHParams:
    num_updates: tyro.conf.Suppress[int] = 1000
    noptepochs: int = 4
    whiten_rewards: bool = False
    shift_mean: bool = False
    eta: float = 1.0


@dataclass
class TaskHParams:
    # Query params
    query_dataset: str = ""

    # Response params
    penalty_reward_value: int = -2
    temperature: float = 1.0
    num_turns: int = 5


@dataclass
class Args:
    # common args
    exp_name: str = "multiturn_chat_refuel"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "multiturn"
    """the wandb's project name"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    push_to_hub: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    deepspeed: bool = True
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 200
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-8
    """the epsilon value for the optimizer"""
    lr: float = 1e-7

    weight_decay: float = 1e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "linear" # might be worth with 
    warm_up_steps: int = 0

    gradient_accumulation_steps: int = 2
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 2
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 16
    """per rank eval batch size"""
    per_device_reward_batch_size: int = 16
    """per device reward batch size"""
    total_episodes: int = 1000000
    """The total number of episodes in the dataset"""

    # optional args filled while running
    world_size: Optional[int] = 4
    """The number of processes (GPUs) to use"""
    batch_size: Optional[int] = 512
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_batch_size: Optional[int] = 128
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""

    # other args
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    """the name of the pretrained model to use"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model: str = "sfairXC/FsfairX-LLaMA3-RM-v0.1"
    """the name of the trained reward model to use"""
    output_dir: str = ""
    """Where to save the model"""
    num_layers_unfrozen: int = 4
    """number of layers to train"""
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    refuel: REFUELHParams = field(default_factory=REFUELHParams)


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(tokenizer, responses):
    trunc_idxs = first_true_indices(responses == tokenizer.eos_token_id).unsqueeze(-1)

    new_size = [1] * (len(responses.size()) - 1) + [MAX_RESPONSE_LENGTH]
    idxs = torch.arange(MAX_RESPONSE_LENGTH, device=responses.device).view(*new_size)

    pad_size = list(responses.size())
    if pad_size[-1] != MAX_RESPONSE_LENGTH:
        pad_size[-1] = MAX_RESPONSE_LENGTH-pad_size[-1]
        postprocessed_responses = torch.cat([responses, torch.full(size=pad_size, fill_value=tokenizer.pad_token_id, dtype=responses.dtype, device=responses.device)], dim=-1)
    else:
        postprocessed_responses = responses
    postprocessed_responses = torch.masked_fill(postprocessed_responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def freeze_bottom_causal_layers(model, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""

    def hf_get_decoder_blocks(model: nn.Module):
        hidden_layers_attrs = (
            "h",
            "layers",
            "model.layers", # <--- for mistral
            "decoder.layers",
            "transformer.h",
            "transformer.blocks",
            "model.decoder.layers",
            "gpt_neox.layers",
            "decoder.block",
        )
        return findattr(model, hidden_layers_attrs)

    hidden_layers = hf_get_decoder_blocks(model)

    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
        hidden_layers_to_freeze += [model.get_input_embeddings(), model.get_output_embeddings()]
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
        hidden_layers_to_freeze += [model.get_input_embeddings()]
        if model.config.tie_word_embeddings:
            hidden_layers_to_freeze += [model.get_output_embeddings()]
    else:
        hidden_layers_to_freeze = []

    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)

def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
    `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
    `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_decoder(model: nn.Module) -> nn.Module:
    decoder_attrs = ("transformer", "model.decoder", "gpt_neox", "decoder")
    return findattr(model, decoder_attrs)


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "model.norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
    )
    return findattr(model, norm_attrs)


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    hidden_layers_attrs = (
        "h", "layers", "model.layers", "decoder.layers", "transformer.h", "transformer.blocks", "model.decoder.layers",
        "gpt_neox.layers", "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_lm_head(model: nn.Module) -> nn.Module:
    return model.get_output_embeddings()


def hf_get_hidden_size(config: PretrainedConfig) -> int:
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


def hf_get_num_hidden_layers(config: PretrainedConfig) -> int:
    num_hidden_layers_attrs = ("num_hidden_layers", "n_layer")
    return findattr(config, num_hidden_layers_attrs)


class ModelBranch(PreTrainedModel):
    """Implements the upper trunk of the pretrained reference model used
    when computing the PPO KL-divergence penalty.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        *,
        num_layers_unfrozen: int,
        frozen=True,
    ):
        """
        Args:
        base_model (transformers.PreTrainedModel): The pretrained model to extract upper trunk from
        num_layers_unfrozen (int): The number of trainable layers
        """

        config = base_model.config
        super().__init__(config)

        # The branch is defined by the last `num_layers_unfrozen` layers of the pretrained model

        decoder_blocks = hf_get_decoder_blocks(base_model)[-num_layers_unfrozen:]
        final_norm = hf_get_decoder_final_norm(base_model)
        lm_head = hf_get_lm_head(base_model)

        with deepspeed.zero.GatheredParameters(
            list(decoder_blocks.parameters()) + list(final_norm.parameters()) + list(lm_head.parameters()),
            modifier_rank=None,
        ):
            self.decoder_blocks = deepcopy(decoder_blocks)
            self.final_norm = deepcopy(final_norm)
            self.lm_head = deepcopy(lm_head)

        self.hidden_size = hf_get_hidden_size(self.config)
        self.model_parallel = False
        self.device_map = None
        self.last_device = None
        self.gradient_checkpointing = False

        # Freeze the entire branch
        if frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)


class LlamaBranch(ModelBranch):

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        past_seen_tokens = 0
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_shape: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _, seq_length = hidden_states.shape[:2]

        past_seen_tokens = 0

        device = hidden_states.device
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_length, device=device
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position)

        # decoder layers
        for decoder_layer in self.decoder_blocks:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits


@torch.no_grad()
def get_reward(
    reward_model,
    input_ids,
    attention_mask,
    tokenizer,
    reward_batch_size,
):
    # Remove BOS
    new_attention_mask = torch.masked_fill(attention_mask, input_ids == tokenizer.bos_token_id, 0)
    new_input_ids = torch.masked_fill(input_ids, input_ids == tokenizer.bos_token_id, tokenizer.eos_token_id)
    # input_ids[:, :query_tokens.shape[1]] = torch.masked_fill(input_ids[:, :query_tokens.shape[1]], ~attention_mask[:, :query_tokens.shape[1]], 0)
    out = []
    mbs = reward_batch_size
    for i in range(math.ceil(new_input_ids.shape[0] / mbs)):
        rewards = reward_model(
            input_ids=new_input_ids[i * mbs : (i + 1) * mbs],
            attention_mask=new_attention_mask[i * mbs : (i + 1) * mbs]
        ).logits
        out.append(rewards)
    return torch.cat(out)


# generate input ids and attention mask
@torch.no_grad()
def generate_helper(args, queries, responses, turns, tokenizer):

    batch_size, total_length = len(queries[0]), TOTAL_LENGTH
    input_ids = torch.full((batch_size, total_length), tokenizer.pad_token_id).to(queries[0].device)
    response_mask = torch.full((args.task.num_turns, batch_size, total_length), 0).to(queries[0].device)
    current_idx = [total_length for _ in range(batch_size)]

    for idx in range(len(responses)-1, -1, -1):

        # add prompt
        if len(queries) == len(responses) + 1 or idx < len(responses)-1:
            sequence_length = first_true_indices(queries[idx + 1] == tokenizer.pad_token_id)
            for b in range(batch_size):
                input_ids[b, current_idx[b] - sequence_length[b]: current_idx[b]] = queries[idx + 1][b, :sequence_length[b]]
                current_idx[b] -= sequence_length[b].item()

        # add response
        sequence_length = first_true_indices(responses[idx] == tokenizer.pad_token_id)
        for b in range(batch_size):
            input_ids[b, current_idx[b] - sequence_length[b]: current_idx[b]] = responses[idx][b, :sequence_length[b]]
            if responses[idx][b, sequence_length[b]-1] != tokenizer.eos_token_id:
                input_ids[b, current_idx[b] - 1] = tokenizer.eos_token_id
                response_mask[idx, b, current_idx[b] - sequence_length[b]: current_idx[b] - 1] = 1
            else:
                response_mask[idx, b, current_idx[b] - sequence_length[b]: current_idx[b]] = 1
            current_idx[b] -= sequence_length[b].item()

    # add first prompt
    for b in range(batch_size):
        # print(current_idx[b], QUERY_LENGTH[turns[b].item()], queries.shape)
        input_ids[b, current_idx[b] - min(QUERY_LENGTH[turns[b].item()], current_idx[b]): current_idx[b]] = queries[0][b][-min(QUERY_LENGTH[turns[b].item()], current_idx[b]):]

    attention_mask = input_ids != tokenizer.pad_token_id
    input_ids = torch.masked_fill(input_ids, ~attention_mask, tokenizer.eos_token_id)
    
    return input_ids, attention_mask, response_mask


@torch.no_grad()
def generate(args, lm_backbone, data, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""

    # init Turn * Batch * Length
    queries, responses = [], []

    # generate
    for i in range(args.task.num_turns):
        queries.append(data[f"llama_prompt_token_turn_{i}"])
        responses.append(data[f"llama_response_token_turn_{i}"])

    queries = torch.stack(queries, dim=0)
    responses = torch.stack(responses, dim=0)

    turns = data['num_turn']
    start_gen_turn = torch.zeros_like(data['num_turn'])

    return generate_from_turn(
                args,
                lm_backbone,
                queries, 
                responses,
                turns,
                start_gen_turn,
                tokenizer,
                generation_config,
            )


@torch.no_grad()
def generate_from_turn(args, lm_backbone, queries, responses, turns, sample_turns, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""

    # init Turn * Batch * Length
    sec_responses = responses

    # generate
    for b in range(len(sample_turns)):
        for t in range(sample_turns[b].item(), turns[b].item()):
            input_ids, attention_mask, _ = generate_helper(args, queries[0:t+1, b:b+1, :], 
                                                           responses[0:t, b:b+1, :], turns, tokenizer)

            output = lm_backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config[turns[b].item()],
                return_dict_in_generate=True,
            )
            output = truncate_response(tokenizer, output.sequences[:, input_ids.shape[1]:])
            sec_responses[t, b:b+1, :] = output

    return queries, sec_responses


@dataclass
class EvalStorage:
    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)


def evaluate(args: Args, reward_model, policy, tokenizer, rm_tokenizer, dataloader, generation_config, sampling=True):

    eval_storage = EvalStorage()
    device = policy.device
    with torch.no_grad():
        for data in tqdm(dataloader):
            # query = data["llama_prompt_token_turn_0"].to(device)
            # context_length = query.shape[1]
            queries, responses = generate(
                args,
                policy,
                data,
                tokenizer,
                generation_config,
            )
            input_ids, attention_mask, _ = generate_helper(args, queries, responses, data['num_turn'], tokenizer)
            score = get_reward(
                reward_model, input_ids, attention_mask, tokenizer, args.per_device_reward_batch_size
            )
            torch.cuda.empty_cache()

            eval_storage.query.extend(data["llama_dialogue"])
            eval_storage.postprocessed_response.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=False))
            eval_storage.score.append(score)

            if sampling:
                break

    eval_score = torch.cat(eval_storage.score).flatten().float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "scores": gather_object(eval_score),
        }
    )
    return eval_storage, eval_df


if __name__ == '__main__':

    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    args.world_size = accelerator.num_processes
    args.batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.refuel.whiten_rewards:
        assert (args.local_batch_size >= 8), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"
    args.refuel.num_updates = args.total_episodes // args.batch_size

    # logging
    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}__{args.output_dir.split('/')[1]}"
    print("Wandb run name: ", run_name)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z, max_bins: None
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    torch.backends.cudnn.deterministic = True

    # policy
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        padding_side='right',
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    rm_tokenizer = AutoTokenizer.from_pretrained(
        "sfairXC/FsfairX-LLaMA3-RM-v0.1", 
        padding_side='right',
        trust_remote_code=True,
    )
    #rm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_config = AutoConfig.from_pretrained(args.base_model, attn_implementation="eager") # Dropout is already disabled for OpenChat
    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=model_config,
        trust_remote_code=True,
    )
    ref_policy = LlamaBranch(policy, num_layers_unfrozen=args.num_layers_unfrozen)

    freeze_bottom_causal_layers(policy, num_layers_unfrozen=args.num_layers_unfrozen)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    policy.generation_config.do_sample = True
    
    #reward_model = AutoModel.from_pretrained(args.reward_model, trust_remote_code=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "sfairXC/FsfairX-LLaMA3-RM-v0.1",
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    reward_model.eval().requires_grad_(False)

    # Ultrafeedback Dataset
    dataset = load_dataset(args.task.query_dataset, split='train')
    dataset = dataset.with_format("torch", columns=["num_turn", "llama_prompt_token_turn_0", "llama_response_token_turn_0",
                                                                "llama_prompt_token_turn_1", "llama_response_token_turn_1",
                                                                "llama_prompt_token_turn_2", "llama_response_token_turn_2",
                                                                "llama_prompt_token_turn_3", "llama_response_token_turn_3",
                                                                "llama_prompt_token_turn_4", "llama_response_token_turn_4"])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)

    validation_dataset = load_dataset(args.task.query_dataset, split="test")
    validation_dataset = validation_dataset.with_format("torch", columns=["llama_dialogue", "num_turn", 
                                                                          "llama_prompt_token_turn_0", "llama_response_token_turn_0",
                                                                          "llama_prompt_token_turn_1", "llama_response_token_turn_1",
                                                                          "llama_prompt_token_turn_2", "llama_response_token_turn_2",
                                                                          "llama_prompt_token_turn_3", "llama_response_token_turn_3",
                                                                          "llama_prompt_token_turn_4", "llama_response_token_turn_4"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size)

    if accelerator.is_main_process:
        pprint(model_config)
        pprint(reward_model.config)

    if args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            policy.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=args.eps,
            weight_decay=args.weight_decay
        )

    kl_ctl = AdaptiveKLController(args.reward.kl_coef, hparams=args.reward.adaptive_kl)
    generation_config, validation_generation_config = {}, {}
    for k in RESPONSE_LENGTH.keys():
        generation_config[k] = GenerationConfig(
            min_new_tokens=RESPONSE_LENGTH[k],
            max_new_tokens=RESPONSE_LENGTH[k],
            temperature=(args.task.temperature + 1e-7),
            top_p=1.0,
            top_k=0,
            do_sample=True,
        )
        validation_generation_config[k] = GenerationConfig(
            max_new_tokens=RESPONSE_LENGTH[k],
            min_new_tokens=RESPONSE_LENGTH[k],
            temperature=0.01 + 1e-7,
            top_p=1.0,
            top_k=0,
            do_sample=True,
        )

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    validation_dataloader = accelerator.prepare(validation_dataloader)
    def repeat_generator():
        while True:
            yield from dataloader
    iter_dataloader = iter(repeat_generator())
    torch.manual_seed(local_seed)  # reset the local seed again

    if args.deepspeed:
        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size

        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if args.offload:
            deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
            eval_ds_config["zero_optimization"] = {
                "stage": 3,
                "stage3_param_persistence_threshold": 1e4,
                "offload_param": {"device": "cpu"},
            }
        accelerator.print(f"{eval_ds_config=}")
        reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        reward_model.eval()
        ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
        ref_policy.eval()
    else:
        reward_model = reward_model.to(device)
        ref_policy = ref_policy.to(device)
    
    accelerator.print("===training policy===")
    global_step = 0
    start_time = time.time()
    stats_shape = (args.refuel.noptepochs, args.gradient_accumulation_steps)

    approxkl_stats = torch.zeros(stats_shape, device=device)
    loss_stats = torch.zeros((args.refuel.noptepochs, args.gradient_accumulation_steps), device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)

    policy.train()
    for update in range(1, args.refuel.num_updates + 1):
        global_step += 1 * args.batch_size
        frac = 1.0 - (update - 1.0) / args.refuel.num_updates
        lrnow = frac * args.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            print("sampling evaluation")
            eval_storage, eval_df = evaluate(
                args,
                reward_model,
                accelerator.unwrap_model(policy),
                tokenizer,
                rm_tokenizer,
                validation_dataloader,
                validation_generation_config,
            )
            validation_score = eval_storage.score[0]
            if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0:
                eval_storage, eval_df = evaluate(
                    args,
                    reward_model,
                    accelerator.unwrap_model(policy),
                    tokenizer,
                    rm_tokenizer,
                    validation_dataloader,
                    validation_generation_config,
                    sampling=False
                )
                if accelerator.is_main_process:
                    eval_df.to_csv(f"runs/{run_name}/table_{global_step}.csv")
                    if args.track:
                        wandb.log({"samples/query_responses": wandb.Table(dataframe=eval_df)}, step=update)
                    else:
                        try:
                            print_rich_table(f"Sample Output at Step {update}", eval_df[:1], console)
                        except Exception as e:
                            print(e)
                # save model
                if args.output_dir:
                    output_dir = os.path.join(args.output_dir, run_name, str(update))
                    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                    time_tensor = torch.tensor([int(time.time())], device=device)
                    time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
                    repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
                    repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                        if args.push_to_hub:
                            tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

                    unwrapped: PreTrainedModel = accelerator.unwrap_model(policy)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        print('saved')
                        unwrapped.save_pretrained(
                            output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(unwrapped),
                            safe_serialization=False,
                            repo_id=repo_id,
                        )
                        if args.push_to_hub:
                            unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)
            del eval_storage, eval_df
            torch.cuda.empty_cache()

            print("generating rollouts")

            input_ids = []
            attention_masks = []
            response_masks = []
            logprobs = []
            ref_logprobs = []
            scores = []

            turns = data["num_turn"]
            sample_turns = []
            for i in range(len(turns)):
                sample_turns.append(torch.randint(high=turns[i].item(), size=(1,)))
            sample_turns = torch.cat(sample_turns).to(device)

            ### first generation
            queries, responses = generate(
                args,
                accelerator.unwrap_model(policy),
                data,
                tokenizer,
                generation_config,
            )
            input_id, attention_mask, response_mask = generate_helper(args, queries, responses, turns, tokenizer)

            # print(1, input_id, tokenizer.batch_decode(input_id, skip_special_tokens=False))

            # Batch * Length
            response_mask = torch.stack([response_mask[t, b, :] for b, t in enumerate(sample_turns)], 0)

            # print(2, input_id * response_mask, tokenizer.batch_decode(input_id * response_mask, skip_special_tokens=False))
            
            # policy log_probs
            output = accelerator.unwrap_model(policy)(
                            input_ids=input_id, 
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_hidden_states=True,
                        )

            # for ref policy
            input_hidden_states = output.hidden_states[-(args.num_layers_unfrozen + 1)]
            output_shape = output.hidden_states[-1].size()

            logits = output.logits[:, :-1]
            logits /= args.task.temperature + 1e-7
            all_logprob = F.log_softmax(logits, dim=-1)
            logprob = torch.gather(all_logprob, 2, input_id[:, 1:].unsqueeze(-1)).squeeze(-1)
            
            del output, logits, all_logprob
            torch.cuda.empty_cache()

            # reference log_probs
            ref_logits = ref_policy(
                                hidden_states=input_hidden_states,
                                output_shape=output_shape,
                                attention_mask=attention_mask,
                            )
            ref_logits = ref_logits[:, :-1]
            ref_logits /= args.task.temperature + 1e-7
            ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            ref_logprob = torch.gather(ref_all_logprob, 2, input_id[:, 1:].unsqueeze(-1)).squeeze(-1)
            del ref_logits, ref_all_logprob, input_hidden_states, output_shape
            torch.cuda.empty_cache()

            score = get_reward(
                reward_model, input_id, attention_mask, tokenizer, args.per_device_reward_batch_size
            )

            ### second generation
            queries, responses = generate_from_turn(
                args,
                accelerator.unwrap_model(policy),
                queries, 
                responses,
                turns,
                sample_turns,
                tokenizer,
                generation_config,
            )
            sec_input_id, sec_attention_mask, sec_response_mask = generate_helper(args, queries, responses, turns, tokenizer)

            # print(turns, sample_turns)

            # print(3, sec_input_id, tokenizer.batch_decode(sec_input_id, skip_special_tokens=False))

            # Batch * Length
            sec_response_mask = torch.stack([sec_response_mask[t, b, :] for b, t in enumerate(sample_turns)], 0)

            # print(4, sec_input_id * sec_response_mask, tokenizer.batch_decode(sec_input_id * sec_response_mask, skip_special_tokens=False))
            
            # policy log_probs
            output = accelerator.unwrap_model(policy)(
                            input_ids=sec_input_id, 
                            attention_mask=sec_attention_mask,
                            return_dict=True,
                            output_hidden_states=True,
                        )

            # for ref policy
            input_hidden_states = output.hidden_states[-(args.num_layers_unfrozen + 1)]
            output_shape = output.hidden_states[-1].size()

            logits = output.logits[:, :-1]
            logits /= args.task.temperature + 1e-7
            all_logprob = F.log_softmax(logits, dim=-1)
            sec_logprob = torch.gather(all_logprob, 2, sec_input_id[:, 1:].unsqueeze(-1)).squeeze(-1)
            
            del output, logits, all_logprob
            torch.cuda.empty_cache()

            # reference log_probs
            ref_logits = ref_policy(
                                hidden_states=input_hidden_states,
                                output_shape=output_shape,
                                attention_mask=sec_attention_mask,
                            )
            ref_logits = ref_logits[:, :-1]
            ref_logits /= args.task.temperature + 1e-7
            ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            sec_ref_logprob = torch.gather(ref_all_logprob, 2, sec_input_id[:, 1:].unsqueeze(-1)).squeeze(-1)
            del ref_logits, ref_all_logprob, input_hidden_states, output_shape
            torch.cuda.empty_cache()

            sec_score = get_reward(
                reward_model, sec_input_id, sec_attention_mask, tokenizer, args.per_device_reward_batch_size
            )

            input_ids.append(torch.stack([input_id, sec_input_id], 1))
            attention_masks.append(torch.stack([attention_mask, sec_attention_mask], 1))
            response_masks.append(torch.stack([response_mask, sec_response_mask], 1))
            logprobs.append(torch.stack([logprob, sec_logprob], 1))
            ref_logprobs.append(torch.stack([ref_logprob, sec_ref_logprob], 1))
            scores.append(torch.stack([score, sec_score], 1))

            input_ids = torch.cat(input_ids, 0).flatten(end_dim=1)
            attention_masks = torch.cat(attention_masks, 0).flatten(end_dim=1)
            response_masks = torch.cat(response_masks, 0).flatten(end_dim=1)
            logprobs = torch.cat(logprobs, 0).flatten(end_dim=1)
            ref_logprobs = torch.cat(ref_logprobs, 0).flatten(end_dim=1)
            scores = torch.cat(scores, 0).flatten()
            del (input_id, sec_input_id, attention_mask, sec_attention_mask, response_mask,\
                 logprob, sec_logprob, ref_logprob, sec_ref_logprob, score, sec_score)
            torch.cuda.empty_cache()

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            contain_pad_token = torch.any(input_ids * response_masks == tokenizer.eos_token_id, dim=-1)
            scores = torch.where(contain_pad_token, scores, torch.full_like(scores, args.task.penalty_reward_value))

            if PRINT:
                print(turns)
                print(sample_turns)
                print('1', tokenizer.batch_decode(input_ids, skip_special_tokens=False))
                print('2', tokenizer.batch_decode(input_ids * attention_masks, skip_special_tokens=False))
                print('3', tokenizer.batch_decode(input_ids * response_masks, skip_special_tokens=False))
                print(contain_pad_token)
                print(scores)
                PRINT = False

            # 4. cumulative logprob
            logprobs = (logprobs * response_masks[:, 1:]).sum(-1)
            ref_logprobs = (ref_logprobs * response_masks[:, 1:]).sum(-1)

            # 5. kl reward and normalization
            kl = logprobs - ref_logprobs
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward + scores
            if args.refuel.whiten_rewards:
                rewards = whiten(rewards, args.refuel.shift_mean)

            accelerator.print("rewards without kl====", scores)
            accelerator.print("rewards with kl====", rewards)
            if accelerator.is_main_process:
                console.print(
                    f"mean_kl",
                    kl.mean().item(),
                    "scores",
                    scores.mean().item(),
                )
            del ref_logprobs
            torch.cuda.empty_cache()

        # Do multiple epochs of refuel training, with a fresh random shuffle in each epoch
        for refuel_epoch_idx in range(args.refuel.noptepochs):
            local_batch_idxs = np.random.permutation(args.local_batch_size)
            gradient_accumulation_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
                mini_batch_end = mini_batch_start + args.per_device_train_batch_size
                mini_batch_inds = local_batch_idxs[mini_batch_start:mini_batch_end] * 2
                mini_batch_inds = np.append(mini_batch_inds, mini_batch_inds + 1)
                with accelerator.accumulate(policy):
                    mb_input_ids = input_ids[mini_batch_inds]
                    mb_attention_masks = attention_masks[mini_batch_inds]
                    mb_response_masks = response_masks[mini_batch_inds]
                    mb_logprobs = logprobs[mini_batch_inds]
                    mb_rewards = rewards[mini_batch_inds]

                    output = policy(
                        input_ids=mb_input_ids, 
                        attention_mask=mb_attention_masks,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    logits = output.logits[:, :-1]
                    logits /= args.task.temperature + 1e-7
                    new_all_logprobs = F.log_softmax(logits, dim=-1)
                    new_logprobs = torch.gather(new_all_logprobs, 2, mb_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    new_logprobs = (new_logprobs * mb_response_masks[:, 1:]).sum(-1)

                    ratio_logprob = new_logprobs - mb_logprobs
                    ratio_logprob = ratio_logprob[:args.per_device_train_batch_size] - ratio_logprob[args.per_device_train_batch_size:]
                    reg_diff = ratio_logprob - args.refuel.eta * (mb_rewards[:args.per_device_train_batch_size] - mb_rewards[args.per_device_train_batch_size:])
                    loss = (reg_diff ** 2).mean()

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    with torch.no_grad():
                        y = args.refuel.eta * (mb_rewards[:args.per_device_train_batch_size] - mb_rewards[args.per_device_train_batch_size:])
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        approxkl_stats[refuel_epoch_idx, gradient_accumulation_idx] = approxkl
                        loss_stats[refuel_epoch_idx, gradient_accumulation_idx] = loss
                        entropy_stats[refuel_epoch_idx, gradient_accumulation_idx] = entropy.mean()
                        ratio_stats[refuel_epoch_idx, gradient_accumulation_idx] = ratio.mean()
                gradient_accumulation_idx += 1
            if accelerator.is_main_process:
                console.print(
                    f"refuel_epoch_idx",
                    refuel_epoch_idx,
                    "approxkl",
                    approxkl_stats[refuel_epoch_idx].mean().item(),
                    "loss",
                    loss_stats[refuel_epoch_idx].mean().item(),
                )
                
        with torch.no_grad():
            mean_kl = kl.mean()
            mean_entropy = -logprobs.mean()
            mean_non_score_reward = non_score_reward.mean()
            mean_generation_length = response_masks.sum(-1).float().mean()
            avg_length = sec_response_mask.sum(-1).float().mean()
            writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar("objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update)
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("objective/validation_score", accelerator.gather(validation_score.mean()).mean().item(), update)
            writer.add_scalar("objective/average_length", accelerator.gather(avg_length).mean().item(), update)
            writer.add_histogram("objective/scores_his", accelerator.gather(scores).cpu().float().numpy().flatten(), update, max_bins=64)
            writer.add_histogram("objective/validation_scores_his", accelerator.gather(validation_score).cpu().float().numpy().flatten(), update, max_bins=64)
            writer.add_scalar("npg/loss/policy", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("npg/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update)
            writer.add_scalar("npg/policy/approxkl", accelerator.gather(approxkl).mean().item(), update)

            writer.add_scalar("npg/policy/initial_loss", accelerator.gather(loss_stats[0]).mean().item(), update)
            writer.add_scalar("npg/policy/final_loss", accelerator.gather(loss_stats[-1]).mean().item(), update)
            writer.add_scalar("npg/policy/delta_loss", accelerator.gather(loss_stats[-1] - loss_stats[0]).mean().item(), update)
            
            writer.add_scalar("npg/policy/mean_gen_len", accelerator.gather(mean_generation_length).mean().item(), update)
            writer.add_scalar("npg/policy/approxkl_avg", accelerator.gather(approxkl_stats).mean().item(), update)
            writer.add_scalar("npg/loss/policy_avg", accelerator.gather(loss_stats).mean().item(), update)
            writer.add_scalar("npg/policy/entropy_avg", accelerator.gather(entropy_stats).mean().item(), update)
            writer.add_scalar("npg/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("npg/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("npg/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("npg/lr", lrnow, update)
            writer.add_scalar("npg/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("npg/eps", eps, update)
            accelerator.print("npg/eps", eps, update)
            if args.reward.use_adaptive_kl:
                kl_ctl.update(mean_kl.item(), args.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
            torch.cuda.empty_cache()

    if args.run_eval:
        eval_storage, eval_df = evaluate(
            args,
            reward_model,
            accelerator.unwrap_model(policy),
            tokenizer,
            rm_tokenizer,
            validation_dataloader,
            validation_generation_config,
            sampling=False
        )
        if accelerator.is_main_process:
            eval_df.to_csv(f"runs/{run_name}/table.csv")
            if args.track:
                wandb.log({"samples/query_responses": wandb.Table(dataframe=eval_df)}, step=update)

    # save model
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, run_name, str(update))
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir, repo_id=repo_id)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

        unwrapped: PreTrainedModel = accelerator.unwrap_model(policy)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(unwrapped),
                safe_serialization=False,
                repo_id=repo_id,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)
