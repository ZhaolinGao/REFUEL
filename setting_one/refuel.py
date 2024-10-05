import os
import random
import time
import math
from dataclasses import asdict, dataclass, field
from functools import partial
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
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset, DatasetDict
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
from torch.optim.lr_scheduler import LambdaLR
from huggingface_hub import snapshot_download
from datetime import timedelta


torch.set_printoptions(threshold=10_000)


@dataclass
class REFUELHParams:
    num_updates: tyro.conf.Suppress[int] = 1000
    whiten_rewards: bool = False
    shift_mean: bool = False
    eta: float = 1.0


@dataclass
class TaskHParams:
    query_dataset: str = ""
    cluster: str = ""
    total_length: int = 2048
    temperature: float = 0.8


@dataclass
class Args:
    # common args
    exp_name: str = "ultrainteract_refuel"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "multiturn_largebatch"
    """the wandb's project name"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    print_sample_output_freq: int = 200
    """How often to print sample output"""

    # optimizer args
    eps: float = 1e-8
    """the epsilon value for the optimizer"""
    lr: float = 3e-7
    """learning rate"""
    weight_decay: float = 1e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    warmup_ratio: float = 0.1
    """warmup ratio"""
    start_idx: int = 0
    """dataset start idx"""
    end_idx: int = -1
    """dataset end idx"""

    gradient_accumulation_steps: int = 32
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 1
    """per rank eval batch size"""
    total_episodes: int = 60000
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
    output_dir: str = ""
    """Where to save the model"""
    task: TaskHParams = field(default_factory=TaskHParams)
    refuel: REFUELHParams = field(default_factory=REFUELHParams)


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def gather_logprob(args, model, tokenizer, token, mask, device):

    token = token.long().to(device).unsqueeze(0)
    mask = mask.long().to(device).unsqueeze(0)

    attention_mask = token != tokenizer.pad_token_id
    input_ids = torch.masked_fill(token, ~attention_mask, tokenizer.eos_token_id)

    with torch.no_grad():
        output = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=True,
                 )
        logits = output.logits[:, :-1]
        logits /= args.task.temperature + 1e-7
        all_logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(all_logprob, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        logprob = (logprob * mask[:, 1:]).sum(-1).item()
        
    return logprob


def evaluate(args, policy, tokenizer, dataloader):

    device = policy.device
    loss, loss_eta_1, sign_align = [], [], []
    with torch.no_grad():
        for data in tqdm(dataloader):
            
            tokens = torch.cat((data["chosen_token"], data["reject_token"]), dim=0)
            logprobs = torch.cat((data["chosen_logprob"], data["reject_logprob"]), dim=0)
            masks = torch.cat((data["chosen_mask"], data["reject_mask"]), dim=0)

            attention_mask = tokens != tokenizer.pad_token_id
            input_ids = torch.masked_fill(tokens, ~attention_mask, tokenizer.eos_token_id)

            output = policy(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            logits = output.logits[:, :-1]
            logits /= args.task.temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(new_all_logprobs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            new_logprobs = (new_logprobs * masks[:, 1:]).sum(-1)

            ratio_logprob = new_logprobs - logprobs
            ratio_logprob = ratio_logprob[:args.per_device_eval_batch_size] - ratio_logprob[args.per_device_eval_batch_size:]
            sign_align.append((ratio_logprob > 0).float().mean().reshape(1))

            reg_diff = ratio_logprob - args.refuel.eta * (data["chosen_reward"] - data["reject_reward"])
            loss.append((reg_diff ** 2).mean().reshape(1))

    loss = torch.cat(loss)
    sign_align = torch.cat(sign_align)
    return {"val_loss" : loss, "sign_align" : sign_align}


if __name__ == '__main__':

    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    args.world_size = accelerator.num_processes
    args.batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.refuel.whiten_rewards:
        assert (args.local_batch_size >= 8), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"
    args.refuel.num_updates = args.total_episodes // args.batch_size

    # logging
    console = Console(force_terminal=True)
    accelerator.wait_for_everyone()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    accelerator.print("Wandb run name: ", run_name)
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
    policy = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
    disable_dropout_in_model(policy)

    # Prompt Collection Dataset
    recompute_log = False
    try:
        dataset = load_dataset(args.task.query_dataset + '_' + args.task.cluster, split='train')
        dataset = dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward", "chosen_logprob",
                                                        "reject_token", "reject_mask", "reject_reward", "reject_logprob"])
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataset = load_dataset(args.task.query_dataset + '_' + args.task.cluster, split='test')
        validation_dataset = validation_dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward", "chosen_logprob",
                                                                              "reject_token", "reject_mask", "reject_reward", "reject_logprob"])
    except:
        dataset = load_dataset(args.task.query_dataset, split='train')
        dataset = dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward",
                                                        "reject_token", "reject_mask", "reject_reward"])
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataset = load_dataset(args.task.query_dataset, split='test')
        validation_dataset = validation_dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward",
                                                                              "reject_token", "reject_mask", "reject_reward"])
        recompute_log = True

    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    if accelerator.is_main_process:
        pprint(policy.config)

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
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.refuel.num_updates * args.warmup_ratio * args.world_size), args.refuel.num_updates * args.world_size)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    policy, optimizer, _, scheduler = accelerator.prepare(policy, optimizer, temp_dataloader, scheduler)

    if recompute_log:
        accelerator.print('gathering validation logprob')
        logprob_0, logprob_1 = [], []
        for i in tqdm(range(len(validation_dataset))):
            logprob_0.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, validation_dataset[i]["chosen_token"], validation_dataset[i]["chosen_mask"], device))
            logprob_1.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, validation_dataset[i]["reject_token"], validation_dataset[i]["reject_mask"], device))
        validation_dataset = validation_dataset.add_column("chosen_logprob", logprob_0)
        validation_dataset = validation_dataset.add_column("reject_logprob", logprob_1)
        validation_dataset = validation_dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward", "chosen_logprob",
                                                                              "reject_token", "reject_mask", "reject_reward", "reject_logprob"])

        accelerator.print('gathering logprob')
        logprob_0, logprob_1 = [], []
        for i in tqdm(range(len(dataset))):
            logprob_0.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, dataset[i]["chosen_token"], dataset[i]["chosen_mask"], device))
            logprob_1.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, dataset[i]["reject_token"], dataset[i]["reject_mask"], device))
        dataset = dataset.add_column("chosen_logprob", logprob_0)
        dataset = dataset.add_column("reject_logprob", logprob_1)
        dataset = dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward", "chosen_logprob",
                                                        "reject_token", "reject_mask", "reject_reward", "reject_logprob"])

        if accelerator.is_main_process:
            temp = DatasetDict({
                "train" : dataset,
                "test"  : validation_dataset,
            })
            temp.push_to_hub(args.task.query_dataset + '_' + args.task.cluster)
        accelerator.wait_for_everyone()

    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
    dataloader = accelerator.prepare(dataloader)
    validation_dataloader = accelerator.prepare(validation_dataloader)
    def repeat_generator():
        while True:
            yield from dataloader
    iter_dataloader = iter(repeat_generator())

    accelerator.print("===training policy===")
    torch.manual_seed(local_seed)  # reset the local seed again
    global_step = 0
    start_time = time.time()

    kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    chosen_kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    reject_kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    ratio_stats = torch.zeros(args.gradient_accumulation_steps, device=device)

    policy.train()
    for update in range(1, args.refuel.num_updates + 1):

        # update parameters
        global_step += 1 * args.batch_size
        lrnow = optimizer.param_groups[0]["lr"]

        # save model
        if (update - 1) % args.print_sample_output_freq == 0: # !!!!!!! and update > 1
            eval_dict = evaluate(args, accelerator.unwrap_model(policy), tokenizer, validation_dataloader)
            writer.add_scalar("objective/validation_loss", accelerator.gather(eval_dict["val_loss"]).mean().item(), update)
            writer.add_scalar("objective/sign_align", accelerator.gather(eval_dict["sign_align"]).mean().item(), update)
            torch.cuda.empty_cache()

        # training
        data = next(iter_dataloader)

        gradient_accumulation_idx = 0
        for mini_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
            mini_batch_end = mini_batch_start + args.per_device_train_batch_size
            with accelerator.accumulate(policy):
                mb_chosen_token = data["chosen_token"][mini_batch_start : mini_batch_end]
                mb_chosen_mask = data["chosen_mask"][mini_batch_start : mini_batch_end]
                mb_chosen_reward = data["chosen_reward"][mini_batch_start : mini_batch_end]
                mb_chosen_logprob = data["chosen_logprob"][mini_batch_start : mini_batch_end]

                mb_reject_token = data["reject_token"][mini_batch_start : mini_batch_end]
                mb_reject_mask = data["reject_mask"][mini_batch_start : mini_batch_end]
                mb_reject_reward = data["reject_reward"][mini_batch_start : mini_batch_end]
                mb_reject_logprob = data["reject_logprob"][mini_batch_start : mini_batch_end]

                mb_tokens = torch.cat((mb_chosen_token, mb_reject_token), dim=0)
                mb_masks = torch.cat((mb_chosen_mask, mb_reject_mask), dim=0)
                mb_logprobs = torch.cat((mb_chosen_logprob, mb_reject_logprob), dim=0)

                attention_mask = mb_tokens != tokenizer.pad_token_id
                input_ids = torch.masked_fill(mb_tokens, ~attention_mask, tokenizer.eos_token_id)

                output = policy(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = output.logits[:, :-1]
                logits /= args.task.temperature + 1e-7
                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                new_logprobs = (new_logprobs * mb_masks[:, 1:]).sum(-1)

                if update == 1:
                    print(('logprobs:', new_logprobs, mb_logprobs))

                ratio_logprob = new_logprobs - mb_logprobs
                ratio_logprob = ratio_logprob[:args.per_device_train_batch_size] - ratio_logprob[args.per_device_train_batch_size:]

                reg_diff = ratio_logprob - args.refuel.eta * (mb_chosen_reward - mb_reject_reward)
                loss = (reg_diff ** 2).mean()

                # accelerator.print(loss)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    logprobs_diff = new_logprobs - mb_logprobs
                    ratio = torch.exp(logprobs_diff)
                    kl_stats[gradient_accumulation_idx] = logprobs_diff.mean()
                    chosen_kl_stats[gradient_accumulation_idx] = logprobs_diff[:args.per_device_train_batch_size].mean()
                    reject_kl_stats[gradient_accumulation_idx] = logprobs_diff[args.per_device_train_batch_size:].mean()
                    loss_stats[gradient_accumulation_idx] = loss
                    ratio_stats[gradient_accumulation_idx] = ratio.mean()
            gradient_accumulation_idx += 1

        del output, new_all_logprobs, new_logprobs
        torch.cuda.empty_cache()

        if accelerator.is_main_process:
            console.print(
                f"update",
                update,
                "kl_stats",
                kl_stats.mean().item(),
                "loss",
                loss_stats.mean().item(),
            )

        with torch.no_grad():
            writer.add_scalar("objective/kl", accelerator.gather(kl_stats).mean().item(), update)
            writer.add_scalar("objective/chosen_kl", accelerator.gather(chosen_kl_stats).mean().item(), update)
            writer.add_scalar("objective/reject_kl", accelerator.gather(reject_kl_stats).mean().item(), update)
            writer.add_scalar("npg/loss/policy", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("npg/loss/policy_avg", accelerator.gather(loss_stats).mean().item(), update)
            
            writer.add_scalar("npg/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("npg/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("npg/lr", lrnow, update)
            writer.add_scalar("npg/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("npg/eps", eps, update)
            accelerator.print("npg/eps", eps, update)
            torch.cuda.empty_cache()

    # save model
    eval_dict = evaluate(args, accelerator.unwrap_model(policy), tokenizer, validation_dataloader)
    writer.add_scalar("objective/validation_loss", accelerator.gather(eval_dict["val_loss"]).mean().item(), update)
    writer.add_scalar("objective/sign_align", accelerator.gather(eval_dict["sign_align"]).mean().item(), update)
    if args.output_dir:
        accelerator.wait_for_everyone()
        output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        accelerator.save_state(output_dir=output_dir)
        accelerator.wait_for_everyone()
    torch.cuda.empty_cache()