from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
from trl import DPOConfig, DPOTrainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import os
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
import torch.nn as nn
import time,math
from tqdm import tqdm
m_path = "./"
train_dataset_path = "train.jsonl"
eval_dataset_path = "eval.jsonl"
### another eval dataset
cross_entr_loss_dataset = "eval_2_cross_entr_loss.jsonl"

model_max_length = 2000
evaluate_before_train = True
use_lora = True
lora_r = 30
lora_alpha = 16
lora_dropout = 0.05
lora_bias = "none"
lora_target_modules = [
    "down_proj",
    "gate_up_proj",
    "o_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
modules_to_save = ["embed_tokens", "lm_head"]
q_lora = False
training_args = DPOConfig(
    output_dir="/data/cvx-coder/outputmodels2",
    num_train_epochs=20,
    bf16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    eval_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=3,
    learning_rate=1e-4,
    weight_decay=0.1,
    adam_beta2=0.95,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    log_level="info",
    logging_dir="/data/cvx-coder/outputmodels2/logs",
    logging_strategy="steps",
    logging_steps=1,
    report_to="tensorboard",
    gradient_checkpointing=True,
    overwrite_output_dir=True,
    metric_for_best_model="eval_train_loss",
    load_best_model_at_end=True,
)

model = AutoModelForCausalLM.from_pretrained(
    m_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
if use_lora:
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,  # This argument serves for adding new tokens.
    )
    if q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

ref_model = AutoModelForCausalLM.from_pretrained(
    m_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    m_path,
    model_max_length=model_max_length,
    trust_remote_code=True,
    add_bos_token=False,  # 没有<s>, no <s> 
)

# ##---------------- Phi3 专用配置
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'
# ##---------------- Phi3 专用配置

def templating(example, tokenizer):
    chosen = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]

    prompt = tokenizer.apply_chat_template(
        chosen[:-1], tokenize=False, add_generation_prompt=False
    )

    chosen = tokenizer.apply_chat_template(
        chosen, tokenize=False, add_generation_prompt=False
    )[len(prompt) :]

    rejected = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["rejected"]},
    ]
    rejected = tokenizer.apply_chat_template(
        rejected, tokenize=False, add_generation_prompt=False
    )[len(prompt) :]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


train_dataset = load_dataset(
    os.path.dirname(train_dataset_path),
    data_files=train_dataset_path,
    trust_remote_code=True,
    split="train",
)  # train[:15%]
train_dataset = train_dataset.map(
    templating,
    fn_kwargs={
        "tokenizer": tokenizer,
    },
)

eval_dataset = load_dataset(
    os.path.dirname(eval_dataset_path),
    data_files=eval_dataset_path,
    trust_remote_code=True,
)  # train[:15%]
eval_dataset = eval_dataset.map(
    templating,
    fn_kwargs={
        "tokenizer": tokenizer,
    },
)

IGNORE_TOKEN_ID = -100

def tokenize_function_all(item, tokenizer, max_len):

    full = tokenizer.apply_chat_template(
        item["messages"], tokenize=False, add_generation_prompt=False
    )

    it = tokenizer(full, truncation=False, padding=False)
    #    tokenizer.eos_token_id
    it["input_ids"] = it["input_ids"] + [tokenizer.pad_token_id]
    it = it["input_ids"][:max_len]
    useful_len = len(it)
    useful_seq = it
    it = it + [tokenizer.pad_token_id] * (max_len - useful_len)
    # if useful_len>1400:
    #     print(useful_len)
    full_text = {}
    full_text["input_ids"] = it
    full_text["attention_mask"] = useful_len * [1] + [0] * (max_len - useful_len)
    full_text["label"] = useful_seq + [IGNORE_TOKEN_ID] * (max_len - useful_len)
    full_text["labels"] = useful_seq + [IGNORE_TOKEN_ID] * (max_len - useful_len)

    return full_text


cross_entr_loss_dataset = load_dataset(
    os.path.dirname(cross_entr_loss_dataset),
    data_files=cross_entr_loss_dataset,
    trust_remote_code=True,
    split="train",
)  # train[:15%]
cross_entr_loss_dataset = cross_entr_loss_dataset.map(
    tokenize_function_all,
    fn_kwargs={"tokenizer": tokenizer, "max_len": model_max_length},
)


class more_eval_loss_trainer(DPOTrainer):

    def __init__(self, cross_entr_loss_dataset, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cross_entr_loss_dataset = cross_entr_loss_dataset

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
 
            runtime = time.time() - start_time
            result = {f"{split}_runtime": round(runtime, 4)}
            if runtime == 0:
                return result
            if num_samples is not None:
                samples_per_second = num_samples / runtime
                result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
            if num_steps is not None:
                steps_per_second = num_steps / runtime
                result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
            if num_tokens is not None:
                tokens_per_second = num_tokens / runtime
                result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
            return result
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)


        start_time = time.time()



        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        ################### cross_entr_loss_dataset start##################################

        dataloader_params = {
            "batch_size": 4,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.cross_entr_loss_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(self.cross_entr_loss_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        test_dataloader = self.accelerator.prepare(DataLoader(cross_entr_loss_dataset, **dataloader_params))
        self.model.eval()
        losses_host = None
        with torch.no_grad():
            for step, inputs in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
                del inputs['messages']
                del inputs['label']
                inputs = {k: torch.tensor(v) for k, v in inputs.items()}
                inputs = self._prepare_inputs(inputs)
                loss = self.model(**inputs)[0].detach()

                if loss is not None:
                    losses = loss.repeat(inputs['input_ids'].shape[0])
                    losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            
            cross_entr_loss = losses_host.mean().cpu().item()


        output.metrics['eval_cross_entr_loss'] =  cross_entr_loss  
        #############################cross_entr_loss_dataset end ########################


        self.log(output.metrics)


        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

dpo_trainer = more_eval_loss_trainer(
    cross_entr_loss_dataset=cross_entr_loss_dataset,
    model=model,
    beta=0.1,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=model_max_length,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)
if evaluate_before_train:
    print(dpo_trainer.evaluate())
dpo_trainer.train()
