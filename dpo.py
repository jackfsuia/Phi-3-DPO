from trl import DPOConfig, DPOTrainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import os
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
)

m_path = "./"
train_dataset_path = "train.jsonl"
eval_dataset_path = "eval.jsonl"

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
    gradient_accumulation_steps=512,
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=1,
    save_total_limit=3,
    learning_rate=5e-6,
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
    load_best_model_at_end=True
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
    add_bos_token=False,  # 没有<s>
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

dpo_trainer = DPOTrainer(
    model,
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
