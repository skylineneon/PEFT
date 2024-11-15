from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig)
# 导入trl库的SFT配置、训练器和数据整理器
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import os


_model_path = "/root/workspace/skyer_huggingface/cache/qwen/Qwen2___5-0___5B-Instruct"
# _model_path = "/root/workspace/skyer_huggingface/cache/skyer"
# _model_path = "/root/workspace/PEFT/checkpoints/sft/checkpoint-99500"

_tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)
_config = AutoConfig.from_pretrained(_model_path, trust_remote_code=True)
# _config.cache_max_batch_size = None
_model = AutoModelForCausalLM.from_pretrained(_model_path,
                                              config=_config,
                                              trust_remote_code=True)
# # file_path = "/root/workspace/skyer_huggingface/datas"
#     _dataset = load_dataset("json", data_files=file_path, split="train")
_dataset = load_dataset(
    "json", data_dir="/root/workspace/skyer_huggingface/datas", split="train")


def preprocess_dataset(data):
    return {"text": f"<s>user\n{data['title']}</s><>assistant\n{data['content']}<s>"}


_dataset = _dataset.map(
    preprocess_dataset, remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()

_response_template = "<s>assistant\n"  # 定义响应模版
# 创建数据整理器
_collator = DataCollatorForCompletionOnlyLM(
    _response_template, tokenizer=_tokenizer)

_training_args = SFTConfig(
    output_dir="checkpoints/sft/qwen",
    dataset_text_field="text",  # 设置数据集文本字段
    max_seq_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100
    # optim="paged_adamw_32bit"长度
)

_trainer = SFTTrainer(
    model=_model,
    tokenizer=_tokenizer,
    args=_training_args,
    train_dataset=_dataset,
    data_collator=_collator
)

_trainer.train()
