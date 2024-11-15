from peft import (LoraConfig,
                  get_peft_model,
                  TaskType)
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoConfig,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForSeq2Seq)
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置环境变量CUDA_VISIBLE_DEVICES为"0"，
# 这意味着你的程序将只使用第一个GPU设备
# 告诉transformers库在离线模式下运行，即不从互联网上下载模型和其他资源
os.environ['TRANSFORMERS_OFFLINE'] = '1'

warnings.filterwarnings("ignore")  # 忽略所有警告


_model_id = "/root/workspace/skyer_huggingface/cache/qwen/Qwen2___5-0___5B-Instruct"
# _model_id = "/root/workspace/skyer_huggingface/cache/skyer" #导入模型
'''
从模型中加载tokenizer、config、model
'''
_tokenizer = AutoTokenizer.from_pretrained(_model_id, trust_remote_code=True)
_config = AutoConfig.from_pretrained(_model_id, trust_remote_code=True)
# _config.cache_max_batch_size = None
_model = AutoModelForCausalLM.from_pretrained(
    _model_id, config=_config, trust_remote_code=True)

# for name ,param in _model.named_parameters():
#     print(name)

# 加载数据集
# _dataset = load_dataset("json", data_files="ruozhiba_qa.json", split="train")
_dataset = load_dataset(
    "json", data_dir="/root/workspace/skyer_huggingface/datas", split="train")


def preprocess_dataset(example):
    MAX_LENGTH = 128
    _input_ids, _attention_mask, _labels = [], [], []
    # _instruction = _tokenizer(
    #     f"<s>user\n{example['instruction']}</s>\n<s>assistant\n", add_special_tokens=False)
    # _response = _tokenizer(
    #     example["output"] + _tokenizer.eos_token, add_special_tokens=False)

    _instruction = _tokenizer(
        f"<s>user\n{example['title']}</s>\n<s>assistant\n", add_special_tokens=False)
    _response = _tokenizer(
        example["content"] + _tokenizer.eos_token, add_special_tokens=False)

    _input_ids = _instruction["input_ids"] + _response["input_ids"]
    _attention_mask = _instruction["attention_mask"] + \
        _response["attention_mask"]

    _labels = [-100] * len(_instruction["input_ids"]) + _response["input_ids"]

    if len(_input_ids) > MAX_LENGTH:
        _input_ids = _input_ids[:MAX_LENGTH]
        _attention_mask = _attention_mask[:MAX_LENGTH]
        _labels = _labels[:MAX_LENGTH]

    return {
        "input_ids": _input_ids,
        "attention_mask": _attention_mask,
        "labels": _labels
    }


_dataset = _dataset.map(
    preprocess_dataset, remove_columns=_dataset.column_names,num_proc=45)
_dataset = _dataset.shuffle()

# 初始化LoRA配置
config = LoraConfig(task_type=TaskType.CAUSAL_LM,  # 设置任务类型为因果语言模型
                    target_modules="all-linear",  # 指定LoRA应用于所有线性层
                    r=8,  # 设置LoRA的秩为8，这意味着每个LoRA层将有两个8维的小型矩阵
                    lora_alpha=16  # 设置LoRA的缩放因子，这影响LoRA层的缩放程度
                    )

# 将上述配置应用到一个预训练模型，以创建一个参数更小、更高效的模型版本
_model = get_peft_model(_model, config)

# model.print_trainable_parameters()

_training_args = TrainingArguments(
    output_dir="checkpoints/lora/qwen",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    save_steps=500
)

trainer = Trainer(
    model=_model,
    args=_training_args,
    train_dataset=_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer, padding=True)
)

trainer.train()
