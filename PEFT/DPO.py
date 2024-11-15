from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig)
from trl import DPOConfig, DPOTrainer # 导入DPO配置和训练器
from modelscope.msdatasets import MsDataset #提供加载modelscope上的数据集方法


_model_path = "/root/workspace/skyer_huggingface/cache/skyer"

_tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)
_config = AutoConfig.from_pretrained(_model_path, trust_remote_code=True)
_config.cache_max_batch_size = None
_model = AutoModelForCausalLM.from_pretrained(_model_path,
                                              config=_config,
                                              trust_remote_code=True)


# 从modelscope上加载数据集
_dataset = MsDataset.load('AI-ModelScope/Chinese-dpo-pairs', split="train") 


def preprocess_dataset(data):
    return {
        "prompt": f"<s>user\n{data['prompt']}</s><s>assistant\n",
        "chosen": f"{data['chosen']}</s>",
        "rejected": f"{data['rejected']}</s>"
    }
    


_dataset = _dataset.map(preprocess_dataset,
                        remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()

# 创建DPO训练配置
_training_args = DPOConfig(
    output_dir="checkpoints/dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=6,
    save_steps=100,
    # optim="paged_adamw_32bit"
)

_trainer = DPOTrainer(
    model=_model,
    tokenizer=_tokenizer,
    args=_training_args,
    train_dataset=_dataset,
    max_prompt_length=512,
    max_length=1024,
    max_target_length=1024
)

_trainer.train()
