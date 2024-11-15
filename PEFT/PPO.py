import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
warnings.filterwarnings("ignore")

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  )
import torch
from datasets import load_dataset
from transformers import (pipeline,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BitsAndBytesConfig)
from trl import AutoModelForCausalLMWithValueHead,PPOConfig, PPOTrainer

import tqdm

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
_r_model = AutoModelForSequenceClassification.from_pretrained("checkpoints/reward/checkpoint-1000")

_reward_model = pipeline(task="text-classification",model=_r_model,tokenizer=_tokenizer)


_tokenizer.add_special_tokens({"bos_token": _tokenizer.eos_token,
                               "pad_token": _tokenizer.eos_token})
_tokenizer.bos_token_id = _tokenizer.eos_token_id
_tokenizer.pad_token_id = _tokenizer.eos_token_id

_bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float32
                                 )

_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2-0.5B-Instruct",
                                                           low_cpu_mem_usage=True,
                                                           quantization_config=_bnb_config)

_ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2-0.5B-Instruct",
                                                               low_cpu_mem_usage=True,
                                                               quantization_config=_bnb_config)

_dataset = load_dataset("json",data_files="ruozhiba.json",split="train")


def preprocess_dataset(data,tokenizer=_tokenizer):
    question = tokenizer(f"<|im_start|>user\n{data['instruction']}<|im_end|><|im_start|>assistant\n{data['output']}<|im_end|>")
    return question["input_ids"]

_dataset = _dataset.map(preprocess_dataset,
                        batch_size=True,
                        num_proc=10,
                        remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()


config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    target_modules="all-linear")


_model = get_peft_model(_model, config)
_model.config.pad_token_id = _model.config.eos_token_id

# model.print_trainable_parameters()
_model.enable_input_require_grads()

_training_args = PPOConfig(
    output_dir="checkpoints/ppo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_steps=100,
    optim="paged_adamw_32bit"
)

_ppo_trainer = PPOTrainer(
    model=_model,
    ref_model=_ref_model,
    tokenizer=_tokenizer,
    args=_training_args,
    train_dataset=_dataset
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.8,
    "do_sample": True,
    "pad_token_id": _tokenizer.eos_token_id,
}

epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(_ppo_trainer.dataloader): 
        query_tensors = batch["input_ids"]
    
        #### Get response from SFTModel
        response_tensors = _ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [_tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = _reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
        #### Run PPO step
        stats = _ppo_trainer.step(query_tensors, response_tensors, rewards)
        # _ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
_ppo_trainer.save_pretrained("my_ppo_model")
