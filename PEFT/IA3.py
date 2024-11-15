from peft import (IA3Config,
                  get_peft_model,
                  TaskType)
from peft import PeftModel
from transformers import pipeline
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForSeq2Seq)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


_model_id = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
# _model_id = "/root/workspace/skyer_huggingface/cache/skyer"

_tokenizer = AutoTokenizer.from_pretrained(_model_id, trust_remote_code=True)
_model = AutoModelForCausalLM.from_pretrained(
    _model_id, trust_remote_code=True)

# _dataset = load_dataset("json", data_files="ruozhiba_qa.json", split="train")


# def preprocess_dataset(example):
#     MAX_LENGTH = 128
#     _input_ids, _attention_mask, _labels = [], [], []
#     _instruction = _tokenizer(
#         f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
#     _response = _tokenizer(
#         example["output"] + _tokenizer.eos_token, add_special_tokens=False)

#     _input_ids = _instruction["input_ids"] + _response["input_ids"]
#     _attention_mask = _instruction["attention_mask"] + \
#         _response["attention_mask"]

#     _labels = [_tokenizer.pad_token_id] * \
#         len(_instruction["input_ids"]) + _response["input_ids"]

#     if len(_input_ids) > MAX_LENGTH:
#         _input_ids = _input_ids[:MAX_LENGTH]
#         _attention_mask = _attention_mask[:MAX_LENGTH]
#         _labels = _labels[:MAX_LENGTH]

#     return {
#         "input_ids": _input_ids,
#         "attention_mask": _attention_mask,
#         "labels": _labels
#     }


# _dataset = _dataset.map(
#     preprocess_dataset, remove_columns=_dataset.column_names)
# _dataset = _dataset.shuffle()


# config = IA3Config(task_type=TaskType.CAUSAL_LM)

# _model = get_peft_model(_model, config)

# # model.print_trainable_parameters()

# _training_args = TrainingArguments(
#     output_dir="checkpoints/IA3",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=1,
#     logging_steps=10,
#     num_train_epochs=1,
#     save_steps=20,
# )
# trainer = Trainer(
#     model=_model,
#     args=_training_args,
#     train_dataset=_dataset,
#     data_collator=DataCollatorForSeq2Seq(
#         tokenizer=_tokenizer, padding=True, label_pad_token_id=_tokenizer.pad_token_id)
# )

# trainer.train()

# 模型推理


peft_model = PeftModel.from_pretrained(
    model=_model, model_id="/root/workspace/PEFT/checkpoints/IA3/checkpoint-740")

peft_model = peft_model.cuda()
ipt = _tokenizer("Human: {}\n{}".format("你是谁？", "").strip(
) + "\n\nAssistant: ", return_tensors="pt").to(peft_model.device)
print(_tokenizer.decode(peft_model.generate(
    **ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True))
