import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForSeq2Seq)
from datasets import load_dataset


_tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
_model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")

ps = 0
for name,param in _model.named_parameters():
    # print(name)
    if "bias" not in name:
        param.requires_grad = False
    else:
        ps += param.numel()
print(ps/sum([param.numel() for _,param in _model.named_parameters()]))

_dataset = load_dataset("json",data_files="ruozhiba.json",split="train")


def preprocess_dataset(example):
    MAX_LENGTH = 256
    _input_ids, _attention_mask, _labels = [], [], []
    _instruction = _tokenizer("\n".join(["Human: ", example["instruction"]]).strip() + "\n\nAssistant: ",add_special_tokens=False)
    _response = _tokenizer(example["output"] + _tokenizer.eos_token,add_special_tokens=False)
    _input_ids = _instruction["input_ids"] + _response["input_ids"]
    _attention_mask = _instruction["attention_mask"] + _response["attention_mask"]
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

_dataset = _dataset.map(preprocess_dataset,remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()


training_args = TrainingArguments(
    output_dir="checkpoints/bitfit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3

)

trainer = Trainer(
    model=_model,
    args=training_args,
    train_dataset=_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer,padding=True)
)

trainer.train()

#模型推理

from transformers import pipeline

pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer)
ipt = "Human: {}\n{}".format("爸爸再婚，我是不是就有了个新娘？", "").strip() + "\n\nAssistant: "
print(pipe(ipt, max_length=256, do_sample=True, ))


