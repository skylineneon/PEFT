import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          DataCollatorWithPadding)
from datasets import load_dataset


_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
_model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-large")
# for name,param in _model.bert.named_parameters():
    # print(name)
    # print(name,param.dtype)
    # param.requires_grad=False
# exit()

_dataset = load_dataset("csv",data_files="ChnSentiCorp_htl_all.csv",split="train")

_dataset = _dataset.filter(lambda x:x["review"] is not None)


def preprocess_dataset(data,tokenizer=_tokenizer):
    _rst = tokenizer(data["review"],max_length=512,truncation=True,padding="max_length")
    _rst["labels"] = data["label"]
    return _rst


_dataset = _dataset.map(preprocess_dataset,remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()
_datasets = _dataset.train_test_split(test_size=0.2)


training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    # gradient_accumulation_steps=12,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.1,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    logging_steps=10,
    # fp16=True
    # optim="adafactor"

)


_accuracy_model = evaluate.load("accuracy")

def compute_accuracy(result):
    _predictions,_labels = result
    _predictions = _predictions.argmax(-1)
    _accuracy =  _accuracy_model(predictions=_predictions,labels=_labels)
    return _accuracy


trainer = Trainer(
    model=_model,
    tokenizer=_tokenizer,
    args=training_args,
    train_dataset=_datasets["train"],
    eval_dataset=_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=_tokenizer),
    compute_metrics=compute_accuracy,
    
)

trainer.train()
# trainer.evaluate()


