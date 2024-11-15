from peft import PeftModel
from transformers import (pipeline,
                          AutoModelForCausalLM,
                          AutoTokenizer)
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TRANSFORMERS_OFFLINE'] = '1'

warnings.filterwarnings("ignore")

_model_path = "/root/workspace/skyer_huggingface/cache/qwen/Qwen2___5-0___5B-Instruct"
_model_id = "/root/workspace/PEFT/checkpoints/lora/qwen/checkpoint-10000"
_model = AutoModelForCausalLM.from_pretrained(
    _model_path, trust_remote_code=False, device_map="cuda")
_tokenizer = AutoTokenizer.from_pretrained(
    _model_path, trust_remote_code=False)
peft_model = PeftModel.from_pretrained(
    model=_model, model_id=_model_id)
peft_model = peft_model.merge_and_unload()
# peft_model.save_pretrained("myqwen2-0.5b")

pipe = pipeline("text-generation",
                model=peft_model,
                tokenizer=_tokenizer,
                device="cuda:0",
                max_length=100)
ipt = f"User:为什么张国荣的粉丝每年都会组织参与纪念活动？Assistant:"
print(pipe(ipt))
