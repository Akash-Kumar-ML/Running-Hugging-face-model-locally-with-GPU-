from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4-8B", trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4-8B",
    device_map={"": 0},           
)


prompts = "User: What is the meaning of insecurity and what are it's types?\nAssistant:"
inputs = tokenizer(prompts, return_tensors="pt").to(model.device)


outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=1.9,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
