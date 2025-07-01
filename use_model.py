import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-0.5B"
lora_adapter_id = "meomeo163/QWEN2.5_chatbot_health_care" 

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, lora_adapter_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = input("Human: ")

# Mã hóa input và chuyển đến thiết bị phù hợp
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=500) # Adjusted max_new_tokens for shorter response
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
