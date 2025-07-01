from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from data_process import get_data
# Load model directly
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Config lora
lora_config = LoraConfig(
    r=8, # LoRA attention dimension
    lora_alpha=16, # Alpha parameter for LoRA scaling
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
    task_type="CAUSAL_LM", # Task type, in this case it's causal language modeling
    # target_modules=["query_key_value"], # Specify the layers to apply LoRA to. Adjust based on your model architecture
)

model_lora = get_peft_model(model, lora_config)

train = get_data()['train']
test = get_data()['test']


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    # Removed evaluation_strategy as it caused a TypeError
    learning_rate=0.0001,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False, # Set to False to avoid authentication error when pushing to Hugging Face Hub
    logging_dir='./logs',
    logging_steps=10, # Log training loss every 10 steps
    report_to=[], # Explicitly set report_to to an empty list to try and force console logging
    logging_first_step=True # Log the first step to confirm logging is working
)


train_dataset_with_labels = train.add_column("labels", train["input_ids"])
test_dataset_with_labels = test.add_column("labels", test["input_ids"])


trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=train_dataset_with_labels, 
    eval_dataset=test_dataset_with_labels,   
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./qwen_chatbot_healthcare")
tokenizer.save_pretrained("./qwen_chatbot_healthcare")