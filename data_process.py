import pandas as pd
from datasets import Dataset # Import Dataset class
from transformers import AutoTokenizer, AutoModelForCausalLM

ds = pd.read_parquet("hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet")

# Load model directly
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

# Convert pandas DataFrame to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(ds)
token_data = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
token_data = token_data.train_test_split(test_size=0.2)

def get_data():
    return token_data