from data_process import get_data
from transformers import default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_PATH = "qwen_chatbot_healthcare"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def load_finetuned_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    # Đảm bảo mô hình ở chế độ evaluation
    model.eval()
    
    # Kiểm tra xem LoRA có được nạp đúng không
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        print("LoRA adapter not loaded correctly")
        # Thử cách khác
        model = PeftModel.from_pretrained(base_model, MODEL_PATH, is_trainable=False)
        model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    return model, tokenizer

def eval_results(model, tokenizer, eval_dataset, batch_size=4, max_length=512):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        # Sử dụng input_ids làm labels nếu không có cột labels
        labels = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_samples": total_tokens
    }

if __name__ == "__main__":
    model, tokenizer = load_finetuned_model()
    test_dataset = get_data()['test']
    results = eval_results(model, tokenizer, test_dataset)
    print(results)
