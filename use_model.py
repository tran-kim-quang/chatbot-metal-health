from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel
import torch
from data_process import get_data
import numpy as np
from tqdm import tqdm

# Đường dẫn tới mô hình đã fine-tune
MODEL_PATH = "qwen_chatbot_healthcare"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

def load_finetuned_model():
    print("Đang nạp mô hình cơ sở...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print("Đang nạp LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    # Đảm bảo mô hình ở chế độ evaluation
    model.eval()
    
    # Kiểm tra xem LoRA có được nạp đúng không
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        print("CẢNH BÁO: Không tìm thấy tham số huấn luyện được. Có thể LoRA adapter không được nạp đúng.")
        print("Thử nạp lại mô hình...")
        # Thử cách khác
        model = PeftModel.from_pretrained(base_model, MODEL_PATH, is_trainable=False)
        model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    return model, tokenizer

def print_model_info(model):
    """
    Hiển thị thông số tham số của mô hình
    """
    print("=" * 60)
    print("THÔNG SỐ MÔ HÌNH QWEN_CHATBOT_HEALTHCARE")
    print("=" * 60)
    
    # Tổng số tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"Tổng số tham số: {total_params:,}")
    print(f"Số tham số huấn luyện được: {trainable_params:,}")
    print(f"Số tham số cố định: {non_trainable_params:,}")
    print(f"Tỷ lệ tham số huấn luyện được: {trainable_params/total_params*100:.2f}%")
    
    # Thông tin về LoRA
    print(f"\nThông tin LoRA:")
    print(f"- LoRA rank (r): 8")
    print(f"- LoRA alpha: 16")
    print(f"- LoRA dropout: 0.05")
    
    # Thông tin thiết bị
    device = next(model.parameters()).device
    print(f"\nThiết bị sử dụng: {device}")
    
    # Kích thước mô hình
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Kích thước mô hình: {model_size_mb:.2f} MB")
    
    print("=" * 60)

def evaluate_model_on_test_set(model, tokenizer, test_dataset, num_samples=50):
    """
    Đánh giá mô hình trên tập test
    """
    print("\nĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)
    
    # Lấy một số mẫu từ tập test để đánh giá
    if len(test_dataset) > num_samples:
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        test_samples = test_dataset.select(indices)
    else:
        test_samples = test_dataset
    
    print(f"Số mẫu đánh giá: {len(test_samples)}")
    
    # Tính toán loss trên tập test
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    print("Đang tính toán loss trên tập test...")
    with torch.no_grad():
        for i in tqdm(range(len(test_samples)), desc="Đánh giá"):
            sample = test_samples[i]
            input_ids = torch.tensor([sample['input_ids']]).to(model.device)
            attention_mask = torch.tensor([sample['attention_mask']]).to(model.device)
            
            # Sử dụng input_ids làm labels nếu không có cột labels
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()
    
    avg_loss = total_loss / len(test_samples)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"Loss trung bình trên tập test: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    return avg_loss, perplexity



