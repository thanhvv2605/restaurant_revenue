import yaml
from src.train import train_model

# Load cấu hình từ config.yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Chạy huấn luyện mô hình
train_model(config)