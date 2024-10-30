import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.model import SimpleNN
from src.data_processing import load_data, clean_data, preprocess_data, split_data

def train_model(config):
    # Load và xử lý dữ liệu
    df = load_data('data/processed/restaurant_data.csv')  # Thay đường dẫn file phù hợp
    df = clean_data(df)
    df, label_encoders = preprocess_data(df)  # Mã hóa các cột chuỗi
    X_train, X_test, y_train, y_test = split_data(df, target_column='Revenue')

    # Chuyển đổi dữ liệu sang tensor
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Tạo DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Khởi tạo mô hình, hàm mất mát và bộ tối ưu
    model = SimpleNN(config['model']['input_dim'], config['model']['hidden_units'], config['model']['output_dim'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Huấn luyện mô hình
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {running_loss/len(train_loader):.4f}")

    # Lưu mô hình
    torch.save(model.state_dict(), 'experiments/exp1/checkpoints/model.pth')
    print("Model saved at experiments/exp1/checkpoints/model.pth")