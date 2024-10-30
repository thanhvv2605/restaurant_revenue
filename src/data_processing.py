import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    # Xóa giá trị thiếu
    df = df.dropna()
    return df

def preprocess_data(df):
    # Mã hóa các cột dạng chuỗi thành số
    label_encoders = {}
    for col in ['Name', 'Location', 'Cuisine', 'Parking Availability']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Lưu lại LabelEncoder để sử dụng lại nếu cần
    return df, label_encoders

def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)