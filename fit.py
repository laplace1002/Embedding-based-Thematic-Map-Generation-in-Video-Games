import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    加载数据，增加 PCA 降维处理
    """
    # 读取CSV
    df = pd.read_csv(data_path, header=None)

    # 分割特征与标签
    X = df.iloc[:, :2560].values
    y = df.iloc[:, 2560:].values

    # 划分训练集与验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 1. 标准化 (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 2. PCA 降维 (关键修改：保留 95% 的方差信息)
    # 这会将 2560 维压缩到几百维，解决维度灾难
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    print(f"PCA将特征维度从 {X.shape[1]} 降到了 {X_train_pca.shape[1]}")

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # 创建数据集和加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 返回 PCA 对象以便保存
    return train_loader, val_loader, scaler, pca


class RegressionModel(nn.Module):
    """
    修改后的模型：参数量更小，适应 PCA 后的数据
    """

    def __init__(
            self,
            input_dim: int,  # 这里的 input_dim 将由 PCA 决定
            hidden_dim1: int = 256,  # 减小隐藏层 (原 1024)
            hidden_dim2: int = 128,  # 减小隐藏层 (原 512)
            output_dim: int = 2,
            dropout_rate: float = 0.4  # 稍微增加 Dropout
    ):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        output = self.output_layer(x)
        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)

        print(f'\nEpoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 保存模型参数 + 输入维度配置 (因为PCA维度是不确定的)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': model.layer1.in_features
            }, 'best_model.pth')
            early_stopping_counter = 0
            print(f'Best model saved (val loss improved to {best_val_loss:.4f})')
        else:
            early_stopping_counter += 1
            print(f'Early stopping counter: {early_stopping_counter}/{patience}')
            if early_stopping_counter >= patience:
                print('Early stopping triggered')
                break

    # 加载最佳模型
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_loss_history, val_loss_history


def main():
    data_path = 'dataset.csv'
    output_dim = 2
    learning_rate = 0.001
    num_epochs = 100
    patience = 15  # 稍微增加 patience

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    print('\nLoading data and applying PCA...')
    # 注意：这里接收了 PCA 对象
    train_loader, val_loader, scaler, pca = load_data(data_path)

    # 动态获取 PCA 降维后的特征数量
    pca_input_dim = train_loader.dataset.tensors[0].shape[1]
    print(f'Input dimension after PCA: {pca_input_dim}')

    print('\nInitializing model...')
    model = RegressionModel(input_dim=pca_input_dim)

    criterion = nn.MSELoss()
    # 关键修改：增加 weight_decay (L2正则化) 防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print('\nStarting training...')
    model, train_loss_history, val_loss_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience
    )

    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss Curves (with PCA + Regularization)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()

    # 保存 Scaler 和 PCA
    print('\nSaving preprocessors...')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(pca, 'pca.joblib')  # 必须保存 PCA 才能做预测
    print('Scaler and PCA saved.')


def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Predicting on device: {device}')

    print('\nLoading preprocessors...')
    try:
        scaler = joblib.load('scaler.joblib')
        pca = joblib.load('pca.joblib')
    except FileNotFoundError:
        print("Error: scaler.joblib or pca.joblib not found. Please train first.")
        return

    print('Loading model...')
    try:
        checkpoint = torch.load('best_model.pth')
        # 从 checkpoint 中读取训练时的输入维度
        input_dim = checkpoint['input_dim']
        model = RegressionModel(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print("Error: best_model.pth not found.")
        return

    print('Loading data from result.txt...')
    try:
        data = pd.read_csv('result.txt', header=None).values
        if data.shape[1] != 2560:
            data = data[:, :2560]  # 容错处理
        print(f'Loaded {len(data)} samples.')
    except Exception as e:
        print(f'Error loading data: {e}')
        return

    # 预测流程：Scaler -> PCA -> Model
    print('Preprocessing data (Scaler + PCA)...')
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)  # 关键：应用同样的 PCA 变换

    data_tensor = torch.tensor(data_pca, dtype=torch.float32).to(device)

    print('Running predictions...')
    with torch.no_grad():
        predictions = model(data_tensor).cpu().numpy()

    np.savetxt('result1.txt', predictions, delimiter='\t', fmt='%.6f')
    print('Predictions saved to result1.txt')


if __name__ == '__main__':
    mode = input("Enter 'train' to train model or 'predict' to run predictions: ").strip().lower()
    if mode == 'train':
        main()
    elif mode == 'predict':
        predict()
    else:
        print("Invalid option.")