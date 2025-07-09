import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error # type: ignore
import os

# 设置随机种子确保结果可复现
def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# 加载数据
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=";")
    print(f"数据基本信息：")
    df.info()
    
    # 转换日期时间格式
    df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
    
    # 只保留整点数据（小时级预测）
    df = df[df['datetime'].dt.minute == 0].copy()
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # 处理缺失值
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # 将数值列转换为float类型
    for col in df.columns:
        if col != 'datetime':
            df[col] = df[col].astype(float)
    
    print(f"数据已处理完毕，共有{len(df)}条记录")
    print(f"数据时间范围：{df['datetime'].min()} 至 {df['datetime'].max()}")
    
    return df

# 数据分割与归一化
def prepare_data(df, target_col='Global_active_power', test_split_date='2009-12-31', seq_length=168):
    # 按时间分割训练集和测试集
    train = df.loc[df['datetime'] <= test_split_date].copy()
    test = df.loc[df['datetime'] > test_split_date].copy()
    
    print(f"训练集时间范围：{train['datetime'].min()} 至 {train['datetime'].max()}，共{len(train)}条记录")
    print(f"测试集时间范围：{test['datetime'].min()} 至 {test['datetime'].max()}，共{len(test)}条记录")
    
    # 提取特征列（排除datetime）
    feature_cols = [col for col in train.columns if col != 'datetime']
    
    # 归一化处理 - 分别为特征和目标值创建scaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    train_scaled = scaler_X.fit_transform(train[feature_cols])
    test_scaled = scaler_X.transform(test[feature_cols])
    
    # 目标列的归一化（单独保存以便后续反归一化）
    scaler_y.fit(train[[target_col]])
    
    # 创建序列数据
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, :])
            y.append(data[i, feature_cols.index(target_col)])  # 只预测目标列
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    print(f"序列数据创建完成：")
    print(f"训练数据形状：X={X_train.shape}, y={y_train.shape}")
    print(f"测试数据形状：X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler_y, test['datetime'].iloc[seq_length:].reset_index(drop=True)

# 定义LSTM模型 - 增加了Dropout层和多层LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层LSTM结构
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=False
        )
        
        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # shape: (batch, seq_len, hidden_size)
        
        # 应用注意力机制
        attn_weights = self.attention(lstm_out)  # shape: (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # shape: (batch, 1, hidden_size)
        context = context.squeeze(1)  # shape: (batch, hidden_size)
        
        # 通过全连接层输出预测值
        output = self.fc(context)
        return output.squeeze()

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, device='cpu'):
    model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
    
    return history

# 评估模型
def evaluate_model(model, test_loader, scaler_y, device='cpu'):
    model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    # 反归一化
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    predictions_inv = scaler_y.inverse_transform(predictions)
    actuals_inv = scaler_y.inverse_transform(actuals)
    
    # 计算评估指标
    mse = mean_squared_error(actuals_inv, predictions_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_inv, predictions_inv)
    
    print(f"\n模型评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return predictions_inv, actuals_inv

# 可视化结果
def visualize_results(predictions, actuals, timestamps, target_col='Global_active_power', sample_size=500):
    plt.figure(figsize=(14, 7))
    
    # 完整预测结果
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, actuals, label='实际值', alpha=0.7)
    plt.plot(timestamps, predictions, label='预测值', alpha=0.7)
    plt.title(f'{target_col} 预测结果 - 完整数据')
    plt.xlabel('日期时间')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    
    # 局部预测结果（前sample_size个点）
    plt.subplot(2, 1, 2)
    plt.plot(timestamps[:sample_size], actuals[:sample_size], label='实际值', alpha=0.7)
    plt.plot(timestamps[:sample_size], predictions[:sample_size], label='预测值', alpha=0.7)
    plt.title(f'{target_col} 预测结果 - 局部数据')
    plt.xlabel('日期时间')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('lstm_prediction_results.png')
    plt.show()

# 主函数
def main():
    # 文件路径
    file_path = 'data//household_power_consumption.txt'
    
    # 加载并预处理数据
    df = load_and_preprocess_data(file_path)
    
    # 准备数据
    X_train, y_train, X_test, y_test, scaler_y, test_timestamps = prepare_data(
        df, 
        target_col='Global_active_power',
        test_split_date='2009-12-31',
        seq_length=168  # 使用7天（168小时）的数据来预测下一个小时
    )
    
    # 转换为PyTorch张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # 创建数据加载器
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 划分训练集和验证集
    val_size = int(0.2 * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_size = X_train.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_size=1
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 训练模型
    print("\n开始训练模型...")
    epochs = 30
    history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    
    # 学习曲线可视化
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), history['train_loss'], label='训练损失')
    plt.plot(range(1, epochs+1), history['val_loss'], label='验证损失')
    plt.title('模型训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    # 评估模型
    print("\n开始评估模型...")
    predictions, actuals = evaluate_model(model, test_loader, scaler_y, device)
    
    # 可视化结果
    visualize_results(predictions, actuals, test_timestamps, 'Global_active_power', sample_size=500)
    
    print("\n预测完成！结果已保存到lstm_prediction_results.png")

if __name__ == "__main__":
    main()    