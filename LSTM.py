import joblib
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv", parse_dates=['DateTime'])
df.set_index('DateTime', inplace=True)
df.sort_index(inplace=True)  # 确保时间顺序
# 数据类型转换
cols_to_float = [
    'Global_active_power', 'Global_reactive_power',
    'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']
# 强制转换这些列为 float 类型，把非法字符（如 '?'）转为 NaN
df[cols_to_float] = df[cols_to_float].apply(pd.to_numeric, errors='coerce')
# print(df[cols_to_float].dtypes)

# 分类字段
sum_cols = ['Global_active_power', 'Global_reactive_power',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
mean_cols = ['Voltage', 'Global_intensity']
weather_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

# 按天汇总
df_daily = pd.DataFrame()
df_daily[sum_cols] = df[sum_cols].resample('D').sum()         # 每天累加
df_daily[mean_cols] = df[mean_cols].resample('D').mean()      # 每天求均值
df_daily[weather_cols] = df[weather_cols].resample('D').first()  # 每天保留1个天气值（值都是一样的，任意保留一个即可）

# print(df_daily.head())
# value = df_daily.loc['2006-12-16', 'Global_active_power']
# print(value)

# 数据缺失值处理
# 1. 查看缺失行（按天为单位）
missing_days = df_daily.isna().any(axis=1).sum()
total_days = len(df_daily)
missing_ratio = missing_days / total_days

print(f"总天数：{total_days}")
print(f"缺失天数：{missing_days}")
print(f"缺失比例：{missing_ratio:.2%}")

# 2. 自动选择处理方式
if missing_ratio <= 0.05:
    print("缺失天数少于5%，选择删除缺失天")
    df_daily_cleaned = df_daily.dropna().copy()
else:
    print("缺失天数多于5%，采用线性插值法进行填补")
    df_daily_cleaned = df_daily.interpolate(method='linear')

# 3. 最终输出检查
print(f"\n缺失处理后剩余天数：{len(df_daily_cleaned)}")
print(f"是否还有缺失值？\n{df_daily_cleaned.isna().sum()}")

# 数据归一化
from sklearn.preprocessing import MinMaxScaler

target_col = 'Global_active_power'
features = df_daily_cleaned.drop(columns=[target_col])
target = df_daily_cleaned[[target_col]]

# 初始化 scaler
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 归一化
features_scaled = pd.DataFrame(feature_scaler.fit_transform(features), columns=features.columns, index=features.index)
target_scaled = pd.DataFrame(target_scaler.fit_transform(target), columns=[target_col], index=target.index)

# 合并
# df_scaled 是归一化后的完整数据集，可直接用于构造滑动窗口样本
df_scaled = pd.concat([features_scaled, target_scaled], axis=1)
# print(df_scaled.head())

# 构造滑动窗口
def create_sliding_window(df_scaled, target_col='Global_active_power',
                          input_window=90, output_window=90):
    """
    根据归一化后的数据 df_scaled 构造滑动窗口样本
    - 输入：过去 input_window 天的多变量特征
    - 输出：未来 output_window 天的 global_active_power
    """
    X, y = [], []

    # 所有输入特征列（除目标列外）
    feature_cols = df_scaled.columns.difference([target_col])

    data = df_scaled.copy()
    data_X = data[feature_cols].values  # shape: [总天数, 特征数]
    data_y = data[target_col].values  # shape: [总天数]

    total_days = len(df_scaled)
    max_start_index = total_days - input_window - output_window + 1

    for start_idx in range(max_start_index):
        end_input = start_idx + input_window
        end_output = end_input + output_window

        X.append(data_X[start_idx: end_input])  # shape: [input_window, 特征数]
        y.append(data_y[end_input: end_output])  # shape: [output_window]

    return np.array(X), np.array(y)

# 构造短期样本：90天输入，预测未来90天
X_short, y_short = create_sliding_window(df_scaled, input_window=90, output_window=90)

# 构造长期样本：90天输入，预测未来365天
X_long, y_long = create_sliding_window(df_scaled, input_window=90, output_window=365)

print("短期预测样本维度：", X_short.shape, y_short.shape)
print("长期预测样本维度：", X_long.shape, y_long.shape)

# 训练集、验证集划分
def train_val_split(X, y, val_ratio=0.2):
    """
    基于时间顺序划分训练集和验证集
    """
    total_samples = len(X)
    split_index = int(total_samples * (1 - val_ratio))

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_val = X[split_index:]
    y_val = y[split_index:]

    return X_train, y_train, X_val, y_val

# 对短期预测样本划分
X_train_short, y_train_short, X_val_short, y_val_short = train_val_split(X_short, y_short, val_ratio=0.2)

# 对长期预测样本划分
X_train_long, y_train_long, X_val_long, y_val_long = train_val_split(X_long, y_long, val_ratio=0.2)

print("短期训练集形状：", X_train_short.shape, y_train_short.shape)
print("短期验证集形状：", X_val_short.shape, y_val_short.shape)

print("长期训练集形状：", X_train_long.shape, y_train_long.shape)
print("长期验证集形状：", X_val_long.shape, y_val_long.shape)

# LSTM 多变量时间序列预测完整封装：五轮训练与评估
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import os

# ========== 模型定义 ==========
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_window, dropout=0.3):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_window)

    def forward(self, x):
        out, _ = self.lstm(x)  # [B, T, H]
        out = out[:, -1, :]    # 取最后一个时间步输出 [B, H]
        out = self.fc(out)     # [B, output_window]
        return out

# ========== 构建数据加载器 ==========
def build_loader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== 单轮训练 + 评估 ==========
def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss = criterion(output, y_batch)
                val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")

# ========== 单轮执行 ==========
def train_and_evaluate_once(X_train, y_train, X_val, y_val, target_scaler,
                            input_window, output_window, input_size,
                            hidden_size=64, num_layers=2, num_epochs=30, lr=1e-3, batch_size=64, device='cpu', dropout=0.3):
    model = LSTMForecast(input_size, hidden_size, num_layers, output_window, dropout=dropout).to(device)
    train_loader = build_loader(X_train, y_train, batch_size)
    val_loader = build_loader(X_val, y_val, batch_size)
    train_model(model, train_loader, val_loader, num_epochs, lr, device)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).cpu().numpy()
        y_true = y_tensor.cpu().numpy()

    if target_scaler:
        y_pred = target_scaler.inverse_transform(y_pred)
        y_true = target_scaler.inverse_transform(y_true)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    std = np.std(y_pred - y_true)

    return mse, mae, std, y_true, y_pred, model

# ========== 多轮实验执行 ==========
def run_k_experiments(k, X_train, y_train, X_val, y_val, target_scaler,
                      input_window, output_window, input_size,
                      hidden_size=64, num_layers=2, num_epochs=30, lr=1e-3, batch_size=64, device='cpu', dropout=0.3):
    mse_list, mae_list, std_list = [], [], []

    for i in range(k):
        print(f"\n\U0001F680 Round {i+1}/{k} =============================")
        mse, mae, std, _, _, _ = train_and_evaluate_once(
            X_train, y_train, X_val, y_val, target_scaler,
            input_window, output_window, input_size,
            hidden_size, num_layers, num_epochs, lr, batch_size, device, dropout
        )
        mse_list.append(mse)
        mae_list.append(mae)
        std_list.append(std)

        print(f"Round {i+1} | MSE: {mse:.4f} | MAE: {mae:.4f} | STD: {std:.4f}")

    print("\nFinal 5-Round Average Results:")
    print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
    print(f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"STD: {np.mean(std_list):.4f} ± {np.std(std_list):.4f}")


# ========== 可视化预测结果（单样本） ==========
def plot_predictions(y_true, y_pred, sample_index=0, title="Prediction vs Actual", days=90, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[sample_index][:days], label="Actual", linewidth=2)
    plt.plot(y_pred[sample_index][:days], label="Predicted", linestyle="--")
    plt.xlabel("Days")
    plt.ylabel("Global Active Power (kW)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ========== 可视化多个样本的子图 ==========
def plot_multiple_predictions(y_true, y_pred, sample_indices=[0, 1, 2, 3], days=90, save_path=None):
    n = len(sample_indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    for i, idx in enumerate(sample_indices):
        axes[i].plot(y_true[idx][:days], label="Actual", linewidth=2)
        axes[i].plot(y_pred[idx][:days], label="Predicted", linestyle="--")
        axes[i].set_title(f"Sample {idx}: Prediction vs Actual")
        axes[i].set_xlabel("Days")
        axes[i].set_ylabel("Power (kW)")
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ========== 封装完整流程：训练 + 可视化 + 评估 + 导出 ==========
def run_full_training_and_plot(X_train, y_train, X_val, y_val, target_scaler,
                               input_window, output_window, input_size,
                               sample_indices=[0, 2, 4, 6],
                               hidden_size=64, num_layers=2, num_epochs=30,
                               lr=1e-3, batch_size=64, device='cpu',dropout=0.3,
                               save_fig_path='prediction_plot.png',
                               save_csv_path='prediction_results.csv'):
    mse, mae, std, y_true, y_pred, model = train_and_evaluate_once(
        X_train, y_train, X_val, y_val, target_scaler,
        input_window, output_window, input_size,
        hidden_size, num_layers, num_epochs, lr, batch_size,device,dropout
    )
    print(f"\nSingle Round Evaluation:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, STD: {std:.4f}")

    # 可视化并保存图像
    plot_multiple_predictions(y_true, y_pred, sample_indices=sample_indices, days=output_window, save_path=save_fig_path)

    # 保存预测结果为 CSV
    pred_df = pd.DataFrame({
        f'sample_{i}_true': y_true[i] for i in sample_indices
    })
    for i in sample_indices:
        pred_df[f'sample_{i}_pred'] = y_pred[i]

    pred_df.to_csv(save_csv_path, index=False)

    return model

# ========== 5轮训练结束后绘图 ==========
def run_5rounds_then_plot(X_train, y_train, X_val, y_val, target_scaler,
                         input_window, output_window, input_size,
                         sample_indices=[0, 2, 4, 6],
                         hidden_size=64, num_layers=2, num_epochs=30,
                         lr=1e-3, batch_size=64, device='cpu',dropout=0.3,
                         save_fig_path='prediction_plot.png',
                         save_csv_path='prediction_results.csv'):
    mse_list, mae_list, std_list = [], [], []
    last_model = None
    last_y_true = None
    last_y_pred = None

    for i in range(5):
        print(f"\nRound {i+1}/5 Training ===========================")
        mse, mae, std, y_true, y_pred, model = train_and_evaluate_once(
            X_train, y_train, X_val, y_val, target_scaler,
            input_window, output_window, input_size,
            hidden_size, num_layers, num_epochs, lr, batch_size, device,dropout
        )
        mse_list.append(mse)
        mae_list.append(mae)
        std_list.append(std)
        print(f"Round {i+1} | MSE: {mse:.4f} | MAE: {mae:.4f} | STD: {std:.4f}")

        # 只保存最后一轮结果用于绘图
        if i == 4:
            last_model = model
            last_y_true = y_true
            last_y_pred = y_pred

    print("\nFinal 5-Round Average Results:")
    print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
    print(f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"STD: {np.mean(std_list):.4f} ± {np.std(std_list):.4f}")

    # 第5轮训练后绘图并保存结果
    plot_multiple_predictions(last_y_true, last_y_pred, sample_indices=sample_indices,
                              days=output_window, save_path=save_fig_path)
    pred_df = pd.DataFrame({f'sample_{i}_true': last_y_true[i] for i in sample_indices})
    for i in sample_indices:
        pred_df[f'sample_{i}_pred'] = last_y_pred[i]
    pred_df.to_csv(save_csv_path, index=False)

    return last_model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = run_5rounds_then_plot(
        X_train=X_train_long,
        y_train=y_train_long,
        X_val=X_val_long,
        y_val=y_val_long,
        target_scaler=target_scaler,
        input_window=90,       # 输入长度（如90天）
        output_window=365,      # 输出长度（短期预测90天）或365（长期预测）
        input_size=X_train_long.shape[2],  # 特征数量
        sample_indices=[0, 1, 2, 3],  # 需要绘图的样本索引
        hidden_size=256,
        num_layers=3,
        num_epochs=100,
        lr=1e-3,
        batch_size=64,
        device=device,
        dropout = 0.3,
        save_fig_path='final_prediction_plot6.png',
        save_csv_path='final_prediction_results6.csv'
    )
    torch.save(model.state_dict(), 'model_long.pth')
    joblib.dump(feature_scaler, 'feature_scaler_long.pkl')
    joblib.dump(target_scaler, 'target_scaler_long.pkl')
    print("长期预测的模型和归一化器已保存！")
