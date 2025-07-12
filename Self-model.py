# ====== 基础依赖 ======
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

torch.manual_seed(42)
np.random.seed(42)

# ====== 1. 数据处理函数 ======

def create_daily_data(df):
    df_daily = pd.DataFrame()
    df_daily[['Global_active_power', 'Global_reactive_power', 'Sub_metering_1',
              'Sub_metering_2', 'Sub_metering_3']] = df[[
        'Global_active_power', 'Global_reactive_power',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].resample('D').sum()

    df_daily[['Voltage', 'Global_intensity']] = df[['Voltage', 'Global_intensity']].resample('D').mean()
    df_daily[['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']] = df[[
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']].resample('D').first()

    # 新增时间特征
    df_daily['weekday'] = df_daily.index.weekday  # 0-6
    df_daily['is_weekend'] = df_daily['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df_daily['month'] = df_daily.index.month
    df_daily['day_of_year'] = df_daily.index.dayofyear

    if df_daily.isna().sum().sum() / len(df_daily) > 0.05:
        df_daily = df_daily.interpolate(method='linear')
    else:
        df_daily = df_daily.dropna()
    return df_daily

def preprocess_train_data(train_path):
    column_names = ['DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    df = pd.read_csv(train_path, parse_dates=['DateTime'])
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)
    df[column_names[1:]] = df[column_names[1:]].apply(pd.to_numeric, errors='coerce')

    df_daily = create_daily_data(df)

    target_col = 'Global_active_power'
    feature_cols = [col for col in df_daily.columns if col != target_col]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = pd.DataFrame(feature_scaler.fit_transform(df_daily[feature_cols]),
                                   columns=feature_cols, index=df_daily.index)
    target_scaled = pd.DataFrame(target_scaler.fit_transform(df_daily[[target_col]]),
                                 columns=[target_col], index=df_daily.index)

    df_scaled = pd.concat([features_scaled, target_scaled], axis=1)

    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    joblib.dump(feature_cols, 'feature_columns.pkl')

    return df_scaled, feature_cols

def preprocess_test_data(test_path):
    column_names = ['DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    df = pd.read_csv(test_path, header=None, parse_dates=[0])
    df.columns = column_names
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)
    df[column_names[1:]] = df[column_names[1:]].apply(pd.to_numeric, errors='coerce')

    df_daily = create_daily_data(df)

    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')

    target_col = 'Global_active_power'
    features = df_daily.drop(columns=[target_col])

    features_scaled = pd.DataFrame(feature_scaler.transform(features),
                                   columns=features.columns, index=df_daily.index)
    target_scaled = pd.DataFrame(target_scaler.transform(df_daily[[target_col]]),
                                 columns=[target_col], index=df_daily.index)

    return pd.concat([features_scaled, target_scaled], axis=1)

def create_sliding_windows(data, input_window, output_window, target_col='Global_active_power'):
    X, y, trend = [], [], []
    feature_cols = [col for col in data.columns if col != target_col]

    data_X = data[feature_cols].values
    data_y = data[target_col].values

    for start in range(len(data) - input_window - output_window + 1):
        end_input = start + input_window
        end_output = end_input + output_window

        X.append(data_X[start:end_input])
        y.append(data_y[end_input:end_output])
        trend.append(data_y[start:end_input])  # 提取趋势用

    return np.array(X), np.array(y), np.array(trend)

# ====== 2. 模型定义（多尺度卷积+趋势+Decoder条件增强） ======

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        B, T, _ = x.size()
        pos = self.pe[:T].unsqueeze(0)
        return x + pos

class TransformerForecast(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_window, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.output_window = output_window

        # 趋势提取模块（GRU）
        self.trend_extractor = nn.GRU(input_size=1, hidden_size=d_model, batch_first=True)
        self.trend_fc = nn.Linear(d_model, output_window)

        # 多尺度卷积金字塔
        self.conv3 = nn.Conv1d(in_channels=input_size, out_channels=d_model // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=input_size, out_channels=d_model // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=input_size,
                               out_channels=d_model - 2 * (d_model // 3), kernel_size=7, padding=3)

        self.global_fc = nn.Sequential(nn.Linear(2, d_model), nn.ReLU())
        self.encoder_embedding = nn.LayerNorm(d_model)
        self.decoder_embedding = nn.Linear(1, d_model)
        self.pos_encoder = RelativePositionalEncoding(d_model)
        self.pos_decoder = RelativePositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, src, tgt, trend):
        # 趋势模块
        trend_in = trend.unsqueeze(-1)
        _, h = self.trend_extractor(trend_in)
        trend_out = self.trend_fc(h[-1])

        # 计算全局特征均值和标准差
        global_feat = torch.stack([src.mean(dim=[1, 2]), src.std(dim=[1, 2])], dim=-1)
        global_feat_emb = self.global_fc(global_feat).unsqueeze(1)  # (B,1,d_model)

        # 多尺度卷积提取特征
        src_perm = src.permute(0, 2, 1)  # (B, C, T)
        conv3_out = self.conv3(src_perm)
        conv5_out = self.conv5(src_perm)
        conv7_out = self.conv7(src_perm)
        src_conv = torch.cat([conv3_out, conv5_out, conv7_out], dim=1).permute(0, 2, 1)  # (B, T, d_model)

        # Encoder Embedding + 位置编码
        src_emb = self.encoder_embedding(src_conv + global_feat_emb)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)

        # Decoder条件建模增强
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        global_feat_dec = global_feat_emb.expand(-1, tgt_emb.size(1), -1)
        tgt_emb = tgt_emb + global_feat_dec  # 条件增强
        tgt_emb = self.pos_decoder(tgt_emb)

        output = self.decoder(tgt_emb, memory)
        res = self.fc(output).squeeze(-1)

        return res + trend_out

# ====== 3. 加权 MSE 损失函数 ======

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, input, target):
        diff = (input - target) ** 2
        weighted_diff = diff * self.weights
        return weighted_diff.mean()

# ====== 4. 训练与评估 ======

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, trend_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            X_batch, y_batch, trend_batch = X_batch.to(device), y_batch.to(device), trend_batch.to(device)

            decoder_input = torch.cat([torch.zeros(y_batch.size(0), 1, 1).to(device),
                                       y_batch[:, :-1].unsqueeze(-1)], dim=1)

            optimizer.zero_grad()
            output = model(X_batch, decoder_input, trend_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_val, y_val, trend_val in val_loader:
                X_val, y_val, trend_val = X_val.to(device), y_val.to(device), trend_val.to(device)
                decoder_input = torch.cat([torch.zeros(y_val.size(0), 1, 1).to(device),
                                           y_val[:, :-1].unsqueeze(-1)], dim=1)
                pred = model(X_val, decoder_input, trend_val)
                val_loss += criterion(pred, y_val).item() * X_val.size(0)

        print(f"Epoch {epoch + 1} - Train: {total_loss / len(train_loader.dataset):.6f}, "
              f"Val: {val_loss / len(val_loader.dataset):.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'improved_best_model_365.pth')


def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, trend in test_loader:
            X, y, trend = X.to(device), y.to(device), trend.to(device)
            decoder_input = torch.cat([torch.zeros(y.size(0), 1, 1).to(device),
                                       y[:, :-1].unsqueeze(-1)], dim=1)
            out = model(X, decoder_input, trend)

            # 反归一化
            out = scaler.inverse_transform(out.cpu().numpy())
            y_true_batch = scaler.inverse_transform(y.cpu().numpy())

            y_pred.append(out)
            y_true.append(y_true_batch)

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    # 三个指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    std = np.std(y_true - y_pred)

    return mse, mae, std, y_true, y_pred

# ====== 5. 主程序入口 ======
def main():
    INPUT_WINDOW = 90
    OUTPUT_WINDOW = 365
    BATCH_SIZE = 64
    EPOCHS = 200
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    LR = 1e-4
    DROPOUT = 0.2
    NUM_EXPERIMENTS = 5  # 5轮训练

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预处理数据
    train_data, feature_cols = preprocess_train_data("train.csv")
    test_data = preprocess_test_data("test.csv")

    # 滑窗构造
    X_train, y_train, trend_train = create_sliding_windows(train_data, INPUT_WINDOW, OUTPUT_WINDOW)
    X_test, y_test, trend_test = create_sliding_windows(test_data, INPUT_WINDOW, OUTPUT_WINDOW)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train),
                                            torch.FloatTensor(y_train),
                                            torch.FloatTensor(trend_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test),
                                           torch.FloatTensor(y_test),
                                           torch.FloatTensor(trend_test)),
                             batch_size=BATCH_SIZE, shuffle=False)

    # 存储每次实验的结果
    results = []

    for exp in range(NUM_EXPERIMENTS):
        print(f"\n=== 实验 {exp + 1}/{NUM_EXPERIMENTS} ===")

        # 每次实验重新初始化模型
        model = TransformerForecast(input_size=len(feature_cols), d_model=D_MODEL, nhead=NHEAD,
                                    num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW,
                                    dropout=DROPOUT).to(device)

        # 加权MSE损失权重
        weights = torch.linspace(1.0, 2.0, steps=OUTPUT_WINDOW).to(device)
        criterion = WeightedMSELoss(weights)

        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_model(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS)

        model.load_state_dict(torch.load('improved_best_model_365.pth'))
        target_scaler = joblib.load('target_scaler.pkl')
        mse, mae, std, y_true, y_pred = evaluate_model(model, test_loader, target_scaler, device)

        results.append({
            'mse': mse,
            'mae': mae,
            'std': std
        })

        print(f"\n实验 {exp + 1} 结果:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"STD: {std:.6f}")

        # 绘制最后一次实验的预测图
        if exp == NUM_EXPERIMENTS - 1:
            plt.figure(figsize=(14, 6))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            draw_len = y_true.shape[1]
            plt.plot(y_true[0][:draw_len], label='True')
            plt.plot(y_pred[0][:draw_len], '--', label='Pred')
            plt.title(f"Self-model预测 {draw_len} 天 (实验 {exp + 1})")
            plt.legend()
            plt.savefig("transformer_trend_prediction_improved_01.png")
            plt.show()

    # 输出所有实验的统计结果
    print("\n=== 最终统计结果(self_model 长期预测) ===")
    print(f"平均 MSE: {np.mean([r['mse'] for r in results]):.6f} ± {np.std([r['mse'] for r in results]):.6f}")
    print(f"平均 MAE: {np.mean([r['mae'] for r in results]):.6f} ± {np.std([r['mae'] for r in results]):.6f}")
    print(f"平均 STD: {np.mean([r['std'] for r in results]):.6f} ± {np.std([r['std'] for r in results]):.6f}")

if __name__ == '__main__':
    main()
