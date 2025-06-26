import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ccxt
import joblib
import numpy as np
import pandas as pd
import ta
import torch
from lightning.pytorch import LightningModule
from pykalman import KalmanFilter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from stable_baselines3 import PPO
import gymnasium as gym
import torch.nn.functional as F
import os
import time

def get_prediction_report():
    def get_btc_and_append_csv(filename='btc_data_15m.csv', return_days=7):
        # exchange = ccxt.binance()
        exchange = ccxt.okx()
        symbol = 'BTC/USDT'
        timeframe = '15m'
        limit = 1000

        now = exchange.milliseconds()
        seven_days_ago = now - 7 * 24 * 60 * 60 * 1000  # 7 days in ms

        # Если файл отсутствует — загружаем 7 дней
        if not os.path.exists(filename):
            print("Файл не найден. Загружаем полные 7 дней данных...")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=seven_days_ago, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Timestamp', inplace=True)
            df.to_csv(filename)
        else:
            existing_df = pd.read_csv(filename, parse_dates=['Timestamp'], index_col='Timestamp')
            last_timestamp_local = existing_df.index.max()

            # Получаем последнюю свечу с биржи
            ohlcv_latest = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            last_timestamp_remote = pd.to_datetime(ohlcv_latest[-1][0], unit='ms')

            # Если данные совпадают — ждём 30 сек и пробуем заново
            if last_timestamp_local >= last_timestamp_remote:
                print(f"[⏳] Последняя свеча ({last_timestamp_local}) уже в файле. Ждём 30 секунд...")
                time.sleep(30)
                return get_btc_and_append_csv(filename, return_days=return_days)

            print(f"Загружаем данные с {last_timestamp_local + pd.Timedelta(minutes=15)}")
            last_timestamp_ms = int(last_timestamp_local.timestamp() * 1000) + 1
            fetch_since = last_timestamp_ms

            max_candles = 1000
            all_new_data = []

            while fetch_since < now:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=max_candles)
                    if not ohlcv:
                        break
                    df_new = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'], unit='ms')
                    df_new.set_index('Timestamp', inplace=True)

                    df_new = df_new[~df_new.index.isin(existing_df.index)]
                    all_new_data.append(df_new)

                    fetch_since = int(df_new.index[-1].timestamp() * 1000) + 1
                    time.sleep(0.2)
                except Exception as e:
                    print("Ошибка при получении данных:", e)
                    break

            if all_new_data:
                full_new_data = pd.concat(all_new_data)
                updated_df = pd.concat([existing_df, full_new_data])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df.sort_index(inplace=True)
                updated_df.to_csv(filename)
            else:
                updated_df = existing_df

        # Возвращаем только последние return_days суток
        updated_df = pd.read_csv(filename, parse_dates=['Timestamp'], index_col='Timestamp')
        cutoff = updated_df.index.max() - pd.Timedelta(days=return_days)
        filtered_df = updated_df[updated_df.index >= cutoff]

        return filtered_df

    class TransformerBinaryClassifier(LightningModule):
        def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, lr=1e-3):
            super().__init__()
            self.save_hyperparameters()

            self.embedding = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(d_model, 1)
            self.lr = lr

        def forward(self, x):
            # x: (batch, seq_len, features)
            x = self.embedding(x)
            x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
            x = self.transformer(x)
            out = self.fc_out(x[-1])  # берем выход последнего шага последовательности
            return out.squeeze(-1)

        def training_step(self, batch, batch_idx):
            x, y, _ = batch
            x = x.float()  # принудительно
            y_hat = self(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y, _ = batch
            x = x.float()
            y_hat = self(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            self.log("val_loss", loss)
            return loss

        def test_step(self, batch, batch_idx):
            x, y, _ = batch
            x = x.float()
            y_hat = self(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            self.log("test_loss", loss)
            # можно вернуть предсказания, если нужно
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    class BTCTradingEnv(gym.Env):
        def __init__(self, df: pd.DataFrame, state_columns: list,
                     initial_balance=10_000, trade_penalty=0.0, max_steps=None,
                     reward_scaling=100.0, use_log_return=True, use_sharpe_bonus=True,
                     holding_penalty=0.001, sharpe_bonus_weight=0.1):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.state_columns = state_columns
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(state_columns),), dtype=np.float32
            )

            self.initial_balance = initial_balance
            self.trade_penalty = trade_penalty
            self.reward_scaling = reward_scaling
            self.max_steps = max_steps if max_steps is not None else len(df) - 1

            self.use_log_return = use_log_return
            self.use_sharpe_bonus = use_sharpe_bonus
            self.holding_penalty = holding_penalty
            self.sharpe_bonus_weight = sharpe_bonus_weight

            self.reset()

        def _next_observation(self):
            return self.df.iloc[self.current_step][self.state_columns].astype(np.float32).values

        def reset(self, *, seed=None, options=None):
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0
            self.entry_price = 0.0
            self.total_profit = 0.0
            self.trades = []
            self.trades_info = []
            self.entry_step = None
            self.total_reward = 0.0
            self.equity_curve = [self.balance]
            self.all_trades_info = []
            return self._next_observation(), {}

        def _calculate_equity(self, current_price):
            if self.position == 1:
                return self.balance + (current_price - self.entry_price)
            return self.balance

        def _calculate_max_drawdown(self):
            equity = np.array(self.equity_curve)
            if len(equity) < 2:
                return 0
            cumulative_max = np.maximum.accumulate(equity)
            drawdowns = (equity - cumulative_max) / cumulative_max
            return drawdowns.min()

        def step(self, action):
            done = False
            reward = 0.0
            current_price = self.df.loc[self.current_step, 'Close']

            # === Покупка ===
            if action == 1 and self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.entry_step = self.current_step

            # === Продажа ===
            elif action == 2 and self.position == 1:
                if self.use_log_return:
                    price_change = np.log(current_price / self.entry_price)
                else:
                    price_change = current_price - self.entry_price

                holding_duration = self.current_step - self.entry_step
                holding_penalty = self.holding_penalty * holding_duration

                reward = price_change - holding_penalty
                self.balance += reward

                self.trades.append(price_change)
                self.trades_info.append({
                    "entry_step": self.entry_step,
                    "exit_step": self.current_step,
                    "profit": price_change,
                    "holding": holding_duration
                })

                self.position = 0
                self.entry_price = 0.0
                self.entry_step = None

            # === Штраф за невалидное действие (например, продажа без позиции) ===
            elif action != 0:
                reward -= self.trade_penalty

            # === Масштабируем награду и обновляем total_reward ===
            reward *= self.reward_scaling
            self.total_reward += reward

            # === Equity ===
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)

            # === Шаг вперёд ===
            self.current_step += 1
            done = self.current_step >= self.max_steps

            # === Принудительно закрываем позицию, если эпизод завершился ===
            if done and self.position == 1:
                if self.use_log_return:
                    price_change = np.log(current_price / self.entry_price)
                else:
                    price_change = current_price - self.entry_price

                holding_duration = self.current_step - self.entry_step
                holding_penalty = self.holding_penalty * holding_duration

                final_reward = price_change - holding_penalty
                self.balance += final_reward

                self.trades.append(price_change)
                self.trades_info.append({
                    "entry_step": self.entry_step,
                    "exit_step": self.current_step,
                    "profit": price_change,
                    "holding": holding_duration
                })

                self.position = 0
                self.entry_price = 0.0
                self.entry_step = None

                self.total_reward += final_reward * self.reward_scaling

            # === Sharpe бонус ===
            if done and self.use_sharpe_bonus and len(self.trades) > 1:
                sharpe = np.mean(self.trades) / (np.std(self.trades) + 1e-8)
                reward += self.sharpe_bonus_weight * sharpe

            # === Обновляем список всех сделок (на каждом шаге) ===
            self.all_trades_info = self.trades_info.copy()

            # === Выдаём наблюдение и информацию ===
            obs = self._next_observation()
            info = {
                'balance': self.balance,
                'equity': equity,
                'position': self.position,
                'step': self.current_step,
                'reward': reward,
                'drawdown': self._calculate_max_drawdown(),
                'total_profit': np.sum(self.trades)
            }

            return obs, reward, done, False, info

        def render(self, mode='human', trades=False):
            print(f"Step: {self.current_step}, Position: {self.position}, Balance: {self.balance:.2f}, "
                  f"Total reward: {self.total_reward:.2f}, Max DD: {self._calculate_max_drawdown():.4f}")
            if trades and self.trades_info:
                print("Trades:")
                for trade in self.trades_info:
                    print(f"  Entry: {trade['entry_step']}, Exit: {trade['exit_step']}, "
                          f"Profit: {trade['profit']:.4f}, Holding: {trade['holding']} steps")

    def compute_indicators_v6(df, trend_emd=None, future_horizon=5, threshold=0.02, kalman_smooth=False):
        if kalman_smooth:

            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            df = df.copy()

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                smoothed, _ = kf.smooth(df[col].values)
                df[col] = smoothed.flatten()

        price_orig = df['Close']

        # Если передан тренд из EMD — использовать его как очищенную цену для индикаторов (для снижения шума)
        price = trend_emd if trend_emd is not None else price_orig

        o, h, l, c, v = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

        # --- Скользящие средние ---
        df['sma_1d'] = price
        df['sma_1w'] = price.rolling(7).mean()
        df['sma_signal'] = (df['sma_1d'] > df['sma_1w']).astype(int)

        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df['ema_crossover'] = (ema12 > ema26).astype(int)

        # --- RSI ---
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_signal'] = (df['rsi'] < 30).astype(int)

        # --- MACD ---
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_signal_bin'] = (macd > macd_signal).astype(int)

        # --- Volatility ---
        window_size = 7
        df['volatility_1d'] = price.rolling(window=window_size).std()
        median_vol = df['volatility_1d'].median()
        df['volatility_signal'] = (df['volatility_1d'] > median_vol).astype(int)
        vol_roll = df['volatility_1d'].rolling(14)
        df['volatility_z'] = (df['volatility_1d'] - vol_roll.mean()) / (vol_roll.std() + 1e-8)

        # --- Bollinger Bands ---
        bb = ta.volatility.BollingerBands(close=price, window=20, window_dev=2)
        df['bb_hband_indicator'] = bb.bollinger_hband_indicator()
        df['bb_lband_indicator'] = bb.bollinger_lband_indicator()

        # --- ATR ---
        df['atr'] = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()

        # --- On Balance Volume ---
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()

        # --- Stochastic RSI ---
        df['stoch_rsi'] = ta.momentum.StochasticOscillator(high=h, low=l, close=c, window=14).stoch()

        # --- Новые индикаторы ---

        # ADX - сила тренда
        df['adx'] = ta.trend.ADXIndicator(high=h, low=l, close=c, window=14).adx()

        # CCI - перепроданность/перекупленность
        df['cci'] = ta.trend.CCIIndicator(high=h, low=l, close=c, window=20).cci()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(high=h, low=l, close=c, lbp=14).williams_r()

        # Parabolic SAR
        df['psar'] = ta.trend.PSARIndicator(high=h, low=l, close=c, step=0.02, max_step=0.2).psar()

        # Momentum
        df['momentum'] = c - c.shift(10)

        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high=h, low=l, close=c, volume=v, window=20).chaikin_money_flow()

        # --- Свечные паттерны ---
        df['bull_candle'] = (c > o).astype(int)
        df['bear_candle'] = (c < o).astype(int)
        hl_range = h - l + 1e-8
        df['hammer'] = ((h - l > 3 * abs(o - c)) &
                        ((c - l) / hl_range > 0.6) &
                        ((o - l) / hl_range > 0.6)).astype(int)
        df['doji'] = (abs(c - o) <= 0.05 * hl_range).astype(int)
        df['shooting_star'] = ((h - l > 3 * abs(o - c)) &
                               ((h - c) / hl_range > 0.6) &
                               ((h - o) / hl_range > 0.6)).astype(int)

        prev_c, prev_o = c.shift(1), o.shift(1)
        df['bullish_engulfing'] = ((prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)).astype(int)
        df['bearish_engulfing'] = ((prev_c > prev_o) & (c < o) & (c < prev_o) & (o > prev_c)).astype(int)
        df['morning_star'] = ((df['bear_candle'].shift(2) == 1) &
                              (df['doji'].shift(1) == 1) &
                              (df['bull_candle'] == 1)).astype(int)
        df['evening_star'] = ((df['bull_candle'].shift(2) == 1) &
                              (df['doji'].shift(1) == 1) &
                              (df['bear_candle'] == 1)).astype(int)

        # --- Корреляции ---
        df['corr_price_volume_7'] = c.rolling(7).corr(v)
        df['corr_obv_price_7'] = df['obv'].rolling(7).corr(c)

        # --- Volume spike ---
        df['volume_spike'] = (v > 1.5 * v.rolling(14).mean()).astype(int)

        # --- Лаги ---
        lag_cols = ['Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'obv', 'stoch_rsi', 'adx', 'cci', 'williams_r',
                    'momentum', 'cmf']
        for col in lag_cols:
            for lag in range(1, 4):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # --- Целевая переменная ---
        df['future_return'] = df['Close'].shift(-future_horizon) / df['Close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)

        # --- Свечная кластеризация ---
        candle_features = pd.DataFrame({
            'body': abs(c - o),
            'upper_shadow': h - np.maximum(c, o),
            'lower_shadow': np.maximum(0, np.minimum(c, o) - l)
        }).replace([np.inf, -np.inf], 0).fillna(0)
        candle_scaled = StandardScaler().fit_transform(candle_features)
        kmeans = KMeans(n_clusters=6, random_state=42).fit(candle_scaled)
        df['candle_cluster'] = kmeans.labels_

        # --- Комбинированный сигнал ---
        signals = ['sma_signal', 'ema_crossover', 'rsi_signal', 'macd_signal_bin', 'volatility_signal']
        df['combined_signal'] = df[signals].sum(axis=1)

        # --- Заполнение пропусков ---
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        return df

    # --- Настройки ---
    TRANSFORMER_PATH = "transformer_binary_15min.pt"
    SCALER_PATH = "scaler_ttf.save"
    TRANSFORMER_LOG = "btc_model_predictions.csv"
    RL_LOG = "btc_rl_inference_log.csv"
    RL_MODEL_PATH = "ppo_btc_trading"

    # --- Загрузка данных и признаков ---
    df_new = get_btc_and_append_csv()
    df_indicators = compute_indicators_v6(df_new, kalman_smooth=True)

    df_model = df_indicators.drop(columns=["future_return", "High", "Close", "Low", "Open"]).copy()
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()
    scaler = joblib.load(SCALER_PATH)

    X = df_model.drop(columns=["target"])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = X_tensor.to(device)

    # --- Timestamp (следующий интервал) ---
    last_timestamp = df_new.index[-1]
    prediction_timestamp = pd.to_datetime(last_timestamp) + pd.Timedelta(minutes=15)

    # --- Трансформер инференс ---
    model = TransformerBinaryClassifier(input_size=X_scaled.shape[1])
    model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(X_tensor)  # Ожидается shape (batch_size=1, seq_len=..., 1) или (1, 1) — зависит от модели
        # Если output shape (1, 1, 1), убираем лишние измерения
        if output.dim() == 3:
            output = output.squeeze(0).squeeze(-1)  # (seq_len,) или (1,)
        elif output.dim() == 2:
            output = output.squeeze(-1)  # (seq_len,) или (batch_size,)

        # Берём последний timestep (прогноз на следующий интервал)
        last_pred_logit = output[-1]  # если seq_len > 1, или output[0] если один предсказанный шаг

        prob = torch.sigmoid(last_pred_logit).item()
        transformer_pred = int(prob > 0.5)

    # --- RL инференс ---
    df_model_rl = df_indicators.drop(columns=["future_return"]).copy()
    df_model_rl = df_model_rl.replace([np.inf, -np.inf], np.nan).dropna()


    excluded = ["future_return", "target"]
    state_columns = [col for col in df_model_rl.columns if col not in excluded]
    env = BTCTradingEnv(df_model_rl, state_columns)
    obs, _ = env.reset()
    rl_model = PPO.load(RL_MODEL_PATH, env=env)
    action, _ = rl_model.predict(obs, deterministic=True)

    # --- Обновление предыдущей строки (для обоих логов) ---
    def update_previous_row(csv_path, close_price, open_price):
        try:
            log_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        except FileNotFoundError:
            return

        if len(log_df) == 0:
            return

        real_class = int(close_price > open_price)
        log_df.at[log_df.index[-1], "real_price"] = close_price
        log_df.at[log_df.index[-1], "real_class"] = real_class
        log_df.to_csv(csv_path, index=False)

    update_previous_row(TRANSFORMER_LOG, df_model_rl["Close"].iloc[-1], df_model_rl["Open"].iloc[-1])
    update_previous_row(RL_LOG, df_model_rl["Close"].iloc[-1], df_model_rl["Open"].iloc[-1])

    # --- Добавление новой строки трансформера ---
    transformer_row = {
        "timestamp": prediction_timestamp,
        "prediction": transformer_pred,
        "probability": prob,
        "real_price": None,
        "real_class": None
    }
    try:
        pred_df = pd.read_csv(TRANSFORMER_LOG, parse_dates=["timestamp"])
    except FileNotFoundError:
        pred_df = pd.DataFrame(columns=transformer_row.keys())
    pred_df = pd.concat([pred_df, pd.DataFrame([transformer_row])], ignore_index=True)
    pred_df.to_csv(TRANSFORMER_LOG, index=False)

    # --- Добавление новой строки RL ---
    rl_row = {
        "timestamp": prediction_timestamp,
        "action": int(action),
        "probability": prob,  # или свой prob если считаешь RL отдельно
        "price": None,
        "real_class": None
    }
    try:
        rl_df = pd.read_csv(RL_LOG, parse_dates=["timestamp"])
    except FileNotFoundError:
        rl_df = pd.DataFrame(columns=rl_row.keys())
    rl_df = pd.concat([rl_df, pd.DataFrame([rl_row])], ignore_index=True)
    rl_df.to_csv(RL_LOG, index=False)

    print(f"[✓] Инференс завершён. Prediction Time: {prediction_timestamp}")

if __name__ == '__main__':
    print(get_prediction_report())