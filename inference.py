import warnings
from collections import deque

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ccxt
import joblib
import numpy as np
import pandas as pd
import ta
import torch
from pykalman import KalmanFilter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from stable_baselines3 import PPO
import gymnasium as gym
import torch.nn.functional as F
import os
import time
import json
from googleapiclient.discovery import build
import io
from datetime import datetime
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
import re

creds_data = json.loads(os.environ['GOOGLE_OAUTH_CREDENTIALS'])
creds = Credentials(
    token=creds_data.get("token"),
    refresh_token=creds_data.get("refresh_token"),
    token_uri=creds_data.get("token_uri"),
    client_id=creds_data.get("client_id"),
    client_secret=creds_data.get("client_secret"),
    scopes=creds_data.get("scopes"),
)

service = build('drive', 'v3', credentials=creds)

LOG_FOLDER_ID = "12WcA1K7_wR8eujJr7aGzMs_GJMfPPpYK"
LOCAL_LOG_DIR = "logs"
LOG_PATTERN = r"btc_rl_tail_.*\.json"

def get_latest_tail_log_json(service, folder_id=LOG_FOLDER_ID, local_dir=LOCAL_LOG_DIR, pattern=LOG_PATTERN):
    query = f"'{folder_id}' in parents and mimeType='application/json' and name contains 'btc_rl_tail_'"
    try:
        results = service.files().list(q=query,
                                       spaces='drive',
                                       fields="files(id, name, createdTime)",
                                       orderBy="createdTime desc").execute()
        items = results.get('files', [])
        if not items:
            print("Нет json-файлов в папке Google Drive.")
            return None

        for f in items:
            if re.fullmatch(pattern, f["name"]):
                file_id = f["id"]
                file_name = f["name"]
                break
        else:
            print("Нет файла, подходящего под паттерн.")
            return None

        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, file_name)

        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(local_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        print(f"[✓] Json-файл скачан локально: {local_path}")
        return local_path

    except HttpError as e:
        print(f"Ошибка Google Drive API: {e}")
        return None

def update_log_json_and_upload(service, new_entry: dict, local_dir=LOCAL_LOG_DIR, folder_id=LOG_FOLDER_ID):
    """
    Загружает последний json лог,
    обновляет последнюю запись реальными данными (если они есть в new_entry),
    добавляет новую запись (если это предсказание),
    сохраняет и загружает обновлённый файл обратно.
    """

    local_path = get_latest_tail_log_json(service, folder_id, local_dir)
    log_data = []

    if local_path and os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                print("[!] Ошибка чтения json, лог считается пустым.")
                log_data = []

    # Если в new_entry есть реальные данные — обновляем последнюю запись
    real_data_present = new_entry.get("real_price") is not None or new_entry.get("real_class") is not None
    if real_data_present and log_data:
        # Обновляем последнюю запись
        last_entry = log_data[-1]
        if new_entry.get("real_price") is not None:
            last_entry["real_price"] = new_entry["real_price"]
        if new_entry.get("real_class") is not None:
            last_entry["real_class"] = new_entry["real_class"]

        # Сохраняем изменения в последней записи
        log_data[-1] = last_entry
        print("[✓] Обновлена последняя запись реальными данными.")

    # Если это новое предсказание (обычно real_price и real_class пустые)
    if not real_data_present:
        # Добавляем новую запись
        log_data.append(new_entry)
        print("[✓] Добавлена новая запись с предсказанием.")

    # Сохраняем файл с временным именем
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_file_name = f"btc_rl_tail_{timestamp_str}.json"
    new_local_path = os.path.join(local_dir, new_file_name)

    with open(new_local_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    # Загружаем обратно на Google Drive
    file_metadata = {
        "name": new_file_name,
        "parents": [folder_id],
        "mimeType": "application/json"
    }
    media = MediaFileUpload(new_local_path, mimetype="application/json")

    # Можно сделать загрузку файла как создание нового
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    print(f"[✓] Лог загружен на Google Drive: {new_file_name}")


def get_latest_tail_log(folder_id="12WcA1K7_wR8eujJr7aGzMs_GJMfPPpYK", local_dir="logs", pattern=r"btc_rl_tail_.*\.csv"):
    # Запрос к Google Drive — получить все csv-файлы с нужной маской в папке
    query = f"'{folder_id}' in parents and mimeType='text/csv' and name contains 'btc_rl_tail_'"
    try:
        results = service.files().list(q=query,
                                       spaces='drive',
                                       fields="files(id, name, createdTime)",
                                       orderBy="createdTime desc").execute()
        items = results.get('files', [])
        if not items:
            raise FileNotFoundError("В указанной папке Google Drive нет файлов с нужной маской.")

        # Найти первый файл, который точно соответствует паттерну
        for f in items:
            if re.fullmatch(pattern, f["name"]):
                file_id = f["id"]
                file_name = f["name"]
                break
        else:
            raise FileNotFoundError("Нет файла, подходящего под паттерн.")

        # Создать локальную папку, если нет
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, file_name)

        # Скачиваем файл
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(local_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        print(f"[✓] Файл скачан локально: {local_path}")
        return local_path

    except HttpError as e:
        print(f"Ошибка Google Drive API: {e}")
        raise


def download_csv_old(file_id, local_path='temp.csv'):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.close()
    return local_path


def upload_file_by_id(file_id, filename, mimetype='text/csv'):
    media = MediaFileUpload(filename, mimetype=mimetype)
    service.files().update(
        fileId=file_id,
        media_body=media
    ).execute()
    print(f"[↻] Файл {filename} обновлён по ID: {file_id}")

upload_file_by_id('1aca3KCgbVKzAqGrNMRzcI5NClfBM5QYM', 'btc_model_predictions.csv')

# === Настройки ===
FOLDER_ID = "12WcA1K7_wR8eujJr7aGzMs_GJMfPPpYK"
TAIL_FILENAME_PREFIX = "btc_rl_tail"
LOCAL_TEMP_CSV = "btc_rl_tail_temp.csv"

# === Функция: получить ID последнего (по времени) файла в папке ===
def get_latest_file_id_in_folder(folder_id):
    query = f"'{folder_id}' in parents and mimeType='text/csv'"
    results = service.files().list(q=query, orderBy="createdTime desc", pageSize=1).execute()
    items = results.get("files", [])
    if not items:
        return None
    return items[0]["id"]

# === Функция: скачать файл по ID ===
def download_csv(file_id, local_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()

# === Функция: загрузить файл в папку с новым именем ===
def upload_csv_to_folder(local_path, folder_id):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{TAIL_FILENAME_PREFIX}_{timestamp}.csv"
    file_metadata = {
        "name": filename,
        "parents": [folder_id]
    }
    media = MediaFileUpload(local_path, mimetype="text/csv")
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    print(f"[↑] Файл загружен: {filename} → ID: {file.get('id')}")

def get_prediction_report():
    def get_btc_and_append_csv(filename='btc_data_15m.csv', return_days=7):
        # exchange = ccxt.binance()
        exchange = ccxt.okx()
        symbol = 'BTC/USDT'
        timeframe = '15m'
        limit = 1000

        now = exchange.milliseconds()
        seven_days_ago = now - 7 * 24 * 60 * 60 * 1000  # 7 days in ms

        download_csv_old('1ydQ_MmeqGNBvqVNWpMjIA4YXl1J9N6hq', 'btc_data_15m.csv')

        # Прочитать
        btc_data_15m = pd.read_csv('btc_data_15m.csv')
        # Если файл отсутствует — загружаем 7 дней
        if btc_data_15m.empty:
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
        upload_file_by_id('1ydQ_MmeqGNBvqVNWpMjIA4YXl1J9N6hq','btc_data_15m.csv')
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
        def __init__(self, df, state_columns,
                     initial_balance=10_000,
                     trade_penalty=0.01,
                     max_steps=None,
                     reward_scaling=100.0,
                     use_log_return=False,
                     use_sharpe_bonus=False,
                     holding_penalty=0.005,
                     sharpe_bonus_weight=0.5,
                     commission=0.001,
                     spread=0.0005,
                     slippage_std=0.001,
                     min_holding_period=8,
                     window_size=672):

            super().__init__()
            self.df = df.reset_index(drop=True)
            self.state_columns = state_columns
            self.window_size = window_size

            self.commission = commission
            self.spread = spread
            self.slippage_std = slippage_std
            self.min_holding_period = min_holding_period

            self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window_size, len(state_columns)),
                dtype=np.float32
            )

            self.initial_balance = initial_balance
            self.trade_penalty = trade_penalty
            self.reward_scaling = reward_scaling
            self.max_steps = max_steps if max_steps is not None else len(df) - 1

            self.use_log_return = use_log_return
            self.use_sharpe_bonus = use_sharpe_bonus
            self.holding_penalty = holding_penalty
            self.sharpe_bonus_weight = sharpe_bonus_weight

            self.all_trades_info = []
            self.episode_trades_info = []

            self.reset()

        def _get_execution_price(self, price, action):
            if action == 1:
                exec_price = price * (1 + self.spread)
            elif action == 2:
                exec_price = price * (1 - self.spread)
            else:
                exec_price = price

            slippage = np.random.normal(0, self.slippage_std)
            exec_price *= (1 + slippage)
            return max(exec_price, 0.0001)

        def _next_observation(self):
            obs = self.df.iloc[self.current_step][self.state_columns].astype(np.float32).values
            obs = np.nan_to_num(obs)
            self.state_window.append(obs)
            return np.array(self.state_window)

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

        def reset(self, *, seed=None, options=None):
            self.current_step = self.window_size
            self.balance = self.initial_balance
            self.position = 0
            self.entry_price = 0.0
            self.entry_step = None
            self.total_reward = 0.0
            self.trades = []
            self.trades_info = []
            self.equity_curve = [self.balance]
            self.episode_trades_info = []
            self.actions_log = []
            self.state_window = deque(maxlen=self.window_size)

            for i in range(self.current_step - self.window_size, self.current_step):
                obs = self.df.iloc[i][self.state_columns].astype(np.float32).values
                self.state_window.append(np.nan_to_num(obs))

            return self._next_observation(), {}

        def step(self, action):
            done = False
            reward = 0.0

            market_price = self.df.loc[self.current_step, 'Close']
            if np.isnan(market_price) or market_price <= 0:
                market_price = 1.0

            max_holding_period = 96
            if self.position == 1 and (self.current_step - self.entry_step) >= max_holding_period:
                action = 2  # форсированная продажа

            exec_price = self._get_execution_price(market_price, action)

            # --- BUY ---
            if action == 1 and self.position == 0:
                commission_cost = exec_price * self.commission
                reward += 0.1
                self.position = 1
                self.entry_price = exec_price + commission_cost
                self.entry_step = self.current_step
                self.balance -= commission_cost

            # --- SELL ---
            elif action == 2 and self.position == 1:
                commission_cost = exec_price * self.commission
                price_change = (exec_price - self.entry_price) / self.entry_price
                holding_duration = self.current_step - self.entry_step
                reward = price_change - self.commission * 2

                if holding_duration < self.min_holding_period:
                    reward -= 0.1
                if reward > 0:
                    reward += 0.2

                self.balance += price_change * self.entry_price
                self.balance -= commission_cost

                trade_info = {
                    "entry_step": self.entry_step,
                    "exit_step": self.current_step,
                    "profit": price_change - self.commission * 2,
                    "holding": holding_duration,
                    "entry_price": self.entry_price,
                    "exit_price": exec_price
                }
                self.trades.append(price_change - self.commission * 2)
                self.episode_trades_info.append(trade_info)
                self.all_trades_info.append(trade_info)

                self.position = 0
                self.entry_price = 0.0
                self.entry_step = None

            # --- Bad actions ---
            elif action == 2 and self.position == 0:
                reward -= 0.005
            elif action == 1 and self.position == 1:
                reward -= 0.005

            # --- HOLD ---
            elif action == 0:
                if self.position == 1:
                    # Нереализованная прибыль (награждаем за выгодные удержания)
                    unrealized = (market_price - self.entry_price) / self.entry_price
                    reward += unrealized * 0.1

                    # Штраф за слишком долгий холд
                    holding_duration = self.current_step - self.entry_step
                    if holding_duration > self.min_holding_period:
                        reward -= self.holding_penalty * max(0, holding_duration - self.min_holding_period)
                else:
                    # Находимся вне позиции и просто холдим — штрафуем слегка
                    reward -= 0.001

            reward = np.clip(reward, -100.0, 100.0)
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0

            self.total_reward += reward
            equity = self._calculate_equity(market_price)
            self.equity_curve.append(equity)

            self.current_step += 1
            done = self.current_step >= self.max_steps

            # --- Force close if still holding ---
            if done and self.position == 1:
                exec_price = self._get_execution_price(market_price, 2)
                commission_cost = exec_price * self.commission
                final_price_change = (exec_price - self.entry_price) / self.entry_price - self.commission * 2
                final_reward = final_price_change

                self.balance += final_price_change * self.entry_price
                self.balance -= commission_cost

                trade_info = {
                    "entry_step": self.entry_step,
                    "exit_step": self.current_step,
                    "profit": final_price_change,
                    "holding": self.current_step - self.entry_step,
                    "entry_price": self.entry_price,
                    "exit_price": exec_price
                }
                self.trades.append(final_price_change)
                self.episode_trades_info.append(trade_info)
                self.all_trades_info.append(trade_info)
                self.total_reward += final_reward * self.reward_scaling
                self.position = 0

            # --- Sharpe Bonus ---
            if done and self.use_sharpe_bonus and len(self.trades) > 1:
                sharpe = np.mean(self.trades) / (np.std(self.trades) + 1e-8)
                reward += self.sharpe_bonus_weight * sharpe

            self.actions_log.append(action)

            obs = self._next_observation()
            info = {
                'balance': self.balance,
                'equity': equity,
                'position': self.position,
                'step': self.current_step,
                'reward': reward,
                'drawdown': self._calculate_max_drawdown(),
                'total_profit': np.sum(self.trades),
                'recent_trade': self.episode_trades_info[-1] if self.episode_trades_info else None,
                'combined_signal': self.df.loc[
                    self.current_step, 'combined_signal'] if 'combined_signal' in self.df.columns else None,
                'rsi': self.df.loc[self.current_step, 'rsi'] if 'rsi' in self.df.columns else None,
                'macd': self.df.loc[self.current_step, 'macd'] if 'macd' in self.df.columns else None,
                'volume_spike': self.df.loc[
                    self.current_step, 'volume_spike'] if 'volume_spike' in self.df.columns else None,
                'candle_cluster': self.df.loc[
                    self.current_step, 'candle_cluster'] if 'candle_cluster' in self.df.columns else None,
                'entry_price': self.entry_price if self.position == 1 else None,
                'unrealized_profit': (
                            (market_price - self.entry_price) / self.entry_price) if self.position == 1 else 0.0,
                'holding_duration': (self.current_step - self.entry_step) if self.position == 1 else 0,
                'actions_log': self.actions_log[-100:]  # последние действия
            }

            return obs, reward, done, False, info

    class TransformerFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, d_model=64, nhead=4, num_layers=2, dropout=0.1):
            # Определим размеры входа
            seq_len, feature_dim = observation_space.shape  # (window_size, num_features)

            super().__init__(observation_space, features_dim=d_model)

            self.seq_len = seq_len
            self.feature_dim = feature_dim

            # Линейный слой для увеличения размерности признаков до d_model
            self.input_proj = nn.Linear(feature_dim, d_model)

            self.norm = nn.LayerNorm(d_model)

            # Трансформер-энкодер
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Агрегация выходов трансформера (например, берём последний токен)
            self.pool = nn.AdaptiveAvgPool1d(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch_size, seq_len, feature_dim)
            x = self.input_proj(x)  # -> (batch_size, seq_len, d_model)
            x = self.norm(x)  # 👈 после линейного слоя
            x = self.transformer_encoder(x)  # -> (batch_size, seq_len, d_model)

            # Берём среднее по временной оси: (batch_size, d_model)
            x = x.mean(dim=1)
            return x

    class TransformerPolicy(ActorCriticPolicy):
        def __init__(self, observation_space, action_space, lr_schedule,
                     net_arch=None, activation_fn=nn.Tanh, **kwargs):
            # Заменим feature_extractor на наш TransformerFeatureExtractor
            super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                features_extractor_class=TransformerFeatureExtractor,
                features_extractor_kwargs=dict(d_model=64, nhead=4, num_layers=2),
                net_arch=[dict(pi=[64], vf=[64])],
                activation_fn=activation_fn,
                **kwargs
            )
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
    RL_LOG = "btc_rl_inference_log_v2.csv"
    RL_MODEL_PATH = "ppo_btc_trading_v18"

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
    obs_before = obs.copy()
    try:
        obs, reward, done, _, info = env.step(action)
        next_obs = obs.copy()
    except IndexError:
        print("⚠️  Достигнут конец df, step() невозможен.")
        reward = 0.0
        done = True
        info = {"error": "IndexError on step()"}
        next_obs = obs_before.copy()

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

    download_csv_old('1aca3KCgbVKzAqGrNMRzcI5NClfBM5QYM', 'btc_model_predictions.csv')
    update_previous_row(TRANSFORMER_LOG, df_model_rl["Close"].iloc[-1], df_model_rl["Open"].iloc[-1])



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



    # === Основная функция: обновить и сохранить двухстрочный лог ===
    def update_rl_log_tail_to_drive(log_path: str, new_row_dict: dict):
        """
        Обновляет последний лог-файл, добавляя новую строку рядом с последней.
        Затем сохраняет в новый файл с текущим временем и загружает на Google Drive.
        """

        if not os.path.exists(log_path):
            print(f"[Ошибка] Файл не найден: {log_path}")
            return

        try:
            rl_df = pd.read_csv(log_path)

            if rl_df.empty:
                print(f"[Предупреждение] Лог пустой. Добавим только новую строку.")
                df_tail = pd.DataFrame([new_row_dict])
            else:
                last_row = rl_df.iloc[-1].to_dict()
                df_tail = pd.DataFrame([last_row, new_row_dict])

            # Сохраняем новый файл с временным именем
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            temp_log_path = f"btc_rl_tail_{timestamp_str}.csv"
            df_tail.to_csv(temp_log_path, index=False)
            print(f"[✓] Лог обновлён: {temp_log_path}")

            # Загружаем на Google Drive
            upload_csv_to_folder(temp_log_path, FOLDER_ID)

        except Exception as e:
            print(f"[Ошибка] Не удалось обновить лог: {e}")


    # new_row = {
    #     "timestamp": prediction_timestamp,
    #     "action": int(action),
    #     "reward": float(reward),
    #     "done": done,
    #     "obs": obs_before.tolist(),
    #     "next_obs": next_obs.tolist(),
    #     "real_price": None,
    #     "entry_price": info.get("entry_price", None),
    #     "unrealized_profit": info.get("unrealized_profit", None),
    #     "position_before": info.get("position", 0),
    #     "close_price": df_model_rl["Close"].iloc[-1],
    #     "open_price": df_model_rl["Open"].iloc[-1],
    # }
    #
    # latest_log_path = get_latest_tail_log()
    # update_rl_log_tail_to_drive(latest_log_path, new_row)

    new_row = {
        "timestamp": prediction_timestamp.isoformat(),  # или строка с датой
        "action": int(action),
        "reward": float(reward),
        "done": done,
        "obs": obs_before.tolist(),
        "next_obs": next_obs.tolist(),
        "real_price": None,  # пока нет реальных данных
        "entry_price": info.get("entry_price", None),
        "unrealized_profit": info.get("unrealized_profit", None),
        "position_before": info.get("position", 0),
        "close_price": df_model_rl["Close"].iloc[-1],
        "open_price": df_model_rl["Open"].iloc[-1],
        "real_class": None
    }

    # Сохраняем предсказание (без реальных данных)
    update_log_json_and_upload(service, new_row)

    real_close_price = df_indicators["Close"].iloc[-1]
    real_open_price = df_indicators["Open"].iloc[-1]

    # Через 15 минут при получении реальных данных:
    real_update = {
        "timestamp": new_row["timestamp"],  # точно такой же timestamp, чтобы обновить
        "real_price": real_close_price,
        "real_class": int(real_close_price > real_open_price),
    }

    update_log_json_and_upload(service, real_update)

    print(f"[✓] Инференс завершён. Prediction Time: {prediction_timestamp}")

if __name__ == '__main__':
    print(get_prediction_report())