# File: trading_bot.py
import pandas as pd
import numpy as np
import requests
import pywavelets as pw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACDIndicator
from xgboost import XGBClassifier
import warnings
import telegram
import os

warnings.filterwarnings("ignore")

# =======================
# Konfigurasi Telegram
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'  # Ganti dengan token bot Telegram Anda
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID'  # Ganti dengan chat ID Anda

bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

def send_telegram_message(message):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print(f"Error mengirim pesan Telegram: {e}")

def send_telegram_photo(photo_path, caption=""):
    try:
        with open(photo_path, 'rb') as photo:
            bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption)
    except Exception as e:
        print(f"Error mengirim gambar ke Telegram: {e}")

# =======================
# Fungsi fetch data dari CoinGecko API
def fetch_data(days=1095):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Gagal mengunduh data: {e}")
        return pd.DataFrame()

# =======================
# Fungsi ekstraksi fitur dari segment harga
def extract_features(segment):
    try:
        # Analisis wavelet
        coeffs = pw.wavedec(segment, 'haar', level=3)
        energy_coeffs = [np.sum(c**2) for c in coeffs]
        # Indikator teknikal
        rsi = RSIIndicator(close=segment, window=14).rsi()
        macd = MACDIndicator(close=segment).macd_diff()
        rsi_latest = rsi.iloc[-1]
        macd_latest = macd.iloc[-1]
        # Gabung fitur
        features = energy_coeffs + [rsi_latest, macd_latest]
        return features
    except Exception as e:
        print(f"Error ekstraksi fitur: {e}")
        # Kembalikan fitur default jika error
        return [0]*5

# =======================
# Membuat dataset dan label
def prepare_dataset(df, window_size=30):
    X, y = [], []
    for start in range(0, len(df) - window_size, window_size):
        segment = df['price'].iloc[start:start + window_size]
        if len(segment) < window_size:
            continue
        feature_vector = extract_features(segment)
        start_price = segment.iloc[0]
        end_price = segment.iloc[-1]
        # Klasifikasi tren
        if end_price > start_price * 1.05:
            label = 1  # Impuls (tren naik)
        elif end_price < start_price * 0.95:
            label = 0  # Koreksi (tren turun)
        else:
            label = 2  # Sideways
        X.append(feature_vector)
        y.append(label)
    return np.array(X), np.array(y)

# =======================
# Melatih model dan menyimpan
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_scaled, y)
    # Simpan model dan scaler
    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    return model, scaler

# =======================
# Muat model dan scaler jika sudah ada
def load_model_and_scaler():
    try:
        model = joblib.load('xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        print("Model atau scaler belum ada. Harap latih terlebih dahulu.")
        return None, None

# =======================
# Prediksi pola terakhir dan visualisasi wavelet
def predict_pola(model, scaler, segment):
    try:
        feature_vector = extract_features(segment)
        feature_scaled = scaler.transform([feature_vector])
        prediction = model.predict(feature_scaled)[0]
        # Visualisasi wavelet coefficients
        coeffs = pw.wavedec(segment, 'haar', level=3)
        plt.figure(figsize=(12,8))
        for i, c in enumerate(coeffs):
            plt.subplot(len(coeffs), 1, i+1)
            plt.plot(c)
            plt.title(f'Wavelet Coeff Level {i+1}')
        plt.tight_layout()
        plt.savefig('wavelet_visualization.png')
        plt.close()
        return prediction
    except Exception as e:
        print(f"Error prediksi pola: {e}")
        return None

# =======================
# Main proses utama
if __name__ == "__main__":
    print("Memulai proses...")
    # Step 1: Unduh data
    df = fetch_data(days=1095)
    if df.empty:
        print("Data kosong, keluar.")
        send_telegram_message("Gagal mengunduh data. Script dihentikan.")
        exit()

    print(f"Data terakhir:\n{df.tail()}")

    # Step 2: Persiapkan dataset dan latih model jika belum ada
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        print("Melatih model baru...")
        X, y = prepare_dataset(df, window_size=30)
        if len(X) == 0:
            print("Data training tidak cukup.")
            send_telegram_message("Data training tidak cukup, script dihentikan.")
            exit()
        model, scaler = train_model(X, y)
        print("Model dan scaler disimpan.")

    # Evaluasi model
    X_all, y_all = prepare_dataset(df, window_size=30)
    X_scaled = scaler.transform(X_all)
    y_pred = model.predict(X_scaled)
    report = classification_report(y_all, y_pred, zero_division=0)
    print("Laporan Klasifikasi:\n", report)
    send_telegram_message(f"Pelatihan selesai.\n\nLaporan:\n{report}")

    # Step 3: Prediksi pola terbaru
    latest_segment = df['price'].iloc[-30:]
    prediction = predict_pola(model, scaler, latest_segment)
    if prediction is None:
        print("Gagal melakukan prediksi.")
        send_telegram_message("Gagal melakukan prediksi pola.")
        exit()

    latest_price = latest_segment.iloc[-1]
    # Step 4: Kirim rekomendasi berdasarkan prediksi
    if prediction == 1:
        message = (
            f"Pola prediksi: IMPULSIF (tren naik)\n"
            f"Order BUY: harga={latest_price:.2f}\n"
            f"SL (Stop Loss): {latest_price * 0.97:.2f}\n"
            f"TP (Take Profit): {latest_price * 1.05:.2f}"
        )
        send_telegram_message(message)
        print(message)
    elif prediction == 0:
        message = (
            f"Pola prediksi: KOREKSI (tren turun)\n"
            f"Order SELL: harga={latest_price:.2f}\n"
            f"SL (Stop Loss): {latest_price * 1.03:.2f}\n"
            f"TP (Take Profit): {latest_price * 0.95:.2f}"
        )
        send_telegram_message(message)
        print(message)
    else:
        message = "Pola prediksi: SIDEWAYS (stagnan), tidak ada order."
        send_telegram_message(message)
        print(message)

    # Kirim visual wavelet sebagai lampiran
    send_telegram_photo('wavelet_visualization.png', caption="Visualisasi Wavelet dari segmen terakhir.")
    print("Proses selesai.")
