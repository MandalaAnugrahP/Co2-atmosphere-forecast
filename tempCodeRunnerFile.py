import matplotlib
matplotlib.use('Agg')  # Menghindari error Tkinter di Flask
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import io
import base64
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load data
df = pd.read_csv('machine-learning/NOAA_CO2 (4).csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Menetapkan frekuensi harian dan mengisi nilai yang hilang jika ada
df = df.asfreq('D')
df['PPM'].interpolate(method='linear', inplace=True)  # Interpolasi untuk mengisi data kosong

# Split data
train_size = int(len(df) * 0.9)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

# Train SARIMAX model
model = SARIMAX(train_data['PPM'], order=(0, 1, 0), seasonal_order=(1, 1, 0, 12))
model_fit = model.fit(disp=False)

# Prediksi untuk data uji
test_pred = model_fit.predict(start=len(train_data), end=len(df) - 1)

# Prediksi masa depan (30 hari ke depan)
future_steps = 30
last_date = test_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
future_pred = model_fit.forecast(steps=future_steps)
future_df = pd.DataFrame({'Date': future_dates, 'Predicted PPM': future_pred})
future_df.set_index('Date', inplace=True)

@app.route('/')
def home():
    # Buat plot hasil prediksi
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data['PPM'], label='Train Data (Aktual)', color='blue')
    plt.plot(test_data.index, test_data['PPM'], label='Test Data (Aktual)', color='green')
    plt.plot(test_data.index, test_pred, label='Prediksi SARIMAX (Test)', color='purple', linestyle='dotted')
    plt.plot(future_df.index, future_df['Predicted PPM'], label='Prediksi Masa Depan', color='red', linestyle='dashed')

    plt.xlabel('Date')
    plt.ylabel('CO2 Concentration (PPM)')
    plt.title('SARIMAX Model - Actual vs Predicted CO2 Concentration')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Simpan gambar plot ke format base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Tutup figure untuk menghindari memory leak
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Konversi prediksi ke dictionary agar bisa ditampilkan di template
    future_predictions = future_df['Predicted PPM'].to_dict()

    return render_template('index.html', plot_url=plot_url, future_predictions=future_predictions)

if __name__ == '__main__':
    app.run(debug=True)
