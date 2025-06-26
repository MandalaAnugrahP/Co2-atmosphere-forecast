from flask import Flask, render_template, send_file
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

# ===========================Load Data untuk LSTM =============================== #
df_lstm = pd.read_csv('machine-learning/CO2.csv')  # Ganti dengan path ke data LSTM Anda

# Proses data LSTM
dp = df_lstm.rename(columns={'Tahun': 'year', 'Bulan': 'month', 'Hari': 'day'})
dp['date'] = pd.to_datetime(dp[['year', 'month', 'day']])
dp = dp.drop(columns=['year', 'month', 'day', 'Decimal'])
dp = dp.set_index('date')

# Filter data untuk tahun 2010 hingga 2025
start_date = '2010-01-01'
end_date = '2025-12-31'
dp = dp[(dp.index >= start_date) & (dp.index <= end_date)]

# Interpolasi untuk mengisi nilai yang hilang
full_dp_range = pd.date_range(start=dp.index.min(), end=dp.index.max())
full_dp = pd.DataFrame(full_dp_range, columns=['date'])  # Ganti 'Date' menjadi 'date'
full_dp.set_index('date', inplace=True)  # Set index ke 'date'
merged_dp = pd.merge(full_dp, dp, how='left', left_index=True, right_index=True)  # Gunakan index untuk penggabungan
merged_dp['PPM'] = merged_dp['PPM'].interpolate(method='linear')

# Pastikan kolom PPM tidak memiliki nilai NaN setelah interpolasi
if merged_dp['PPM'].isnull().any():
    raise ValueError("Data PPM masih memiliki nilai NaN setelah interpolasi.")

# Normalisasi data untuk LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(merged_dp['PPM'].values.reshape(-1, 1))

# Membuat dataset untuk LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Tentukan jumlah langkah waktu
time_step = 30  # Misalnya, menggunakan 30 hari sebelumnya
X, y = create_dataset(scaled_data, time_step)

# Reshape input ke dalam format [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Membangun model LSTM dengan Keras
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))  # Output layer

# Kompilasi model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model_lstm.fit(X, y, epochs=100, batch_size=32)

# Menyimpan model LSTM
model_lstm.save('lstm_model.h5')

@app.route('/lstm')
def lstm():
    # Buat plot menggunakan Plotly untuk LSTM
    fig_lstm = go.Figure()
    
    # Tambahkan data train (warna biru)
    fig_lstm.add_trace(go.Scatter(x=merged_dp.index, y=merged_dp['PPM'],
                                   mode='lines', name='Test Data',
                                   line=dict(color='blue')))
    
    # Tambahkan data test (warna hijau) jika ada
    if 'test_data' in locals():  # Pastikan test_data ada
        fig_lstm.add_trace(go.Scatter(x=test_data.index, y=test_data['PPM'],
                                       mode='lines', name='Test Data',
                                       line=dict(color='green')))
    
    # Menghitung prediksi LSTM untuk masa depan
    last_data = scaled_data[-time_step:]  # Ambil data terakhir untuk prediksi
    predictions_lstm = []

    for _ in range(future_steps):
        last_data = last_data.reshape((1, time_step, 1))
        pred = model_lstm.predict(last_data)
        predictions_lstm.append(pred[0, 0])  # Ambil nilai prediksi
        last_data = np.append(last_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)  # Ubah pred menjadi 3D

    # Kembalikan ke skala asli
    predictions_lstm = scaler.inverse_transform(np.array(predictions_lstm).reshape(-1, 1))

    # Buat tanggal untuk prediksi masa depan
    future_dates_lstm = pd.date_range(start=merged_dp.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')

    # Tambahkan prediksi LSTM (warna merah)
    fig_lstm.add_trace(go.Scatter(x=future_dates_lstm, y=predictions_lstm.flatten(),
                                   mode='lines', name='Forecasting LSTM',
                                   line=dict(color='red')))
    
    # Format plot
    fig_lstm.update_layout(
        title="LSTM Model - Aktual Test vs Forecast CO2 Concentration",
        xaxis_title="Date",
        yaxis_title="CO2 Concentration (PPM)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Konversi plot ke HTML
    plot_html_lstm = pio.to_html(fig_lstm, full_html=False)
    
    # Tabel data peramalan
    future_df_lstm = pd.DataFrame({'Date': future_dates_lstm, 'Forecast PPM': predictions_lstm.flatten()})
    future_df_lstm.set_index('Date', inplace=True)
    
    return render_template('lstm.html', plot_html=plot_html_lstm, future_df=future_df_lstm.to_html())
# ========================= LSTM SELESAI ============================================#

# === Load Data GTCO2 === #
df_gtco2 = pd.read_csv('machine-learning/GTCO2.csv')

# Buat grafik GTCO2 menggunakan Plotly
fig_gtco2 = go.Figure()
fig_gtco2.add_trace(go.Scatter(x=df_gtco2['Year'], y=df_gtco2['Emissions'],
                               mode='lines+markers', name='Emissions (GtCO₂)',
                               line=dict(color='red')))
fig_gtco2.update_layout(
    title="GTCO₂ Emissions Over Time",
    xaxis_title="Year",
    yaxis_title="Emissions (GtCO₂)",
    template="plotly_white"
)
plot_html_gtco2 = pio.to_html(fig_gtco2, full_html=False)

# === Load Data Peramalan CO₂ Jangka Pendek === #
df = pd.read_csv('machine-learning/NOAA_CO2 (4).csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Menetapkan frekuensi harian dan mengisi nilai yang hilang jika ada
df = df.asfreq('D')
df['PPM'].interpolate(method='linear')  # Interpolasi untuk mengisi data kosong

# Split data untuk SARIMAX (90% train, 10% test)
train_size_sarimax = int(len(df) * 0.9)
train_data_sarimax, test_data_sarimax = df.iloc[:train_size_sarimax], df.iloc[train_size_sarimax:]

# Train SARIMAX model
model_sarimax = SARIMAX(train_data_sarimax['PPM'], order=(0, 1, 0), seasonal_order=(1, 1, 0, 12))
model_fit_sarimax = model_sarimax.fit(disp=False)

# Prediksi untuk data uji SARIMAX
test_pred_sarimax = model_fit_sarimax.predict(start=len(train_data_sarimax), end=len(df) - 1)

# Prediksi masa depan (30 hari ke depan) untuk SARIMAX
future_steps = 30
last_date = test_data_sarimax.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
future_pred_sarimax = model_fit_sarimax.forecast(steps=future_steps)
future_df_sarimax = pd.DataFrame({'Date': future_dates, 'Forecast PPM': future_pred_sarimax})
future_df_sarimax.set_index('Date', inplace=True)

# Split data untuk ARIMA (80% train, 20% test)
train_size_arima = int(len(df) * 0.8)
train_data_arima, test_data_arima = df.iloc[:train_size_arima], df.iloc[train_size_arima:]

# Train ARIMA model
model_arima = ARIMA(train_data_arima['PPM'], order=(50, 1, 40))  # Ganti dengan parameter ARIMA yang sesuai
model_fit_arima = model_arima.fit()

# Prediksi untuk data uji ARIMA
test_pred_arima = model_fit_arima.predict(start=len(train_data_arima), end=len(df) - 1)

# Prediksi masa depan (30 hari ke depan) untuk ARIMA
future_pred_arima = model_fit_arima.forecast(steps=future_steps)
future_df_arima = pd.DataFrame({'Date': future_dates, 'Forecast PPM': future_pred_arima})
future_df_arima.set_index('Date', inplace=True)

@app.route('/')
def home():
    return render_template('dashboard.html', plot_html_gtco2=plot_html_gtco2)

@app.route('/forecast')
def forecast():
    # Buat plot menggunakan Plotly untuk SARIMAX
    fig = go.Figure()
    
    # Tambahkan data train (warna biru)
    fig.add_trace(go.Scatter(x=train_data_sarimax.index, y=train_data_sarimax['PPM'],
                             mode='lines', name='Train Data (Aktual)',
                             line=dict(color='blue')))
    
    # Tambahkan data test (warna hijau)
    fig.add_trace(go.Scatter(x=test_data_sarimax.index, y=test_data_sarimax['PPM'],
                             mode='lines', name='Test Data (Aktual)',
                             line=dict(color='green')))
    
    # Tambahkan prediksi test (warna ungu, garis putus-putus)
    fig.add_trace(go.Scatter(x=test_data_sarimax.index, y=test_pred_sarimax,
                             mode='lines', name='Forecast SARIMAX (Test)',
                             line=dict(color='purple', dash='dot')))
    
    # Tambahkan prediksi masa depan (warna merah, garis putus-putus)
    fig.add_trace(go.Scatter(x=future_df_sarimax.index, y=future_df_sarimax['Forecast PPM'],
                             mode='lines', name='Forecast Masa Depan',
                             line=dict(color='red', dash='dash')))
    
    # Format plot
    fig.update_layout(
        title="SARIMAX Model - Actual vs Forecast CO2 Concentration",
        xaxis_title="Date",
        yaxis_title="CO2 Concentration (PPM)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Konversi plot ke HTML
    plot_html_sarimax = pio.to_html(fig, full_html=False)
    
    # Konversi prediksi masa depan ke dictionary agar bisa dipakai di template
    future_predictions = future_df_sarimax.to_dict()['Forecast PPM']
    
    return render_template('index.html', plot_html=plot_html_sarimax, future_predictions=future_predictions)

@app.route('/arima')
def arima():
    # Buat plot menggunakan Plotly untuk ARIMA
    fig_arima = go.Figure()
    
    # Tambahkan data train (warna biru)
    fig_arima.add_trace(go.Scatter(x=train_data_arima.index, y=train_data_arima['PPM'],
                                    mode='lines', name='Train Data (Aktual)',
                                    line=dict(color='blue')))
    
    # Tambahkan data test (warna hijau)
    fig_arima.add_trace(go.Scatter(x=test_data_arima.index, y=test_data_arima['PPM'],
                                    mode='lines', name='Test Data (Aktual)',
                                    line=dict(color='green')))
    
    # Tambahkan prediksi test (warna ungu, garis putus-putus)
    fig_arima.add_trace(go.Scatter(x=test_data_arima.index, y=test_pred_arima,
                                    mode='lines', name='Forecast ARIMA (Test)',
                                    line=dict(color='purple', dash='dot')))
    
    # Tambahkan prediksi masa depan (warna merah, garis putus-putus)
    fig_arima.add_trace(go.Scatter(x=future_df_arima.index, y=future_df_arima['Forecast PPM'],
                                    mode='lines', name='Forecast Masa Depan',
                                    line=dict(color='red', dash='dash')))
    
    # Format plot
    fig_arima.update_layout(
        title="ARIMA Model - Actual vs Forecast CO2 Concentration",
        xaxis_title="Date",
        yaxis_title="CO2 Concentration (PPM)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Konversi plot ke HTML
    plot_html_arima = pio.to_html(fig_arima, full_html=False)
    
    # Konversi prediksi masa depan ke dictionary agar bisa dipakai di template
    future_predictions_arima = future_df_arima.to_dict()['Forecast PPM']
    
    return render_template('arima.html', plot_html=plot_html_arima, future_predictions=future_predictions_arima)


# Route untuk download
@app.route('/download')
def download_page():
    return render_template('download.html')

@app.route('/download_data')
def download_data():
    # Simpan data asli ke file sementara
    file_path = "CO2_Actual_Data.csv"
    df.to_csv(file_path)
    return send_file(file_path, as_attachment=True)

@app.route('/download_predictions')
def download_predictions():
    # Simpan data prediksi SARIMAX ke file sementara
    file_path = "CO2_Forecast_SARIMA.csv"
    future_df_sarimax.to_csv(file_path)
    return send_file(file_path, as_attachment=True)

@app.route('/download_arima_predictions')
def download_arima_predictions():
    # Simpan data prediksi ARIMA ke file sementara
    file_path_arima = "CO2_ARIMA_Forecast.csv"
    future_df_arima.to_csv(file_path_arima)
    return send_file(file_path_arima, as_attachment=True)

@app.route('/register')
def register():
    return render_template('pendaftaran.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/keluar')
def keluar():
    return render_template('keluar.html')

if __name__ == '__main__':
    app.run(debug=True)
