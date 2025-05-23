import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR   = r"C:/Users/HP/Desktop/VAFSUD/merged_nifty50_data.csv"            # your raw CSVs: RELIANCE.csv, etc.
MODEL_DIR  = r'C:/Users/HP/Desktop/models/rnn/'        # will hold .h5 and scalers
PRED_DIR   = r'C:/Users/HP/Desktop/predictions/rnn/'   # will hold predictions CSVs

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
# 1. CONFIGURATION
tickers = [
    'RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK',
    'SBIN','LT','ITC','KOTAKBANK','HINDUNILVR',
    'AXISBANK','BHARTIARTL','HCLTECH','HDFC','ULTRACEMCO',
    'MARUTI','ONGC','POWERGRID','TITAN','WIPRO'
]


SEQ_LEN    = 60       # window size
TEST_RATIO = 0.2

# 2. TECHNICAL INDICATORS
def add_technical_indicators(df):
    df['MA_10']  = df['Close'].rolling(10).mean()
    df['MA_50']  = df['Close'].rolling(50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    delta = df['Close'].diff()
    gain  = delta.where(delta>0, 0).rolling(14).mean()
    loss  = -delta.where(delta<0, 0).rolling(14).mean()
    rs    = gain / loss
    df['RSI']   = 100 - (100 / (1 + rs))
    return df.dropna()

# 3. WINDOWING
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])   # predict the Close price
    return np.array(X), np.array(y)

# 4. MAIN LOOP
for ticker in tickers:
    print(f"\n=== {ticker} ===")
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.isfile(path):
        print("  CSV not found, skipping.")
        continue

    # -- a) load, sort, add indicators
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df = add_technical_indicators(df)

    # -- b) extract features & scale
    features = ['Close','MA_10','MA_50','EMA_10','RSI']
    data = df[features].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # save the scaler for real-time use
    with open(os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # -- c) create sequences
    X, y = create_sequences(data_scaled, SEQ_LEN)

    # align dates with y
    dates = df['Date'].values[SEQ_LEN:]

    # -- d) train/test split
    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test      = dates[split:]

    # reshape for RNN
    X_train = X_train.reshape((-1, SEQ_LEN, len(features)))
    X_test  = X_test.reshape((-1, SEQ_LEN, len(features)))

    # -- e) build model
    model = Sequential([
        SimpleRNN(50, input_shape=(SEQ_LEN, len(features))),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # -- f) train
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50, batch_size=32,
        callbacks=[es], verbose=1
    )

    # save model
    model.save(os.path.join(MODEL_DIR, f"{ticker}_rnn.h5"))

    # -- g) predict on test
    y_pred_scaled = model.predict(X_test)
    # we need to inverse transform just the Close column,
    # so pad predictions with zeros for the other 4 features
    pad = np.zeros((len(y_pred_scaled), len(features)-1))
    inv_input = np.hstack([y_pred_scaled, pad])
    y_pred = scaler.inverse_transform(inv_input)[:,0]
    y_true = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1,1), pad])
    )[:,0]

    # -- h) save predictions CSV
    out = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_true,
        'Predicted': y_pred
    })
    out_path = os.path.join(PRED_DIR, f"predictions_{ticker}.csv")
    out.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")

print("\nAll stocks done!")

# 5. OPTIONAL: example real-time forecast for NEXT DAY for a single ticker
def predict_next_day(ticker):
    """
    Load the saved model+scaler, read the last SEQ_LEN days
    from the original CSV, compute indicators, and predict
    the next day's Close price.
    """
    import tensorflow as tf

    # load data & indicators
    df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df = add_technical_indicators(df)

    # take only the last SEQ_LEN rows
    last = df.iloc[-SEQ_LEN:][['Close','MA_10','MA_50','EMA_10','RSI']].values

    # load scaler & model
    scaler = pickle.load(open(os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"), 'rb'))
    model  = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{ticker}_rnn.h5"))

    # scale, reshape, predict
    scaled = scaler.transform(last)
    x = scaled.reshape((1, SEQ_LEN, len(features)))
    pred_scaled = model.predict(x)
    inv = scaler.inverse_transform(np.hstack([pred_scaled, np.zeros((1,4))]))[0,0]
    return inv

# Usage:
# print("Tomorrow's forecast for TCS:", predict_next_day('TCS'))