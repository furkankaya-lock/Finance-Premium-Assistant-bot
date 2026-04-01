"""
ai_engine/lstm_model.py
=======================
Professional TensorFlow/Keras LSTM Price Direction Model
Architecture: Stacked Bidirectional LSTM + Attention + Residual Connections
Fallback: MLPRegressor (sklearn) if TensorFlow not available
Auto-detection: GPU if available, CPU otherwise

Output: 0.0 – 1.0 probability (>0.55 = bullish, <0.45 = bearish)
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor

log = logging.getLogger("CryptoBot.LSTM")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# TENSORFLOW AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers

    # Suppress TF logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")

    # GPU memory growth (prevents OOM)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    TF_AVAILABLE = True
    _backend = "TensorFlow " + tf.__version__ + (" + GPU" if gpus else " (CPU)")
    log.info(f"[LSTM] TensorFlow available: {_backend}")

except ImportError:
    log.warning("[LSTM] TensorFlow not found — using MLPRegressor fallback")
    log.warning("[LSTM] Install: pip install tensorflow==2.16.1")
    _backend = "MLPRegressor (fallback)"


# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────

FEATURES = [
    # Price
    "close", "open", "high", "low",
    # Volume
    "volume", "volume_ratio", "obv",
    # Trend
    "ema_9", "ema_21", "ema_50", "ema_200",
    "ema_trend", "golden_cross",
    # Momentum
    "rsi_14", "rsi_21", "macd", "macd_signal", "macd_hist",
    "momentum", "stoch_k", "stoch_d",
    # Volatility
    "atr_14", "volatility", "bb_pos", "bb_width",
    # Other
    "williams_r", "cci", "vwap",
    # Price changes
    "price_chg_1h", "price_chg_4h",
]

# Minimal feature set when some indicators are missing
FEATURES_MIN = [
    "close", "volume", "rsi_14", "macd", "macd_hist",
    "ema_9", "ema_21", "ema_50", "bb_pos", "bb_width",
    "momentum", "volume_ratio", "volatility", "atr_14",
    "stoch_k", "williams_r",
]


# ─────────────────────────────────────────────────────────────
# TENSORFLOW LSTM MODEL BUILDER
# ─────────────────────────────────────────────────────────────

def _build_tf_model(lookback: int, n_features: int,
                    units: int = 128,
                    dropout: float = 0.25,
                    learning_rate: float = 0.0005) -> "keras.Model":
    """
    Stacked Bidirectional LSTM with:
    - Residual connections
    - Batch Normalization
    - Attention mechanism
    - L2 regularization
    """
    inp = keras.Input(shape=(lookback, n_features), name="sequence_input")

    # ── Block 1: First Bidirectional LSTM ──
    x = layers.Bidirectional(
        layers.LSTM(units, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4)),
        name="bilstm_1"
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # ── Block 2: Second Bidirectional LSTM ──
    x2 = layers.Bidirectional(
        layers.LSTM(units // 2, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4)),
        name="bilstm_2"
    )(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(dropout)(x2)

    # ── Block 3: Third LSTM ──
    x3 = layers.LSTM(units // 4, return_sequences=True,
                     name="lstm_3")(x2)

    # ── Attention Mechanism ──
    # Self-attention: weigh timesteps by importance
    attn_scores = layers.Dense(1, activation="tanh")(x3)
    attn_weights = layers.Softmax(axis=1)(attn_scores)
    context = layers.Multiply()([x3, attn_weights])
    context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)

    # ── Residual: also take last timestep of block 2 ──
    last_step = layers.Lambda(lambda t: t[:, -1, :])(x2)

    # ── Merge ──
    merged = layers.Concatenate()([context, last_step])
    merged = layers.BatchNormalization()(merged)

    # ── Dense Head ──
    dense = layers.Dense(64, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4))(merged)
    dense = layers.Dropout(dropout * 0.6)(dense)
    dense = layers.Dense(32, activation="relu")(dense)
    dense = layers.Dropout(dropout * 0.4)(dense)

    # ── Output: sigmoid → price direction probability ──
    out = layers.Dense(1, activation="sigmoid", name="direction_prob")(dense)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ─────────────────────────────────────────────────────────────
# MLPRegressor FALLBACK MODEL
# ─────────────────────────────────────────────────────────────

def _build_mlp_model() -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=42,
        tol=1e-5,
    )


# ─────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────

class LSTMFiyatModeli:
    """
    Production-grade LSTM price direction model.

    TensorFlow available  → Stacked BiLSTM + Attention (full power)
    TensorFlow missing    → Enhanced MLPRegressor (fallback)

    Output of .tahmin():
    {
        "olasilik":   0.0 – 1.0,
        "yon":        "yukari" | "asagi" | "nötr",
        "guven":      0.0 – 1.0,
        "pred_fiyat": float,
        "guncel":     float,
        "backend":    str,
    }
    """

    LOOKBACK   = 48    # 48h of 1h candles → ~2 days context
    BATCH_SIZE = 32
    MAX_EPOCHS = 150

    def __init__(self, symbol: str):
        self.symbol     = symbol
        self.tf_model   = None      # TensorFlow model
        self.mlp_model  = None      # Fallback model
        self.scaler_x   = RobustScaler()   # Robust to outliers
        self.scaler_y   = MinMaxScaler(feature_range=(0, 1))
        self.features   = FEATURES
        self.egitildi   = False
        self.backend    = _backend
        self.metrics_   = {}

        self.tf_path    = os.path.join(MODEL_DIR, f"lstm_tf_{symbol}")
        self.mlp_path   = os.path.join(MODEL_DIR, f"lstm_mlp_{symbol}.pkl")
        self.scaler_path= os.path.join(MODEL_DIR, f"lstm_scalers_{symbol}.pkl")

        self._load()

    # ── FEATURE RESOLUTION ────────────────────────────────────

    def _resolve_features(self, df: pd.DataFrame) -> list:
        """Use full feature set if available, else minimal."""
        available = [f for f in FEATURES if f in df.columns]
        if len(available) >= 16:
            return available
        minimal = [f for f in FEATURES_MIN if f in df.columns]
        log.warning(f"[LSTM/{self.symbol}] Using minimal features ({len(minimal)} cols)")
        return minimal

    # ── TRAINING ──────────────────────────────────────────────

    def egit(self, df: pd.DataFrame) -> dict:
        """
        Train on historical OHLCV + indicator data.
        Minimum 150 candles required.
        """
        df = df.copy().reset_index(drop=True)
        self.features = self._resolve_features(df)
        df_clean = df.dropna(subset=self.features)

        if len(df_clean) < self.LOOKBACK + 30:
            log.warning(f"[LSTM/{self.symbol}] Insufficient data: {len(df_clean)} rows")
            return {"status": "insufficient_data", "rows": len(df_clean)}

        log.info(f"[LSTM/{self.symbol}] Training | rows:{len(df_clean)} | "
                 f"features:{len(self.features)} | backend:{self.backend}")

        X, y = self._prepare(df_clean)
        if X is None or len(X) < 40:
            return {"status": "insufficient_windows"}

        split = int(len(X) * 0.85)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        if TF_AVAILABLE:
            result = self._train_tensorflow(X_tr, y_tr, X_val, y_val)
        else:
            result = self._train_mlp(X_tr, y_tr, X_val, y_val)

        self.egitildi = True
        self.metrics_ = result
        self._save()

        log.info(
            f"[LSTM/{self.symbol}] Training complete | "
            f"val_acc:{result.get('val_acc', 0):.2%} | "
            f"direction_acc:{result.get('direction_acc', 0):.2%}"
        )
        return result

    def _train_tensorflow(self, X_tr, y_tr, X_val, y_val) -> dict:
        """Full BiLSTM training with early stopping and LR scheduling."""
        lookback, n_feat = X_tr.shape[1], X_tr.shape[2]

        self.tf_model = _build_tf_model(
            lookback=lookback,
            n_features=n_feat,
            units=128,
            dropout=0.25,
        )

        # Binary labels for direction classification
        y_tr_bin  = (y_tr > 0).astype(np.float32)
        y_val_bin = (y_val > 0).astype(np.float32)

        # Callbacks
        cbs = [
            callbacks.EarlyStopping(
                monitor="val_auc", patience=20,
                restore_best_weights=True, mode="max"
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=8, min_lr=1e-6, verbose=0
            ),
            callbacks.ModelCheckpoint(
                filepath=self.tf_path + "_best",
                monitor="val_auc", save_best_only=True,
                mode="max", verbose=0
            ),
        ]

        history = self.tf_model.fit(
            X_tr, y_tr_bin,
            validation_data=(X_val, y_val_bin),
            epochs=self.MAX_EPOCHS,
            batch_size=self.BATCH_SIZE,
            callbacks=cbs,
            verbose=0,
            class_weight=self._class_weights(y_tr_bin),
        )

        # Metrics
        val_loss   = min(history.history.get("val_loss", [1.0]))
        val_acc    = max(history.history.get("val_accuracy", [0.5]))
        val_auc    = max(history.history.get("val_auc", [0.5]))
        epochs_run = len(history.history["loss"])

        # Direction accuracy
        y_pred_raw = self.tf_model.predict(X_val, verbose=0).flatten()
        direction_acc = float(np.mean(
            (y_pred_raw > 0.5) == (y_val > 0)
        ))

        return {
            "status":        "ok",
            "backend":       "tensorflow",
            "val_loss":      round(float(val_loss), 4),
            "val_acc":       round(float(val_acc), 4),
            "val_auc":       round(float(val_auc), 4),
            "direction_acc": round(direction_acc, 4),
            "epochs":        epochs_run,
            "features":      len(self.features),
        }

    def _train_mlp(self, X_tr, y_tr, X_val, y_val) -> dict:
        """MLPRegressor fallback training."""
        n, lb, feat = X_tr.shape
        X_tr_flat  = X_tr.reshape(n, lb * feat)
        X_val_flat = X_val.reshape(len(X_val), lb * feat)

        self.mlp_model = _build_mlp_model()
        self.mlp_model.fit(X_tr_flat, y_tr.ravel())

        y_pred = self.mlp_model.predict(X_val_flat)
        direction_acc = float(np.mean(
            (y_pred > np.median(y_pred)) == (y_val > np.median(y_val))
        ))

        try:
            mape = mean_absolute_percentage_error(
                np.abs(y_val) + 1e-8, np.abs(y_pred) + 1e-8
            )
        except Exception:
            mape = 0.0

        return {
            "status":        "ok",
            "backend":       "mlp_fallback",
            "direction_acc": round(direction_acc, 4),
            "mape":          round(float(mape), 4),
            "features":      len(self.features),
        }

    # ── PREDICTION ────────────────────────────────────────────

    def tahmin(self, df: pd.DataFrame) -> dict:
        """
        Predict price direction from last LOOKBACK candles.
        Returns probability + direction + confidence.
        """
        # Auto-train if not trained
        if not self.egitildi:
            log.info(f"[LSTM/{self.symbol}] Not trained — auto-training...")
            result = self.egit(df)
            if result.get("status") != "ok":
                return self._neutral_result(df)

        df = df.copy().reset_index(drop=True)
        self.features = self._resolve_features(df)
        df_clean = df.dropna(subset=self.features)

        if len(df_clean) < self.LOOKBACK:
            return self._neutral_result(df_clean)

        # Get last LOOKBACK candles
        window_raw = df_clean[self.features].values[-self.LOOKBACK:]
        window_scaled = self.scaler_x.transform(window_raw)  # (LOOKBACK, features)

        current_price = float(df_clean["close"].iloc[-1])

        if TF_AVAILABLE and self.tf_model is not None:
            prob = self._predict_tf(window_scaled)
        elif self.mlp_model is not None:
            prob = self._predict_mlp(window_scaled)
        else:
            return self._neutral_result(df_clean)

        # Direction and confidence
        if prob > 0.55:
            yon   = "yukari"
        elif prob < 0.45:
            yon   = "asagi"
        else:
            yon   = "nötr"

        guven = abs(prob - 0.5) * 2  # 0.0 – 1.0

        # Predicted price (approximate)
        pred_change = (prob - 0.5) * 0.02  # max ±1% implied move
        pred_fiyat  = current_price * (1 + pred_change)

        return {
            "olasilik":   round(float(prob), 4),
            "yon":        yon,
            "guven":      round(float(guven), 4),
            "pred_fiyat": round(pred_fiyat, 4),
            "guncel":     round(current_price, 4),
            "backend":    self.backend,
            "features":   len(self.features),
        }

    def _predict_tf(self, window_scaled: np.ndarray) -> float:
        """TensorFlow prediction."""
        X = window_scaled[np.newaxis, :, :]  # (1, LOOKBACK, features)
        prob = float(self.tf_model.predict(X, verbose=0)[0, 0])
        return np.clip(prob, 0.0, 1.0)

    def _predict_mlp(self, window_scaled: np.ndarray) -> float:
        """MLP fallback prediction."""
        X_flat = window_scaled.flatten().reshape(1, -1)
        raw = float(self.mlp_model.predict(X_flat)[0])
        # Normalize to 0-1
        prob = 1 / (1 + np.exp(-raw * 50))
        return np.clip(prob, 0.0, 1.0)

    # ── PERFORMANCE REPORT ────────────────────────────────────

    def performance_report(self) -> dict:
        """Return training metrics and model info."""
        return {
            "symbol":    self.symbol,
            "backend":   self.backend,
            "trained":   self.egitildi,
            "lookback":  self.LOOKBACK,
            "features":  len(self.features),
            "metrics":   self.metrics_,
            "tf_available": TF_AVAILABLE,
        }

    def yeniden_egit(self, df: pd.DataFrame, force: bool = False) -> dict:
        """Force retrain (used for periodic retraining)."""
        if force:
            self.egitildi = False
            self.tf_model = None
            self.mlp_model = None
        return self.egit(df)

    # ── DATA PREPARATION ──────────────────────────────────────

    def _prepare(self, df: pd.DataFrame):
        """
        Create supervised learning dataset.
        X: (samples, LOOKBACK, features)
        y: future price change (signed)
        """
        data = df[self.features].values
        closes = df["close"].values

        # Fit scalers on all data
        self.scaler_x.fit(data)

        data_scaled = self.scaler_x.transform(data)

        X_list, y_list = [], []
        horizon = 3   # Predict 3-candle ahead direction

        for i in range(self.LOOKBACK, len(data) - horizon):
            window = data_scaled[i - self.LOOKBACK:i]  # (LOOKBACK, features)
            future_ret = (closes[i + horizon] - closes[i]) / closes[i]
            X_list.append(window)
            y_list.append(future_ret)

        if not X_list:
            return None, None

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        return X, y

    @staticmethod
    def _class_weights(y_bin: np.ndarray) -> dict:
        """Balance class weights for imbalanced datasets."""
        n_pos = np.sum(y_bin)
        n_neg = len(y_bin) - n_pos
        total = len(y_bin)
        w_pos = total / (2 * n_pos + 1e-8)
        w_neg = total / (2 * n_neg + 1e-8)
        return {0: float(w_neg), 1: float(w_pos)}

    def _neutral_result(self, df) -> dict:
        price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        return {
            "olasilik": 0.5, "yon": "nötr", "guven": 0.0,
            "pred_fiyat": price, "guncel": price,
            "backend": self.backend, "features": 0,
        }

    # ── PERSISTENCE ───────────────────────────────────────────

    def _save(self):
        """Save model + scalers."""
        try:
            # Save scalers
            with open(self.scaler_path, "wb") as f:
                pickle.dump({
                    "scaler_x": self.scaler_x,
                    "scaler_y": self.scaler_y,
                    "features": self.features,
                    "lookback": self.LOOKBACK,
                    "metrics":  self.metrics_,
                }, f)

            if TF_AVAILABLE and self.tf_model is not None:
                self.tf_model.save(self.tf_path)
                log.info(f"[LSTM/{self.symbol}] TF model saved → {self.tf_path}")
            elif self.mlp_model is not None:
                with open(self.mlp_path, "wb") as f:
                    pickle.dump(self.mlp_model, f)
                log.info(f"[LSTM/{self.symbol}] MLP model saved → {self.mlp_path}")

        except Exception as e:
            log.error(f"[LSTM/{self.symbol}] Save failed: {e}")

    def _load(self):
        """Load model + scalers if they exist."""
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    d = pickle.load(f)
                self.scaler_x = d["scaler_x"]
                self.scaler_y = d["scaler_y"]
                self.features = d.get("features", FEATURES_MIN)
                self.metrics_ = d.get("metrics", {})

                if TF_AVAILABLE and os.path.exists(self.tf_path):
                    self.tf_model = keras.models.load_model(self.tf_path)
                    self.egitildi = True
                    log.info(f"[LSTM/{self.symbol}] TF model loaded ✓")
                elif os.path.exists(self.mlp_path):
                    with open(self.mlp_path, "rb") as f:
                        self.mlp_model = pickle.load(f)
                    self.egitildi = True
                    log.info(f"[LSTM/{self.symbol}] MLP model loaded ✓")
        except Exception as e:
            log.warning(f"[LSTM/{self.symbol}] Load failed (will retrain): {e}")
            self.egitildi = False
