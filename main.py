"""
Pattern Recognition for Financial Time Series Forecasting
Stocks: Reliance Industries (RELIANCE.NS), TCS (TCS.NS), Infosys (INFY.NS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS     = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
NAMES       = ["Reliance Industries", "TCS", "Infosys"]
START       = "2018-01-01"
END         = "2024-01-01"
WINDOW_LEN  = 30        # STFT window length (trading days)
HOP_SIZE    = 5         # STFT hop size
PRED_STEPS  = 5         # predict 5 days ahead
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# TASK 1: DATA PREPARATION
# ─────────────────────────────────────────────

def fetch_data(tickers, start, end):
    """Download and align closing prices for all tickers."""
    print("\n[Task 1] Fetching stock data...")
    series_list = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        # yfinance 1.2.0 may return MultiIndex columns — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        close.name = ticker
        series_list.append(close)
        print(f"  {ticker}: {len(close)} trading days")

    # Align to common dates
    combined = pd.concat(series_list, axis=1).dropna()
    print(f"  Common trading days after alignment: {len(combined)}")
    return combined

def normalize(df):
    """Min-max normalize each column independently."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler

def plot_timeseries(df, names):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Task 1 — Normalized Stock Price Time Series", fontsize=14, fontweight="bold")
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for i, (col, name) in enumerate(zip(df.columns, names)):
        axes[i].plot(df.index, df[col], color=colors[i], linewidth=1.2)
        axes[i].set_ylabel(name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig("figures/01_timeseries.png", dpi=150)
    plt.close()
    print("  Saved: figures/01_timeseries.png")

# ─────────────────────────────────────────────
# TASK 2: SIGNAL PROCESSING
# ─────────────────────────────────────────────

def compute_fft(signal):
    """Compute one-sided FFT magnitude spectrum."""
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(N)
    magnitude = np.abs(fft_vals) / N
    return freqs, magnitude

def compute_stft(signal, window_len, hop_size):
    """
    Manual sliding-window STFT.
    Returns:
        times      : array of window center indices
        freqs      : frequency bin indices (0 to window_len//2)
        spectrogram: 2D array [freq_bins x time_frames]
    """
    n_freqs = window_len // 2 + 1
    starts  = range(0, len(signal) - window_len, hop_size)
    spec    = []
    times   = []
    for s in starts:
        segment = signal[s : s + window_len]
        window  = np.hanning(window_len)
        windowed = segment * window
        ft = np.fft.rfft(windowed)
        spec.append(np.abs(ft) ** 2)   # power spectrum
        times.append(s + window_len // 2)
    spectrogram = np.array(spec).T     # [freq_bins x time_frames]
    freqs = np.arange(n_freqs)
    return np.array(times), freqs, spectrogram

def plot_signal_analysis(df, names):
    """Plot FFT spectrum and STFT spectrogram for each stock."""
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # ── Frequency Spectra
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Task 2 — Frequency Spectrum (FFT)", fontsize=13, fontweight="bold")
    for i, (col, name) in enumerate(zip(df.columns, names)):
        freqs, mag = compute_fft(df[col].values)
        axes[i].plot(freqs, mag, color=colors[i], linewidth=1)
        axes[i].set_title(name)
        axes[i].set_xlabel("Normalized Frequency")
        axes[i].set_ylabel("Magnitude")
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/02_fft_spectrum.png", dpi=150)
    plt.close()
    print("  Saved: figures/02_fft_spectrum.png")

    # ── Spectrograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Task 2 — STFT Spectrograms", fontsize=13, fontweight="bold")
    for i, (col, name) in enumerate(zip(df.columns, names)):
        times, freqs, spec = compute_stft(df[col].values, WINDOW_LEN, HOP_SIZE)
        spec_db = 10 * np.log10(spec + 1e-10)   # convert to dB
        im = axes[i].imshow(
            spec_db, aspect="auto", origin="lower",
            cmap="inferno", extent=[0, len(times), 0, len(freqs)]
        )
        axes[i].set_title(name)
        axes[i].set_xlabel("Time Frame")
        axes[i].set_ylabel("Frequency Bin")
        fig.colorbar(im, ax=axes[i], label="Power (dB)")
    plt.tight_layout()
    plt.savefig("figures/03_spectrograms.png", dpi=150)
    plt.close()
    print("  Saved: figures/03_spectrograms.png")

# ─────────────────────────────────────────────
# TASK 3: MODEL DEVELOPMENT
# ─────────────────────────────────────────────

def build_spectrogram_dataset(df, window_len, hop_size, pred_steps):
    """
    For each stock: generate (spectrogram_image, future_price) pairs.
    Spectrogram image shape: [1, freq_bins, time_frames_per_window]
    """
    X_all, y_all = [], []

    for col in df.columns:
        signal = df[col].values
        # Slide a "meta-window" across the full signal.
        # Each sample = spectrogram of a local chunk → predict price pred_steps ahead.
        chunk = window_len * 4   # chunk size for one spectrogram
        for start in range(0, len(signal) - chunk - pred_steps, hop_size):
            segment = signal[start : start + chunk]
            _, _, spec = compute_stft(segment, window_len, hop_size)
            # Resize to fixed shape for CNN input
            target_t, target_f = 16, 16
            # Downsample spectrogram to fixed size
            from scipy.ndimage import zoom
            fy = target_f / spec.shape[0]
            fx = target_t / spec.shape[1]
            spec_resized = zoom(spec, (fy, fx))
            # Normalize spectrogram
            s_min, s_max = spec_resized.min(), spec_resized.max()
            if s_max - s_min > 1e-8:
                spec_resized = (spec_resized - s_min) / (s_max - s_min)
            X_all.append(spec_resized[np.newaxis, :, :])   # [1, F, T]
            # Target: price pred_steps ahead of the chunk
            future_idx = start + chunk + pred_steps - 1
            y_all.append(signal[future_idx])

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    print(f"  Dataset: {X.shape[0]} samples, spectrogram shape {X.shape[1:]}")
    return X, y

class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SpectrogramCNN(nn.Module):
    """
    CNN that takes a single-channel spectrogram image [1, F, T]
    and predicts the next stock price (regression).
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # [16, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [16, 8, 8]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [32, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [32, 4, 4]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [64, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))                  # [64, 2, 2]
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.regressor(self.features(x)).squeeze(-1)

def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_run = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_run += criterion(pred, yb).item() * len(yb)
        val_loss = val_run / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    return train_losses, val_losses

def plot_cnn_architecture():
    """Draw a simple CNN architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.suptitle("Task 3 — CNN Architecture", fontsize=13, fontweight="bold")

    layers = [
        ("Input\n[1×16×16]",    0.5,  "#90CAF9"),
        ("Conv2d+BN\n16 filters", 2.2, "#64B5F6"),
        ("MaxPool\n[16×8×8]",   3.9,  "#42A5F5"),
        ("Conv2d+BN\n32 filters", 5.6, "#2196F3"),
        ("MaxPool\n[32×4×4]",   7.3,  "#1E88E5"),
        ("Conv2d+BN\n64 filters", 9.0, "#1565C0"),
        ("AvgPool\n[64×2×2]",  10.7,  "#0D47A1"),
        ("FC 128\n+ Dropout",  12.0,  "#FF7043"),
        ("Output\n(price)",    13.3,  "#FF5722"),
    ]
    for label, x, color in layers:
        rect = plt.Rectangle((x - 0.55, 1.0), 1.1, 2.0,
                              facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, 2.0, label, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
        if x < 13.3:
            ax.annotate("", xy=(x + 0.65, 2.0), xytext=(x + 0.55, 2.0),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    plt.tight_layout()
    plt.savefig("figures/04_cnn_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/04_cnn_architecture.png")

# ─────────────────────────────────────────────
# TASK 4: ANALYSIS
# ─────────────────────────────────────────────

def evaluate_and_plot(model, val_loader, train_losses, val_losses):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    mse  = mean_squared_error(all_true, all_preds)
    rmse = np.sqrt(mse)
    print(f"\n[Task 4] Evaluation on validation set:")
    print(f"  MSE  = {mse:.6f}")
    print(f"  RMSE = {rmse:.6f}")

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle("Task 4 — Model Evaluation", fontsize=13, fontweight="bold")

    # Prediction vs Actual
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(all_true[:200],  label="Actual",    color="#2196F3", linewidth=1.2)
    ax1.plot(all_preds[:200], label="Predicted", color="#FF5722", linewidth=1.2, linestyle="--")
    ax1.set_title("Predicted vs Actual (first 200 samples)")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Normalized Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.98, 0.05, f"MSE = {mse:.5f}\nRMSE = {rmse:.5f}",
             transform=ax1.transAxes, ha="right", va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), fontsize=9)

    # Training curves
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(train_losses, label="Train MSE", color="#4CAF50")
    ax2.plot(val_losses,   label="Val MSE",   color="#FF5722")
    ax2.set_title("Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/05_evaluation.png", dpi=150)
    plt.close()
    print("  Saved: figures/05_evaluation.png")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    # ── Task 1
    raw_df = fetch_data(TICKERS, START, END)
    norm_df, scaler = normalize(raw_df)
    norm_df.columns = TICKERS
    plot_timeseries(norm_df, NAMES)
    print("  Task 1 complete.")

    # ── Task 2
    print("\n[Task 2] Signal processing...")
    plot_signal_analysis(norm_df, NAMES)
    print("  Task 2 complete.")

    # ── Task 3
    print("\n[Task 3] Building dataset and training CNN...")
    X, y = build_spectrogram_dataset(norm_df, WINDOW_LEN, HOP_SIZE, PRED_STEPS)

    # Train / Val split (80/20)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = SpectrogramDataset(X_train, y_train)
    val_ds   = SpectrogramDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = SpectrogramCNN().to(DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), "cnn_stock_model.pth")
    print("  Model saved: cnn_stock_model.pth")

    plot_cnn_architecture()
    print("  Task 3 complete.")

    # ── Task 4
    evaluate_and_plot(model, val_loader, train_losses, val_losses)
    print("\n✓ All tasks complete. Check the figures/ folder.")