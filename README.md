SREE LEKSHMI H
University Reg.No:TCR24CS067
# Pattern Recognition for Financial Time Series Forecasting

## Stocks Used
| Company | NSE Ticker |
|---|---|
| Reliance Industries | RELIANCE.NS |
| Tata Consultancy Services | TCS.NS |
| Infosys | INFY.NS |

## Setup


# 1. Create virtual environment 
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py


## Output Files

| File | Task | Description |
|---|---|---|
| `figures/01_timeseries.png` | Task 1 | Normalized stock price time series |
| `figures/02_fft_spectrum.png` | Task 2 | Frequency spectrum (FFT) |
| `figures/03_spectrograms.png` | Task 2 | STFT spectrograms for all 3 stocks |
| `figures/04_cnn_architecture.png` | Task 3 | CNN architecture diagram |
| `figures/05_evaluation.png` | Task 4 | Predicted vs actual + loss curves |
| `cnn_stock_model.pth` | Task 3 | Saved CNN model weights |

## Pipeline Overview


NSE Stock Data (yfinance)
        │
        ▼
  Normalize (Min-Max)          ← Task 1
        │
        ▼
  FFT + STFT Spectrogram       ← Task 2
  [1 × 16 × 16 image per sample]
        │
        ▼
  CNN Regression Model         ← Task 3
  (Conv → BN → ReLU → Pool) × 3
        │
        ▼
  Predicted Price (t + 5 days) ← Task 4
  Evaluated with MSE / RMSE

## CNN Architecture Summary

```
Input        [1 × 16 × 16]
Conv2d(16)   [16 × 16 × 16]  + BN + ReLU
MaxPool      [16 × 8 × 8]
Conv2d(32)   [32 × 8 × 8]   + BN + ReLU
MaxPool      [32 × 4 × 4]
Conv2d(64)   [64 × 4 × 4]   + BN + ReLU
AvgPool      [64 × 2 × 2]
FC(256→128)  + ReLU + Dropout(0.3)
FC(128→32)   + ReLU
FC(32→1)     → predicted price
```

## Notes
- Data range: 2018–2024 (~1500 trading days)
- Prediction horizon: 5 days ahead
- STFT window: 30 days, hop: 5 days
- Training: 80% | Validation: 20%
- Epochs: 30, LR: 1e-3 with step decay
