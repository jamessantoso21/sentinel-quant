@echo off
echo ============================================
echo   Sentinel Quant - Model Training Pipeline
echo   GPU: RTX 4060 (8GB VRAM)
echo ============================================
echo.

cd /d "%~dp0"

echo [Step 1/4] Installing training dependencies...
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3 gymnasium pandas numpy httpx tqdm
echo.

echo [Step 2/4] Downloading BTC historical data (2 years)...
echo This will take about 5 minutes...
echo.
python download_data.py
echo.

echo [Step 3/4] Training LSTM model (30-60 minutes)...
echo.
python train_lstm.py
echo.

echo [Step 4/4] Training PPO model (1-2 hours)...
echo.
python train_ppo.py
echo.

echo ============================================
echo   Training Complete!
echo ============================================
echo.
echo Models saved in: backend/models/
echo   - lstm_model.pt
echo   - ppo_model.zip
echo.
echo Next: Push to Railway to activate in voting system!
echo.
pause
