# Sentinel Quant - Railway Deployment

## Services
This project deploys the following services:
- **Backend API** (FastAPI + Python)
- **PostgreSQL** (Railway plugin)
- **Redis** (Railway plugin)

## Environment Variables
Set these in Railway dashboard:

```
SECRET_KEY=your-production-secret-key-here
BINANCE_API_KEY=your-binance-testnet-key
BINANCE_API_SECRET=your-binance-testnet-secret
BINANCE_TESTNET=true
TRADING_ENABLED=true
CONFIDENCE_THRESHOLD=0.70
MAX_POSITION_SIZE_USDT=100.0
MAX_DAILY_LOSS_PERCENT=5.0
```

## Deploy Steps
1. Push to GitHub
2. Connect Railway to GitHub repo
3. Add PostgreSQL plugin
4. Add Redis plugin
5. Set environment variables
6. Deploy!
