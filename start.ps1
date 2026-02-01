# AI Voice Detection API - Quick Start Script
# Run this script to start your API server

Write-Host "🚀 Starting AI Voice Detection API..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    & .\.venv\Scripts\Activate.ps1
}

# Install/Update dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Your API will be available at:" -ForegroundColor Cyan
Write-Host "   http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation available at:" -ForegroundColor Cyan
Write-Host "   http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "🔑 API Key:" -ForegroundColor Cyan
Write-Host "   sk_live_abc123xyz789_secure_key_2024" -ForegroundColor White
Write-Host ""
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host ""

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
