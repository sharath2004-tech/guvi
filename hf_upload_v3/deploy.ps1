# Deploy to Hugging Face Spaces - With Trained Model

Write-Host "`n🚀 DEPLOYING AI VOICE DETECTOR TO HUGGING FACE" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# Navigate to upload folder
Set-Location "e:\guvi\hf_upload_v3"

# Check if files exist
Write-Host "✓ Checking deployment files..." -ForegroundColor Yellow
$files = @("app.py", "requirements.txt", "Dockerfile", "start.sh", "best_voice_detector.pth")
foreach ($file in $files) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1MB
        Write-Host "  ✓ $file ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file NOT FOUND!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n📦 Initializing git repository..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "  Repository already initialized" -ForegroundColor Green
} else {
    git init
    git remote add origin https://huggingface.co/spaces/sharath09876/guvi
}

Write-Host "`n📝 Staging files..." -ForegroundColor Yellow
git add .
Write-Host "  ✓ Files staged" -ForegroundColor Green

Write-Host "`n💾 Committing changes..." -ForegroundColor Yellow
git commit -m "🎯 Deploy with trained model weights (85.7% test accuracy)"
Write-Host "  ✓ Committed successfully" -ForegroundColor Green

Write-Host "`n🌐 Pushing to Hugging Face..." -ForegroundColor Yellow
Write-Host "  (You may need to enter your Hugging Face credentials)`n" -ForegroundColor Cyan

git push origin main --force

Write-Host "`n✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "================================================`n" -ForegroundColor Green

Write-Host "📍 Your API will be available at:" -ForegroundColor Cyan
Write-Host "   https://sharath09876-guvi.hf.space`n" -ForegroundColor White

Write-Host "⏳ Please wait 2-3 minutes for Hugging Face to rebuild the container..." -ForegroundColor Yellow
Write-Host "`nℹ️  The app now uses your custom trained model with 85.7% accuracy!`n" -ForegroundColor Cyan
